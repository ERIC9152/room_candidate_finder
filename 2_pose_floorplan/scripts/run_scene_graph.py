"""Run a trained CoVisPose checkpoint over the tour scenes in ``data/group``.

For every scene it predicts a relative pose for each panorama pair the tour's
hotspots link, solves one global pose per panorama, turns the predicted floor
boundaries into room polygons, and writes everything into
``<scene>/covispose/``:

    edges.json          per-pair relative poses (all three estimators)
    poses.json          the solved global pose per panorama
    direct_preview.json root / used_edges / world_polys / world_transforms,
                        the same schema the existing preview_tmp output uses
    layout_local.json   per-panorama room polygon in its own frame
    boundary.json       the dense per-column floor boundary, unsimplified
    floorplan.png       the assembled plan
    report.json         scene statistics, residuals, and -- where the scene has
                        an existing result to compare against -- the rigid
                        alignment error against it

    python scripts/run_scene_graph.py --ckpt ckpt/server_r50_c1024/best.pt \
        --data-root /path/to/data --tta-rolls 4

Co-visibility gating (models trained with negatives, configs/server_c10neg.yaml):
``--min-covis-frac 0.10`` drops every edge whose predicted co-visible fraction
is below 0.10 before solving. Such a model says "nothing in common" when it
means it; on ZInD's door-adjacent pairs, edges it calls < 10 % co-visible have
a pose within 10 deg only 6-12 % of the time (docs/PAPER_COMPARISON.md section
5). The graph may then split into components -- each is solved in its own
frame, and plotted side by side, since nothing relates them.

    python scripts/run_scene_graph.py --ckpt ckpt/server_c10neg/epoch_015.pt \
        --tta-rolls 4 --tta-flip --min-covis-frac 0.10 --out-name covispose_c10neg \
        --summary covispose_c10neg_scenes.json

Units are this dataset's world units (one camera height = 50, from the
``z0`` in every ``layout/<pano>.json``), so the output is directly comparable
with ``preview_tmp/*.json``.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.evaluate import forward_tta, load_model_from_ckpt
from covispose.scene_graph import (
    UNITS_PER_CAMERA_HEIGHT, align_rigid, boundary_to_polygon, edge_residuals,
    global_poses, load_scene, simplify_polygon, to_world, transform_matrix,
)


def load_pano(path: str, h: int, w: int) -> torch.Tensor:
    """Decode one panorama to the network's input size."""
    with Image.open(path) as im:
        im.draft("RGB", (w, h))      # 8192x4096 -> 1024x512 in the DCT domain
        im = im.convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        a = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().div_(255.0)


@torch.no_grad()
def predict_edges(model, scene, device, batch_size=4, tta_rolls=1, tta_flip=False,
                  img_h=512, img_w=1024, progress=True):
    """Run the network over every edge. Returns (edge records, mean bon per pano)."""
    cache = {}

    def img(pid):
        if pid not in cache:
            cache[pid] = load_pano(scene.panos[pid], img_h, img_w)
        return cache[pid]

    recs = []
    bon_sum, bon_n = {}, {}
    for s in range(0, len(scene.edges), batch_size):
        chunk = scene.edges[s:s + batch_size]
        a = torch.stack([img(x[0]) for x in chunk]).to(device)
        b = torch.stack([img(x[1]) for x in chunk]).to(device)
        out = forward_tta(model, a, b, tta_rolls, tta_flip)

        t = out["t"].float().cpu().numpy()
        yaw = model.decode_angle(out["yaw_vec"].float()).cpu().numpy()
        cov_a = torch.sigmoid(out["covis_logit_a"].float()).cpu().numpy()
        cov_b = torch.sigmoid(out["covis_logit_b"].float()).cpu().numpy()
        bon_a = out["bon_a"].float().cpu().numpy()
        bon_b = out["bon_b"].float().cpu().numpy()
        has_geo = "geo_t" in out
        if has_geo:
            g_t = out["geo_t"].float().cpu().numpy()
            g_yaw = model.decode_angle(out["geo_yaw_vec"].float()).cpu().numpy()

        for i, (pa, pb) in enumerate(chunk):
            for pid, bon in ((pa, bon_a[i]), (pb, bon_b[i])):
                bon_sum[pid] = bon_sum.get(pid, 0.0) + bon
                bon_n[pid] = bon_n.get(pid, 0) + 1
            rec = {
                "a": pa, "b": pb,
                "decoder": {"t": t[i].tolist(), "yaw": float(yaw[i])},
                "covis_mean": float((cov_a[i].mean() + cov_b[i].mean()) / 2),
                # the fraction of columns the head calls co-visible -- the
                # quantity the gate thresholds (same as evaluate_split's)
                "covis_frac": float(((cov_a[i] > 0.5).mean() + (cov_b[i] > 0.5).mean()) / 2),
            }
            if has_geo:
                rec["geometric"] = {"t": g_t[i].tolist(), "yaw": float(g_yaw[i])}
                dt = float(np.linalg.norm(g_t[i] - t[i]))
                dy = abs(math.atan2(math.sin(g_yaw[i] - yaw[i]), math.cos(g_yaw[i] - yaw[i])))
                rec["agreement"] = {"t_diff": dt, "yaw_diff_deg": math.degrees(dy)}
            recs.append(rec)
        if progress:
            print(f"    {min(s + batch_size, len(scene.edges))}/{len(scene.edges)} edges",
                  flush=True)

    bon = {k: bon_sum[k] / bon_n[k] for k in bon_sum}
    return recs, bon


def edge_confidence(rec: dict) -> float:
    """How much the solver should trust this edge.

    Two independent signals: how much of the pair the co-visibility head says is
    shared, and how far the decoder and the geometric estimator disagree. They
    fail differently, so a pair both like is a pair worth building on.
    """
    conf = max(float(rec["covis_mean"]), 1e-3)
    if "agreement" in rec:
        a = rec["agreement"]
        conf *= math.exp(-a["t_diff"] / 0.5) * math.exp(-a["yaw_diff_deg"] / 20.0)
    return max(conf, 1e-4)


def compare_with_existing(scene_dir, theta, c, scale, nodes=None):
    """Rigidly align our camera centres onto the existing pipeline's, if present.

    ``nodes`` restricts the alignment to one connected component: separate
    components have no relative pose, so aligning them jointly is meaningless.
    """
    p = os.path.join(scene_dir, "preview_tmp", "direct_preview.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        wt = d.get("world_transforms", {})
        common = [k for k in wt if k in theta and (nodes is None or k in nodes)]
        if len(common) < 3:
            return None
        ours = np.array([c[k] for k in common])
        theirs = np.array([[wt[k][0][2], wt[k][1][2]] for k in common])
        a = align_rigid(ours, theirs)
        a["n_common"] = len(common)
        a["rms_camera_heights"] = a["rms"] / scale
        return a
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def component_offsets(c, polys, comp_id, gap):
    """Place connected components side by side for plotting.

    Each component is solved in its own frame, rooted at the origin, so drawn as
    is they would pile up on top of each other. Largest component first, at
    its own coordinates; the rest to its right, in decreasing size.
    """
    members = {}
    for n, k in comp_id.items():
        members.setdefault(k, []).append(n)
    order = sorted(members, key=lambda k: -len(members[k]))
    off, x_next = {}, None
    for k in order:
        pts = np.vstack([np.asarray(c[n]).reshape(1, 2) for n in members[k]]
                        + [polys[n] for n in members[k] if n in polys])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if x_next is None:
            off[k] = np.zeros(2)
            x_next = hi[0] + gap
        else:
            off[k] = np.array([x_next - lo[0], 0.0])
            x_next += (hi[0] - lo[0]) + gap
    return {n: off[comp_id[n]] for n in comp_id}


def plot_floorplan(path, scene, theta, c, polys, comp_id, title, edges=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shift = component_offsets(c, polys, comp_id, gap=2.0 * UNITS_PER_CAMERA_HEIGHT)
    c = {n: np.asarray(c[n]) + shift[n] for n in c}
    polys = {n: p + shift[n] for n, p in polys.items()}
    if edges is None:
        edges = scene.edges

    fig, ax = plt.subplots(figsize=(11, 11))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (pid, poly) in enumerate(sorted(polys.items())):
        col = colors[i % 20]
        ring = np.vstack([poly, poly[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=col, lw=1.4, alpha=0.9)
        ax.fill(ring[:, 0], ring[:, 1], color=col, alpha=0.08)
        ax.plot(*c[pid], "o", color=col, ms=6, mec="k", mew=0.5)
        ax.annotate(f"{scene.room_idx.get(pid, '')}", c[pid], fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
        d = 0.6 * UNITS_PER_CAMERA_HEIGHT
        ax.plot([c[pid][0], c[pid][0] - d * math.sin(theta[pid])],
                [c[pid][1], c[pid][1] + d * math.cos(theta[pid])], color=col, lw=1.0)
    for a, b in edges:
        ax.plot([c[a][0], c[b][0]], [c[a][1], c[b][1]], "k-", lw=0.6, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("world units (1 camera height = 50)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_scene(model, mcfg, scene_dir, args, device):
    scene = load_scene(scene_dir)
    out_dir = os.path.join(scene_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {scene.scene_id}: {len(scene.panos)} panos, {len(scene.edges)} edges")
    if not scene.edges:
        print("    no edges -- nothing to assemble")
        return {"scene": scene.scene_id, "status": "no_edges"}

    t0 = time.time()
    recs, bon = predict_edges(model, scene, device, args.batch_size,
                              args.tta_rolls, args.tta_flip, mcfg.img_h, mcfg.img_w)
    scale = args.units_per_camera_height

    # ---- measurements for the solver, in world units --------------------- #
    est = args.estimator
    # This dataset's floor-plan frame is the mirror of the one the network
    # predicts in, reflected about the x axis: y and yaw both negate. Measured
    # against two independent references -- the tour's own hotspot directions
    # (all 42 scenes) and the existing pipeline's assembled poses (19 scenes) --
    # over the four candidate frame changes:
    #
    #   candidate     hotspot dir median   ref yaw median   ref |t| median
    #   identity          95.5 deg            91.6 deg        2.85 heights
    #   negate y          30.3 deg            44.9 deg        0.89 heights  <-
    #   negate x         149.7 deg            44.9 deg        3.80 heights
    #   rotate 180        84.5 deg            91.6 deg        2.18 heights
    #
    # Note yaw alone cannot tell the two mirrors apart -- it negates in both.
    # A mirrored world is still a valid rigid world, so this only re-expresses
    # the same prediction in the dataset's frame.
    mir = -1.0 if args.world_mirror else 1.0
    edges = []
    for r in recs:
        src = r.get(est, r["decoder"])
        r["confidence"] = edge_confidence(r)
        r["kept"] = r["covis_frac"] >= args.min_covis_frac
        if not r["kept"]:
            continue
        edges.append({"a": r["a"], "b": r["b"],
                      "t": np.array([src["t"][0], mir * src["t"][1]]) * scale,
                      "yaw": mir * float(src["yaw"]),
                      "confidence": r["confidence"]})

    theta, c, used, ignored, comp_id, roots = global_poses(
        list(scene.panos), edges, refine=not args.no_refine)
    res = edge_residuals(edges, theta, c)

    # ---- polygons -------------------------------------------------------- #
    polys_local, polys_world, transforms = {}, {}, {}
    for pid in scene.panos:
        if pid not in bon:
            continue
        p = boundary_to_polygon(bon[pid], max_dist=args.max_room_dist) * scale
        p = p * np.array([1.0, mir])            # same mirror as the poses
        p = simplify_polygon(p, args.simplify_tol * scale)
        polys_local[pid] = p
        polys_world[pid] = to_world(p, theta[pid], c[pid])
        transforms[pid] = transform_matrix(theta[pid], c[pid])

    # ---- write ----------------------------------------------------------- #
    def dump(name, obj):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    dump("edges.json", {
        "scene_id": scene.scene_id,
        "units": "world units (1 camera height = %.1f)" % scale,
        "estimator_used": est,
        "world_mirror": bool(args.world_mirror),
        "edges": [{
            "i": r["a"], "j": r["b"],
            "measurement": {"dx": r[est if est in r else "decoder"]["t"][0] * scale,
                            "dy": mir * r[est if est in r else "decoder"]["t"][1] * scale,
                            "dtheta": mir * r[est if est in r else "decoder"]["yaw"]},
            "convention": "p_j = Rot(dtheta) p_i + (dx, dy)",
            "confidence": r["confidence"],
            "covis_mean": r["covis_mean"],
            "covis_frac": r["covis_frac"],
            "kept": r["kept"],
            "estimators": {k: {"t_world": [v["t"][0] * scale, v["t"][1] * scale],
                               "yaw_rad": v["yaw"]}
                           for k, v in r.items() if k in ("decoder", "geometric")},
            "agreement": r.get("agreement"),
        } for r in recs],
    })
    dump("poses.json", {
        "scene_id": scene.scene_id,
        "units": "world units (1 camera height = %.1f)" % scale,
        "convention": "p_world = Rot(theta) p_local + (x, y)",
        "roots": {str(k): v for k, v in roots.items()},
        "poses": {pid: {"x": float(c[pid][0]), "y": float(c[pid][1]),
                        "theta_rad": float(theta[pid]),
                        "theta_deg": math.degrees(theta[pid]),
                        "component": int(comp_id[pid]),
                        "room_idx": scene.room_idx.get(pid, -1)}
                  for pid in sorted(scene.panos)},
    })
    dump("direct_preview.json", {
        "root": roots.get(0),
        "used_edges": [{"from": e["a"], "to": e["b"]} for e in used],
        "ignored_loop_edges": ignored,
        "world_polys": {k: v.tolist() for k, v in polys_world.items()},
        "world_transforms": transforms,
    })
    dump("layout_local.json", {k: v.tolist() for k, v in polys_local.items()})
    # The dense per-column boundary, before any simplification: wall fitting and
    # the multi-view bundle adjustment need every column, not a polygon whose
    # vertices have already been chosen for us.
    dump("boundary.json", {
        "units": "HorizonNet bon, radians, one value per column",
        "n_cols": int(mcfg.n_cols),
        "world_mirror": bool(args.world_mirror),
        "note": "local frame, NOT mirrored; apply the scene mirror when going to world",
        "bon": {k: np.asarray(v).round(6).tolist() for k, v in bon.items()},
    })

    sizes = {}
    for k in comp_id.values():
        sizes[k] = sizes.get(k, 0) + 1
    main_comp = max(sizes, key=sizes.get)
    comparison = compare_with_existing(scene_dir, theta, c, scale,
                                       nodes={n for n, k in comp_id.items() if k == main_comp})
    t_res = [r["t_residual"] for r in res] or [float("nan")]
    y_res = [abs(r["yaw_residual_deg"]) for r in res] or [float("nan")]
    report = {
        "scene": scene.scene_id,
        "status": "ok",
        "n_panos": len(scene.panos),
        "n_edges": len(scene.edges),
        "n_components": len(set(comp_id.values())),
        "covis_gate": {
            "min_covis_frac": args.min_covis_frac,
            "n_edges_kept": len(edges),
            "n_isolated_panos": sum(1 for v in sizes.values() if v == 1),
            "largest_component": sizes[main_comp],
        },
        "estimator": est,
        "world_mirror": bool(args.world_mirror),
        "tta": {"rolls": args.tta_rolls, "flip": args.tta_flip},
        "units_per_camera_height": scale,
        "loop_closure_residuals": {
            "t_median": float(np.median(t_res)), "t_p90": float(np.percentile(t_res, 90)),
            "t_median_camera_heights": float(np.median(t_res) / scale),
            "yaw_median_deg": float(np.median(y_res)),
            "yaw_p90_deg": float(np.percentile(y_res, 90)),
        },
        "covis_mean": float(np.mean([r["covis_mean"] for r in recs])),
        "agreement_decoder_vs_geometric": {
            "t_median": float(np.median([r["agreement"]["t_diff"] for r in recs])),
            "yaw_median_deg": float(np.median([r["agreement"]["yaw_diff_deg"] for r in recs])),
        } if "agreement" in recs[0] else None,
        "vs_existing_preview": comparison,
        "seconds": time.time() - t0,
    }
    dump("report.json", report)
    dump("edge_residuals.json", res)

    plot_floorplan(os.path.join(out_dir, "floorplan.png"), scene, theta, c, polys_world,
                   comp_id, f"{scene.scene_id} -- CoVisPose ({est}"
                            f"{', TTA' if args.tta_rolls > 1 or args.tta_flip else ''}"
                            + (f", covis gate {args.min_covis_frac:.2f}: {len(edges)}/{len(recs)} edges,"
                               f" {report['n_components']} components"
                               if args.min_covis_frac > 0 else "")
                            + ")",
                   edges=[(e["a"], e["b"]) for e in edges])

    print(f"    edges kept {len(edges)}/{len(recs)}, "
          f"components {report['n_components']}, loop residual median "
          f"{report['loop_closure_residuals']['t_median']:.1f} units "
          f"({report['loop_closure_residuals']['t_median_camera_heights']:.3f} camera heights), "
          f"yaw {report['loop_closure_residuals']['yaw_median_deg']:.2f} deg")
    if comparison and "rms" in comparison:
        print(f"    vs existing preview: rms {comparison['rms']:.1f} units over "
              f"{comparison['n_common']} panos, mirror={comparison['mirror']}")
    print(f"    -> {out_dir}  ({report['seconds']:.0f}s)")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/server_r50_c1024/best.pt")
    ap.add_argument("--data-root", default=os.environ.get("SCENE_ROOT", r"C:\data"),
                    help="the directory that contains group/<scene>/manifest.json")
    ap.add_argument("--scenes", default="", help="comma-separated scene ids (default: all)")
    ap.add_argument("--estimator", choices=["decoder", "geometric"], default="decoder",
                    help="which relative pose feeds the solver; both are always recorded")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tta-rolls", type=int, default=1)
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--weights", choices=["auto", "ema", "raw"], default="auto")
    ap.add_argument("--world-mirror", dest="world_mirror", action="store_true", default=True,
                    help="mirror predictions into this dataset's world frame (default on; "
                         "its panoramas run the opposite way round from ZInD, measured)")
    ap.add_argument("--no-world-mirror", dest="world_mirror", action="store_false")
    ap.add_argument("--no-refine", action="store_true",
                    help="stop at the spanning tree, skip the global least squares")
    ap.add_argument("--simplify-tol", type=float, default=0.05,
                    help="polygon simplification tolerance, in camera heights")
    ap.add_argument("--max-room-dist", type=float, default=20.0,
                    help="clamp for the predicted floor distance, in camera heights")
    ap.add_argument("--units-per-camera-height", type=float, default=UNITS_PER_CAMERA_HEIGHT)
    ap.add_argument("--summary", default="covispose_scenes.json",
                    help="written under --data-root")
    ap.add_argument("--min-covis-frac", type=float, default=0.0,
                    help="drop edges whose predicted co-visible fraction is below this "
                         "(0 = keep all; only meaningful for models trained with negatives)")
    ap.add_argument("--out-name", default="covispose",
                    help="per-scene output folder name, so runs can sit side by side")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, blob = load_model_from_ckpt(args.ckpt, device, args.weights)
    print(f"checkpoint : {args.ckpt} (epoch {blob.get('epoch')}, {blob['loaded_weights']} weights)")
    print(f"input      : {mcfg.img_h}x{mcfg.img_w}, {mcfg.n_cols} columns, geo_pose={mcfg.geo_pose}")
    if args.estimator == "geometric" and not mcfg.geo_pose:
        raise SystemExit("this checkpoint has no geometric pose branch")

    group = os.path.join(args.data_root, "group")
    wanted = [s for s in args.scenes.split(",") if s] or sorted(
        d for d in os.listdir(group)
        if os.path.isfile(os.path.join(group, d, "manifest.json")))
    print(f"scenes     : {len(wanted)}")

    reports = []
    for s in wanted:
        reports.append(run_scene(model, mcfg, os.path.join(group, s), args, device))

    ok = [r for r in reports if r.get("status") == "ok"]
    path = os.path.join(args.data_root, args.summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "estimator": args.estimator,
                   "min_covis_frac": args.min_covis_frac, "out_name": args.out_name,
                   "tta": {"rolls": args.tta_rolls, "flip": args.tta_flip},
                   "scenes": reports}, f, ensure_ascii=False, indent=1)
    print(f"\n{len(ok)}/{len(reports)} scenes assembled -> {path}")
    if ok:
        t = [r["loop_closure_residuals"]["t_median_camera_heights"] for r in ok]
        y = [r["loop_closure_residuals"]["yaw_median_deg"] for r in ok]
        print(f"loop-closure residual, median over scenes: {np.median(t):.3f} camera heights, "
              f"{np.median(y):.2f} deg")
        cmp_ = [r["vs_existing_preview"] for r in ok
                if r.get("vs_existing_preview") and "rms" in r["vs_existing_preview"]]
        if cmp_:
            print(f"vs existing preview ({len(cmp_)} scenes): rms median "
                  f"{np.median([x['rms'] for x in cmp_]):.1f} units, "
                  f"mirror needed in {sum(x['mirror'] for x in cmp_)}/{len(cmp_)}")


if __name__ == "__main__":
    main()
