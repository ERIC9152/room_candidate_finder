"""Visualisation helpers: ground-truth verification and prediction overlays."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .geometry import b_to_row, col_centers_theta, theta_to_col
from .targets import PairTargets
from .zind_parse import PanoRecord

# Matplotlib is imported lazily so that training never pays for it.


def load_pano(path: str, width: int = 1024, height: int = 512) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((width, height), Image.BILINEAR))


def _hue_strip(values: np.ndarray, mask: np.ndarray, out_w: int, strip_h: int = 18) -> np.ndarray:
    """Colour bar: hue = angle in [-pi, pi), grey where ``mask`` is False."""
    import matplotlib.cm as cm

    hue = (np.asarray(values) / (2 * math.pi) + 0.5) % 1.0
    rgb = cm.get_cmap("hsv")(hue)[:, :3]
    rgb[~np.asarray(mask)] = 0.22
    # nearest-neighbour resize from n_cols to out_w
    idx = np.floor(np.arange(out_w) * len(hue) / out_w).astype(int)
    return np.repeat(rgb[idx][None, :, :], strip_h, axis=0)


def _draw_boundary(ax, bon: np.ndarray, width: int, height: int, **kw):
    n = len(bon)
    x = (np.arange(n) + 0.5) * width / n
    y = b_to_row(bon, height)
    ax.plot(x, y, **kw)


def plot_prediction_vs_gt(
    rec_a: PanoRecord,
    rec_b: PanoRecord,
    tg: PairTargets,
    pred: dict,
    out_path: str = "pred.png",
    img_w: int = 1024,
    img_h: int = 512,
    title_extra: str = "",
):
    """Qualitative check: predicted vs ground-truth per-column outputs.

    ``pred`` holds numpy arrays for one pair:
    ``covis_a/b`` (probabilities), ``corr_a/b`` (radians), ``bon_a/b``
    (radians), ``t`` (2,), ``yaw`` (float).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_a = load_pano(rec_a.image_path, img_w, img_h)
    img_b = load_pano(rec_b.image_path, img_w, img_h)
    n = tg.n_cols
    x = (np.arange(n) + 0.5) * img_w / n

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 0.62, 0.62, 0.62],
                          hspace=0.42, wspace=0.18)

    # --- the two panos with GT vs predicted boundary --------------------- #
    for row, (img, rec, bon_gt, bon_pr, cov_gt, cov_pr, tag) in enumerate([
        (img_a, rec_a, tg.bon_a, pred["bon_a"], tg.covis_a, pred["covis_a"], "A"),
        (img_b, rec_b, tg.bon_b, pred["bon_b"], tg.covis_b, pred["covis_b"], "B"),
    ]):
        ax = fig.add_subplot(gs[row, :])
        ax.imshow(img)
        ax.plot(x, b_to_row(bon_gt, img_h), "-", color="#00e5ff", lw=2.0, label="GT boundary")
        ax.plot(x, b_to_row(bon_pr, img_h), "--", color="#ffd400", lw=1.8, label="predicted")
        # co-visibility ribbons along the top
        ax.fill_between(x, 4, 16, where=cov_gt.astype(bool), color="#00e5ff", alpha=0.85,
                        step="mid", label="GT co-visible")
        ax.fill_between(x, 20, 32, where=(cov_pr > 0.5), color="#ffd400", alpha=0.85,
                        step="mid", label="predicted co-visible")
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"pano {tag}: {rec.uid}", fontsize=9)
        ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.75)

    # --- boundary curves -------------------------------------------------- #
    for col, (bon_gt, bon_pr, tag) in enumerate([
        (tg.bon_a, pred["bon_a"], "A"), (tg.bon_b, pred["bon_b"], "B")
    ]):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(np.degrees(bon_gt), color="#0077b6", lw=1.6, label="GT")
        ax.plot(np.degrees(bon_pr), color="#e07b00", lw=1.4, ls="--", label="pred")
        err = np.degrees(np.abs(bon_pr - bon_gt)).mean()
        ax.set_title(f"floor boundary {tag} (mean err {err:.2f} deg)", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("deg below horizon")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # --- co-visibility ---------------------------------------------------- #
    for col, (cov_gt, cov_pr, tag) in enumerate([
        (tg.covis_a, pred["covis_a"], "A"), (tg.covis_b, pred["covis_b"], "B")
    ]):
        ax = fig.add_subplot(gs[3, col])
        ax.fill_between(np.arange(len(cov_gt)), cov_gt.astype(float), color="#0077b6",
                        alpha=0.35, step="mid", label="GT")
        ax.plot(cov_pr, color="#e07b00", lw=1.5, label="predicted prob")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        acc = ((cov_pr > 0.5) == cov_gt.astype(bool)).mean()
        ax.set_title(f"co-visibility {tag} (acc {acc:.3f})", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # --- correspondence --------------------------------------------------- #
    for col, (corr_gt, corr_pr, cov_gt, tag) in enumerate([
        (tg.corr_a2b, pred["corr_a"], tg.covis_a, "A->B"),
        (tg.corr_b2a, pred["corr_b"], tg.covis_b, "B->A"),
    ]):
        ax = fig.add_subplot(gs[4, col])
        m = cov_gt.astype(bool)
        idx = np.arange(len(corr_gt))
        ax.plot(idx[m], np.degrees(corr_gt[m]), ".", ms=3, color="#0077b6", label="GT (co-visible)")
        ax.plot(idx[m], np.degrees(corr_pr[m]), ".", ms=3, color="#e07b00", label="predicted")
        if m.any():
            d = np.degrees(np.abs(np.arctan2(np.sin(corr_pr[m] - corr_gt[m]),
                                             np.cos(corr_pr[m] - corr_gt[m])))).mean()
        else:
            d = float("nan")
        ax.set_title(f"correspondence {tag} (mean err {d:.2f} deg, co-visible only)", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("azimuth in other pano (deg)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    t_err = float(np.linalg.norm(np.asarray(pred["t"]) - tg.t_ab))
    yaw_err = abs(math.degrees(math.atan2(math.sin(pred["yaw"] - tg.yaw_ab),
                                          math.cos(pred["yaw"] - tg.yaw_ab))))
    fig.suptitle(
        f"prediction vs ground truth{('  --  ' + title_extra) if title_extra else ''}\n"
        f"pose GT t=({tg.t_ab[0]:+.3f}, {tg.t_ab[1]:+.3f}) yaw={math.degrees(tg.yaw_ab):+.1f}deg   |   "
        f"pred t=({pred['t'][0]:+.3f}, {pred['t'][1]:+.3f}) yaw={math.degrees(pred['yaw']):+.1f}deg   |   "
        f"|dt|={t_err:.3f}  dyaw={yaw_err:.1f}deg",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pair_targets(
    rec_a: PanoRecord,
    rec_b: PanoRecord,
    tg: PairTargets,
    covis_score: Optional[float] = None,
    out_path: str = "pair.png",
    img_w: int = 1024,
    img_h: int = 512,
    n_rays: int = 24,
):
    """Four-panel ground-truth check for one pair.

    Panels 1 & 2: the two panos with the per-column floor-wall boundary drawn
    on top, a co-visibility mask bar, and a *correspondence hue bar*.  The hue
    bar for pano A is coloured by the azimuth each column maps to in pano B;
    the bar for pano B is coloured by B's own azimuth.  If the correspondence
    ground truth is right, the same physical wall gets the same colour in both
    bars, so the colour pattern over co-visible regions must line up with
    matching scene content.

    Panel 3: top-down view, both room polygons brought into B's frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_a = load_pano(rec_a.image_path, img_w, img_h)
    img_b = load_pano(rec_b.image_path, img_w, img_h)
    n = tg.n_cols
    theta_cols = col_centers_theta(n)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.28, wspace=0.15)

    for row, (img, rec, bon, covis, hue_vals, tag) in enumerate(
        [
            (img_a, rec_a, tg.bon_a, tg.covis_a, tg.corr_a2b, "A"),
            (img_b, rec_b, tg.bon_b, tg.covis_b, theta_cols, "B"),
        ]
    ):
        ax = fig.add_subplot(gs[row, :])
        strip = _hue_strip(hue_vals, covis, img_w, strip_h=20)
        covis_bar = np.repeat(
            np.where(covis, 1.0, 0.0)[np.floor(np.arange(img_w) * n / img_w).astype(int)][None, :, None]
            * np.array([[[0.1, 0.9, 0.35]]]),
            12,
            axis=0,
        )
        pad = np.ones((4, img_w, 3))
        canvas = np.concatenate(
            [strip, pad, covis_bar, pad, img.astype(np.float64) / 255.0], axis=0
        )
        ax.imshow(canvas)
        off = strip.shape[0] + 4 + covis_bar.shape[0] + 4
        x = (np.arange(n) + 0.5) * img_w / n
        y = b_to_row(bon, img_h) + off
        ax.plot(x, y, "-", color="#00e5ff", lw=1.4, label="GT floor-wall boundary")
        yc = y.copy()
        yc[~covis] = np.nan
        ax.plot(x, yc, "-", color="#ff2d55", lw=2.6, label="co-visible")
        ax.set_xlim(0, img_w)
        ax.set_ylim(canvas.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"pano {tag}: {rec.uid}   [{rec.layout_source}, {'primary' if rec.is_primary else 'secondary'}]"
            f"   covis={covis.mean():.2f}"
            + ("   (top bar: hue = matching azimuth in B)" if tag == "A" else "   (top bar: hue = own azimuth)"),
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # ---- top-down panel -------------------------------------------------- #
    ax = fig.add_subplot(gs[2, 0])
    pa_in_b = tg.pts_a_in_b
    pb = np.stack(
        [-np.sin(theta_cols), np.cos(theta_cols)], axis=1
    ) * (1.0 / np.tan(np.clip(tg.bon_b, 1e-4, math.pi / 2 - 1e-6)))[:, None]

    ax.plot(pb[:, 0], pb[:, 1], ".", ms=2.0, color="#1f77b4", label="B boundary (B frame)")
    ax.plot(pa_in_b[:, 0], pa_in_b[:, 1], ".", ms=2.0, color="#ff7f0e", label="A boundary -> B frame")
    ax.plot(pa_in_b[tg.covis_a, 0], pa_in_b[tg.covis_a, 1], ".", ms=4.5, color="#d62728",
            label="A co-visible")
    ax.plot([0], [0], "s", color="#1f77b4", ms=9, label="camera B")
    ax.plot([tg.t_ab[0]], [tg.t_ab[1]], "s", color="#ff7f0e", ms=9, label="camera A")
    # heading of A drawn in B's frame
    hd = np.array([-math.sin(tg.yaw_ab), math.cos(tg.yaw_ab)])
    ax.arrow(tg.t_ab[0], tg.t_ab[1], hd[0] * 0.6, hd[1] * 0.6, color="#ff7f0e",
             head_width=0.12, length_includes_head=True)
    ax.arrow(0, 0, 0, 0.6, color="#1f77b4", head_width=0.12, length_includes_head=True)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(
        f"top-down (B frame, units = camera height)\n"
        f"GT rel. pose  t=({tg.t_ab[0]:+.3f}, {tg.t_ab[1]:+.3f})  "
        f"yaw={math.degrees(tg.yaw_ab):+.1f}deg  k={tg.scale_ab:.4f}  "
        f"|t|={np.linalg.norm(tg.t_ab):.3f}",
        fontsize=9,
    )

    # ---- correspondence panel -------------------------------------------- #
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(np.arange(n), theta_to_col(tg.corr_a2b, n), ".", ms=2.5, color="#bbbbbb",
            label="all columns")
    ax.plot(np.where(tg.covis_a)[0], theta_to_col(tg.corr_a2b, n)[tg.covis_a], ".", ms=3.5,
            color="#d62728", label="co-visible")
    ax.set_xlabel("column in A")
    ax.set_ylabel("corresponding column in B")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ttl = "angular correspondence A -> B"
    if covis_score is not None:
        ttl += f"\nZInD covis_score={covis_score:.3f}   ours(A)={tg.covis_a.mean():.3f}  ours(B)={tg.covis_b.mean():.3f}"
    ax.set_title(ttl, fontsize=9)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
