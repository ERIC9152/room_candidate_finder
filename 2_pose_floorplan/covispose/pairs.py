"""Building the pano-pair index from ZInD's ``covis_scores.csv``.

``covis_scores.csv`` is the pair source: 458 103 rows, each an ordered pair of
panos on one floor together with Zillow's own co-visibility score.  Empirically
(see docs/PHASE1_NOTES.md) every row is a pair *within one complete room*, and
our per-column co-visibility ground truth reproduces the published score with
r = 0.994 -- so the CSV score and our targets are consistent definitions.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .zind_parse import HouseRecord, normalize_house_id

DEFAULT_COVIS_MIN = 0.30
"""Default lower bound on ZInD's covis_score for a pair to be used.

Chosen (not from the paper -- see README "assumptions") because below ~0.2 the
two panos share so little of the room that the relative pose is frequently
under-constrained, which is label noise rather than a hard example.  0.30 keeps
about 61 % of all rows.  Distribution over the full CSV: p25 0.135, p50 0.497,
p75 0.876.
"""

DEFAULT_MIN_BASELINE = 0.10
"""Minimum camera-to-camera distance, in camera heights (~15 cm), for a pair.

ZInD contains a small number of pano pairs shot from essentially the same spot
(|t| ~ 0.005).  Their translation *direction* is undefined, so they are pure
label noise for a pose regressor.  They are only ~1 % of the rows.  Not from
the paper -- see README "assumptions".
"""

DOOR_MATCH_TOL = 0.5
"""Two complete rooms connect when a door or opening annotated in one lies
within this many camera heights (~0.7 m) of one annotated in the other.

The two annotations of one doorway are its two faces, a wall's thickness
apart; two different doorways on one wall are at least a door's width
(median 0.59 camera heights) apart.  See assumption A18.
"""

PAIR_KINDS = ("pos", "neg", "adj")
"""What a row of a pair index is:

* ``pos`` -- ZInD ``covis_scores.csv`` pair, within one complete room.
* ``neg`` -- two complete rooms that do not connect, and zero geometric
  co-visibility.  Supervises co-visibility and boundary, never pose (A19).
* ``adj`` -- two complete rooms connected by a door or opening.  Exact pose,
  but co-visibility through the doorway is unknown without door states, so
  these are evaluation-only: the ZInD stand-in for the tour data.
"""

PAIR_FIELDS = (
    "house_id", "floor_id", "room_a", "pano_a", "room_b", "pano_b",
    "covis_score", "baseline", "dyaw_deg", "kind",
)


@dataclass(frozen=True)
class PairRef:
    house_id: str
    floor_id: str
    room_a: str
    pano_a: str
    room_b: str
    pano_b: str
    covis_score: float
    baseline: float = float("nan")   # |t_ab|, camera heights
    dyaw_deg: float = float("nan")   # relative yaw, degrees in [-180, 180)
    kind: str = "pos"                # one of PAIR_KINDS

    @property
    def key_a(self) -> str:
        return f"{self.floor_id}/{self.room_a}/{self.pano_a}"

    @property
    def key_b(self) -> str:
        return f"{self.floor_id}/{self.room_b}/{self.pano_b}"

    def as_row(self) -> Dict[str, str]:
        return {
            "house_id": self.house_id,
            "floor_id": self.floor_id,
            "room_a": self.room_a,
            "pano_a": self.pano_a,
            "room_b": self.room_b,
            "pano_b": self.pano_b,
            "covis_score": "%.6f" % self.covis_score,
            "baseline": "%.6f" % self.baseline,
            "dyaw_deg": "%.4f" % self.dyaw_deg,
            "kind": self.kind,
        }


def read_covis_scores(
    csv_path: str,
    houses: Optional[Iterable[str]] = None,
    covis_min: float = DEFAULT_COVIS_MIN,
    covis_max: float = 1.01,
) -> List[PairRef]:
    """Read covis_scores.csv, keeping rows for ``houses`` within the score band."""
    want = None if houses is None else {normalize_house_id(h) for h in houses}
    out: List[PairRef] = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            house = normalize_house_id(row["home_id"])
            if want is not None and house not in want:
                continue
            score = float(row["covis_score"])
            if not (covis_min <= score < covis_max):
                continue
            out.append(
                PairRef(
                    house_id=house,
                    floor_id=row["floor_id"],
                    room_a=row["room_id1"],
                    pano_a=row["pano_id1"],
                    room_b=row["room_id2"],
                    pano_b=row["pano_id2"],
                    covis_score=score,
                )
            )
    return out


def write_pair_index(path: str, pairs: Sequence[PairRef]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PAIR_FIELDS))
        w.writeheader()
        for p in pairs:
            w.writerow(p.as_row())


def read_pair_index(path: str) -> List[PairRef]:
    with open(path, "r", newline="") as f:
        return [
            PairRef(
                house_id=r["house_id"],
                floor_id=r["floor_id"],
                room_a=r["room_a"],
                pano_a=r["pano_a"],
                room_b=r["room_b"],
                pano_b=r["pano_b"],
                covis_score=float(r["covis_score"]),
                baseline=float(r.get("baseline", "nan")),
                dyaw_deg=float(r.get("dyaw_deg", "nan")),
                kind=r.get("kind") or "pos",  # indices written before negatives
            )
            for r in csv.DictReader(f)
        ]


def room_connections(house: HouseRecord, tol: float = DOOR_MATCH_TOL) -> Set[Tuple[str, str, str]]:
    """``(floor_id, complete_room_x, complete_room_y)``, x < y, for every pair
    of complete rooms that share a door or an opening."""
    mids = defaultdict(list)  # floor -> [(complete room, midpoint (global), pose scale)]
    for r in house.panos.values():
        segs = [s for s in (r.doors, r.openings) if len(s)]
        if not segs:
            continue
        for m in r.pose.to_global(np.concatenate(segs).mean(axis=1)):
            mids[r.floor_id].append((r.complete_room_id, m, r.pose.scale))
    out = set()
    for fid, items in mids.items():
        rooms = [i[0] for i in items]
        pts = np.array([i[1] for i in items])
        scale = np.array([i[2] for i in items])
        # global floor-plan units = camera heights x pose scale
        near = (np.linalg.norm(pts[:, None] - pts[None], axis=-1)
                < tol * np.minimum(scale[:, None], scale[None]))
        for i, j in zip(*np.nonzero(near)):
            if rooms[i] != rooms[j]:
                out.add((fid,) + tuple(sorted((rooms[i], rooms[j]))))
    return out
