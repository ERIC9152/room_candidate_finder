"""Parsing of ZInD ``zind_data.json`` into flat per-pano records.

Only the fields CoVisPose needs are kept: the floor polygon, the floor-plan
transformation, and a couple of quality flags.  The network has no W/D/O head
(neither has the paper's), but doors and openings are kept as geometry: the
pair index uses them to tell which rooms connect (A18).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geometry import PanoPose

LAYOUT_KEYS = ("layout_visible", "layout_raw", "layout_complete")


@dataclass
class PanoRecord:
    house_id: str          # zero-padded, e.g. "0000"
    floor_id: str          # e.g. "floor_01"
    complete_room_id: str  # e.g. "complete_room_03"
    room_id: str           # partial room, e.g. "partial_room_06"
    pano_id: str           # e.g. "pano_11"
    image_path: str        # absolute path to the jpg
    pose: PanoPose
    vertices: np.ndarray   # (M, 2) floor polygon in the pano-local frame
    camera_height: float
    ceiling_height: float
    is_primary: bool
    is_inside: bool
    layout_source: str     # which of LAYOUT_KEYS the vertices came from
    # (K, 2, 2) wall segments in the pano-local frame, from the same layout
    doors: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))
    openings: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))

    @property
    def key(self) -> str:
        return f"{self.floor_id}/{self.room_id}/{self.pano_id}"

    @property
    def uid(self) -> str:
        return f"{self.house_id}/{self.key}"


@dataclass
class HouseRecord:
    house_id: str
    panos: Dict[str, PanoRecord] = field(default_factory=dict)  # keyed by PanoRecord.key

    def by_floor(self) -> Dict[str, List[PanoRecord]]:
        out: Dict[str, List[PanoRecord]] = {}
        for p in self.panos.values():
            out.setdefault(p.floor_id, []).append(p)
        return out


def _wdo_segments(flat) -> np.ndarray:
    """ZInD stores each W/D/O as three entries: its two end points on the wall
    and then (z_bottom, z_top).  Keep the end points."""
    a = np.asarray(flat or [], dtype=np.float64).reshape(-1, 2)
    n = len(a) // 3
    return a[: 3 * n].reshape(n, 3, 2)[:, :2, :] if n else np.zeros((0, 2, 2))


def house_dir(data_root: str, house_id: str) -> str:
    return os.path.join(data_root, house_id)


def normalize_house_id(home_id) -> str:
    """covis_scores.csv stores unpadded ids ("0", "1574"); folders are 4-digit."""
    return "%04d" % int(home_id)


def load_house(
    data_root: str,
    house_id: str,
    layout_key: str = "layout_visible",
    primary_only: bool = True,
    inside_only: bool = True,
) -> HouseRecord:
    """Load one house.  ``layout_key`` falls back through :data:`LAYOUT_KEYS`."""
    house_id = normalize_house_id(house_id)
    hdir = house_dir(data_root, house_id)
    with open(os.path.join(hdir, "zind_data.json"), "r") as f:
        blob = json.load(f)

    house = HouseRecord(house_id=house_id)
    fallbacks = (layout_key,) + tuple(k for k in LAYOUT_KEYS if k != layout_key)

    for floor_id, floor_data in blob["merger"].items():
        for croom_id, croom_data in floor_data.items():
            for room_id, room_data in croom_data.items():
                for pano_id, pano_data in room_data.items():
                    if primary_only and not pano_data.get("is_primary", False):
                        continue
                    if inside_only and not pano_data.get("is_inside", True):
                        continue

                    src = next((k for k in fallbacks if k in pano_data), None)
                    if src is None:
                        continue
                    verts = np.asarray(pano_data[src]["vertices"], dtype=np.float64)
                    if verts.ndim != 2 or verts.shape[0] < 3:
                        continue

                    rec = PanoRecord(
                        house_id=house_id,
                        floor_id=floor_id,
                        complete_room_id=croom_id,
                        room_id=room_id,
                        pano_id=pano_id,
                        image_path=os.path.join(hdir, pano_data["image_path"].replace("/", os.sep)),
                        pose=PanoPose.from_zind(pano_data["floor_plan_transformation"]),
                        vertices=verts,
                        camera_height=float(pano_data.get("camera_height", 1.0)),
                        ceiling_height=float(pano_data.get("ceiling_height", 1.6)),
                        is_primary=bool(pano_data.get("is_primary", False)),
                        is_inside=bool(pano_data.get("is_inside", True)),
                        layout_source=src,
                        doors=_wdo_segments(pano_data[src].get("doors")),
                        openings=_wdo_segments(pano_data[src].get("openings")),
                    )
                    house.panos[rec.key] = rec
    return house


def load_partition(path: str) -> Dict[str, List[str]]:
    """Official ZInD split -> {"train": ["0000", ...], "val": [...], "test": [...]}."""
    with open(path, "r") as f:
        part = json.load(f)
    return {k: [normalize_house_id(h) for h in v] for k, v in part.items()}


def house_scale_meters(data_root: str, house_id: str) -> Optional[float]:
    """``scale_meters_per_coordinate`` (per floor) if present, else None."""
    p = os.path.join(house_dir(data_root, normalize_house_id(house_id)), "zind_data.json")
    with open(p, "r") as f:
        blob = json.load(f)
    return blob.get("scale_meters_per_coordinate")
