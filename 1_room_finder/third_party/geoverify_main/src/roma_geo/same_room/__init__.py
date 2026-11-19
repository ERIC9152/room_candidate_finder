# -*- coding: utf-8 -*-
"""同房判別。兩種方法都吃兩張全景，幾何結果共用同一套取得方式。"""
from ..names import image_stem
from .door import decides_same_room as _by_door
from .geometry import geometry_for
from .wedge import decides_same_room as _by_wedge

__all__ = ["by_wedge", "by_door"]


def by_wedge(img_a, img_b, regions_dir=None, produce=None):
    """有位姿、內點至少 30、兩側楔形都至少 60°。"""
    return _by_wedge(geometry_for(img_a, img_b, regions_dir, produce))


def align_doors(geometry, img_a, img_b, doors_a, doors_b):
    """快取若是 B、A 的順序，把門清單對回呼叫端的圖 A、圖 B。"""
    ga = (geometry or {}).get("img_a")
    gb = (geometry or {}).get("img_b")
    if ga == image_stem(img_b) and gb == image_stem(img_a):
        return doors_b, doors_a
    return doors_a, doors_b


def by_door(img_a, img_b, doors_a, doors_b, regions_dir=None, produce=None, contain=0.8):
    """拿掉某一側八成以上落在門裡的共視。還留得下一段才算同房。"""
    geometry = geometry_for(img_a, img_b, regions_dir, produce)
    doors_a, doors_b = align_doors(geometry, img_a, img_b, doors_a, doors_b)
    return _by_door(geometry.get("regions") or [], doors_a, doors_b, contain)
