# -*- coding: utf-8 -*-
"""這一對的幾何結果：產出目錄裡有就讀，沒有就跑 RoMa 與幾何驗證並寫回去。"""
import json
import os

from ..config import regions_root
from ..names import house_of, image_stem


def pair_filename(house, stem_a, stem_b):
    return "%s_%s_%s.json" % (house, stem_a, stem_b)


def named_paths(regions_dir, house, stem_a, stem_b):
    names = [pair_filename(house, stem_a, stem_b), pair_filename(house, stem_b, stem_a)]
    return [os.path.join(regions_dir, name) for name in names]


def existing_path(regions_dir, house, stem_a, stem_b):
    for path in named_paths(regions_dir, house, stem_a, stem_b):
        if os.path.isfile(path):
            return path
    return None


def load_named(regions_dir, house, stem_a, stem_b):
    path = existing_path(regions_dir, house, stem_a, stem_b)
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_saved(regions_dir, img_a, img_b):
    return load_named(regions_dir, house_of(img_a), image_stem(img_a), image_stem(img_b))


def save_result(regions_dir, geometry, img_a, img_b):
    house = geometry.get("house") or house_of(img_a)
    stem_a = geometry.get("img_a") or image_stem(img_a)
    stem_b = geometry.get("img_b") or image_stem(img_b)
    os.makedirs(regions_dir, exist_ok=True)
    path = os.path.join(regions_dir, pair_filename(house, stem_a, stem_b))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(geometry, fh, ensure_ascii=False)
    return path


def geometry_for(img_a, img_b, regions_dir=None, produce=None):
    """先讀產出目錄。沒有這一對時，produce 預設是 verify_pair，結果寫回同一目錄。"""
    regions_dir = regions_dir or regions_root()
    saved = load_saved(regions_dir, img_a, img_b)
    if saved is not None:
        return saved
    if produce is None:
        from .. import verify_pair
        produce = verify_pair
    geometry = produce(img_a, img_b)
    save_result(regions_dir, geometry, img_a, img_b)
    return geometry
