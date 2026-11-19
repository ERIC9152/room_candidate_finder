# -*- coding: utf-8 -*-
"""同環正例：走廊／戶外名稱，以及從資料夾或 JSON 讀組。"""
import json
import os

_OUTDOOR = (
    "yard", "driveway", "street", "deck", "porch", "patio", "road",
    "garden", "pool", "balcony", "outdoor", "exterior", "dt out",
    "陽台", "戶外", "庭院", "院子",
)


def is_hallway(name):
    text = name.lower()
    return "hallway" in text or "走廊" in name


def is_outdoor(name):
    text = name.lower()
    return any(token in text for token in _OUTDOOR)


def dump_groups(groups, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"groups": list(groups)}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_groups(path):
    """JSON 或房子／房間名資料夾。每個房間至少兩張圖才算一組。"""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data.get("groups") or [])
    groups = []
    for house in sorted(os.listdir(path)):
        house_dir = os.path.join(path, house)
        if not os.path.isdir(house_dir):
            continue
        for name in sorted(os.listdir(house_dir)):
            room_dir = os.path.join(house_dir, name)
            if not os.path.isdir(room_dir):
                continue
            ids = []
            for filename in sorted(os.listdir(room_dir)):
                if filename.lower().endswith(".jpg"):
                    ids.append(os.path.splitext(filename)[0])
            if len(ids) >= 2:
                groups.append({"house": house, "name": name, "ids": ids})
    return groups


def pair_keys(groups):
    keys = set()
    for group in groups:
        ids = group["ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                keys.add((group["house"], frozenset((ids[i], ids[j]))))
    return keys


def pair_names(groups):
    """(house, frozenset(id)) → 房間名。"""
    named = {}
    for group in groups:
        ids = group["ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                named[(group["house"], frozenset((ids[i], ids[j])))] = group["name"]
    return named
