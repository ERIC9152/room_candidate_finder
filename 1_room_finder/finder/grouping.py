# -*- coding: utf-8 -*-
"""Human decisions -> room groups. Decisions are appended to a JSONL file (latest wins)."""
import json
import os
import time
from collections import defaultdict

RELATIONS = ("same_room", "different_room", "uncertain")


def pair_key(house, a, b):
    a, b = sorted((a, b))
    return (house, a, b)


class Decisions:
    def __init__(self, path, reviewer=""):
        self.path, self.reviewer, self.latest = path, reviewer, {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        e = json.loads(line)
                        k = pair_key(e["house"], e["a"], e["b"])
                        if e.get("relation") is None:
                            self.latest.pop(k, None)          # cleared
                        else:
                            self.latest[k] = e

    def set(self, house, a, b, relation, note="", score=None, thresholds=None):
        if relation is not None and relation not in RELATIONS:
            raise ValueError(relation)
        a, b = sorted((a, b))
        e = dict(house=house, a=a, b=b, relation=relation, note=note, reviewer=self.reviewer,
                 at=time.strftime("%Y-%m-%dT%H:%M:%S"), score=score, thresholds=thresholds)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        k = (house, a, b)
        if relation is None:
            self.latest.pop(k, None)
        else:
            self.latest[k] = e
        return e

    def get(self, house, a, b):
        e = self.latest.get(pair_key(house, a, b))
        return e["relation"] if e else None


def build_groups(house, panos, decisions):
    """Union-find over same_room decisions. Returns (groups, conflicts).
    groups: sorted id lists (size >= 2); conflicts: different_room pairs inside one group."""
    par = {p: p for p in panos}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for (h, a, b), e in decisions.latest.items():
        if h == house and e["relation"] == "same_room" and a in par and b in par:
            par[f(a)] = f(b)
    comp = defaultdict(list)
    for p in panos:
        comp[f(p)].append(p)
    groups = sorted((sorted(v) for v in comp.values() if len(v) >= 2), key=lambda g: g[0])
    conflicts = [(a, b) for (h, a, b), e in decisions.latest.items()
                 if h == house and e["relation"] == "different_room" and a in par and b in par and f(a) == f(b)]
    return groups, conflicts


def export_groups(path, houses, decisions, thresholds, source_commit=""):
    """Write room groups in the room_classification.json group format."""
    groups_out, conflicts_out = [], []
    for house, panos in houses.items():
        groups, conflicts = build_groups(house, panos, decisions)
        for i, g in enumerate(groups, 1):
            src = [f"{a}__{b}" for (h, a, b), e in decisions.latest.items()
                   if h == house and e["relation"] == "same_room" and a in g]
            groups_out.append(dict(room_id=f"{house}_room_{i:03d}", task=house, floor="", room_label="",
                                   space_kind="room", panorama_uuids=g, status="human_verified", notes="",
                                   source_pairs=sorted(src), created_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
        conflicts_out += [dict(house=house, a=a, b=b) for a, b in conflicts]
    from .scores import RULE_TEXT
    data = dict(schema_version=1, generator="room_candidate_finder", classifier=dict(
        rule=RULE_TEXT.get(thresholds.get("mode", "plain"), RULE_TEXT["plain"]),
        geoverify_commit=source_commit, **thresholds),
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%S"), groups=groups_out, conflicts=conflicts_out,
        pair_reviews=[dict(pair_id=f"{a}__{b}", uuid_a=a, uuid_b=b, relation=e["relation"], human_verified=True,
                           notes=e.get("note", ""), reviewer=e.get("reviewer", ""), reviewed_at=e["at"],
                           meta=dict(task=h)) for (h, a, b), e in sorted(decisions.latest.items())])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return data
