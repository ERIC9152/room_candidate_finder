# -*- coding: utf-8 -*-
"""圓上的方位角重疊。角度可以跨過 ±180 的接縫。"""


def _norm(deg):
    x = deg % 360.0
    if x >= 180.0:
        x -= 360.0
    return x


def _merge(intervals):
    items = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not items:
        return []
    out = [list(items[0])]
    for a, b in items[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered_length(start, width, arcs):
    """區間 [start, start+width) 被 arcs 蓋住的長度。arcs 是 (起點, 寬度)。"""
    if width <= 0.0 or not arcs:
        return 0.0
    origin = _norm(start)

    def pieces(s, w):
        s = _norm(s)
        if s + w <= 180.0:
            return [(s, s + w)]
        return [(s, 180.0), (-180.0, s + w - 360.0)]

    span = pieces(origin, width)
    covered = []
    for origin_b, span_b in arcs:
        covered.extend(pieces(origin_b, span_b))
    total = 0.0
    for a0, a1 in span:
        for c0, c1 in _merge(covered):
            total += max(0.0, min(a1, c1) - max(a0, c0))
    return min(total, width)
