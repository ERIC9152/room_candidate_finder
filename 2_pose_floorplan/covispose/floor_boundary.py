"""Single-panorama geometric wall/floor junction prediction.

These targets describe structural wall bases, including furniture-occluded
wall bases. They are not manually annotated visible RGB edge segmentations.
"""
import numpy as np
import torch
from torch.nn import functional as F
from shapely.geometry import Polygon, Point
from .corners import CornerNet
from .geometry import col_centers_theta, dir_of_theta, ray_cast_polygon


def floor_annotations(pano, width=1024, opening_margin_deg=2.):
    raw = np.asarray(pano['layout_raw']['vertices'], float)
    shape = Polygon(raw)
    if not shape.is_valid or not shape.contains(Point(0, 0)):
        raise ValueError('invalid_raw_or_camera_outside')
    dirs = dir_of_theta(col_centers_theta(width))
    distance = ray_cast_polygon(np.zeros(2), dirs, raw)
    target = np.arctan2(1., distance)
    valid = np.isfinite(distance) & (distance > 0) & (target > np.radians(2)) & (target < np.radians(80))
    visible = np.asarray(pano.get('layout_visible', {}).get('vertices', []), float)
    if len(visible) < 3 or not Polygon(visible).is_valid:
        raise ValueError('missing_or_invalid_visible_layout')
    visible_distance = ray_cast_polygon(np.zeros(2), dirs, visible)
    visible_angle = np.arctan2(1., visible_distance)
    disagreement = ~np.isfinite(visible_distance) | (abs(visible_angle-target) > np.radians(2))
    openings = np.asarray(pano['layout_raw'].get('openings', []), float).reshape(-1,2)
    if len(openings) % 3:
        raise ValueError('unexpected_opening_format')
    opening_mask = np.zeros(width, bool)
    for triple in openings.reshape(-1,3,2):
        hit = ray_cast_polygon(np.zeros(2), dirs, triple[:2], closed=False)
        opening_mask |= np.isfinite(hit) & (hit <= distance+.03)
    margin = int(np.ceil(opening_margin_deg*width/360))
    expanded = opening_mask.copy()
    for shift in range(1,margin+1):
        expanded |= np.roll(opening_mask,shift) | np.roll(opening_mask,-shift)
    valid &= ~expanded & ~disagreement
    target = np.nan_to_num(target, nan=0., posinf=0., neginf=0.).astype(np.float32)
    return target, valid.astype(np.float32), dict(opening_fraction=float(expanded.mean()),
        disagreement_fraction=float(disagreement.mean()), valid_fraction=float(valid.mean()))


class FloorBoundaryNet(CornerNet):
    """ResNet34 + circular-context BiLSTM, one junction elevation per column."""
    def forward(self, x):
        return super().forward(x).sigmoid()*(torch.pi/2)


def boundary_loss(pred, target, valid):
    # Smooth L1 in degrees tolerates remaining annotation outliers. No penalty
    # smooths over real concave corners; room regularization remains downstream.
    error = (pred.float()-target.float())*(180/torch.pi)
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none', beta=1.)
    return (loss*valid).sum()/valid.sum().clamp_min(1)
