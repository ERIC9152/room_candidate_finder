"""Independent panorama wall-corner experiment; no changes to CoVisPose heads."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .geometry import theta_of, theta_to_col, dir_of_theta, ray_cast_polygon, col_centers_theta
from .model import ColumnEncoder


def corner_annotations(pano, width=1024, sigma_deg=1.0):
    """Visible structural turns of layout_raw, excluding artificial openings.

    Visibility is geometric only: this does not label furniture occlusion.
    Excluded turns and opening rays are ignored, not trained as negatives.
    """
    poly = np.asarray(pano['layout_raw']['vertices'], dtype=float)
    opening = np.asarray(pano['layout_raw'].get('openings', []), dtype=float).reshape(-1, 2)
    endpoints = opening[:len(opening)//3*3].reshape(-1, 3, 2)[:, :2].reshape(-1, 2)
    angles = theta_of(poly)
    positions, ignored = [], []
    for i, v in enumerate(poly):
        a, b = v-poly[i-1], poly[(i+1) % len(poly)]-v
        norms = np.linalg.norm(a)*np.linalg.norm(b)
        turn = math.degrees(math.acos(np.clip(a@b/max(norms, 1e-12), -1, 1)))
        # Cast just either side of a vertex to avoid endpoint floating-point ambiguity.
        rays = dir_of_theta(np.array([angles[i]-1e-6, angles[i]+1e-6]))
        hits = ray_cast_polygon(np.zeros(2), rays, poly)
        visible = np.min(hits) >= np.linalg.norm(v)-0.02
        artificial = len(endpoints) and np.min(np.linalg.norm(endpoints-v, axis=1)) < 0.03
        u = float(theta_to_col(angles[i], width))
        if norms > 1e-8 and turn >= 10 and visible and not artificial:
            positions.append(u)
        else:
            ignored.append(u)
    grid = np.arange(width)
    sigma = sigma_deg*width/360
    target = np.zeros(width, np.float32)
    valid = np.ones(width, np.float32)
    for u in ignored:
        dist = np.abs((grid-u+width/2) % width-width/2)
        valid[dist < 3*sigma] = 0
    if len(endpoints):
        dirs = dir_of_theta(col_centers_theta(width))
        raw_dist = ray_cast_polygon(np.zeros(2), dirs, poly)
        for segment in endpoints.reshape(-1, 2, 2):
            od = ray_cast_polygon(np.zeros(2), dirs, segment, closed=False)
            valid[np.isfinite(od) & (od <= raw_dist+0.03)] = 0
    for u in positions:
        dist = np.abs((grid-u+width/2) % width-width/2)
        target = np.maximum(target, np.exp(-0.5*(dist/sigma)**2).astype(np.float32))
        valid[dist < 3*sigma] = 1
    return target, valid, positions


class CornerDataset(torch.utils.data.Dataset):
    def __init__(self, records, augment=False, width=1024, height=512):
        self.records, self.augment = records, augment
        self.width, self.height = width, height

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with Image.open(rec['image']) as im:
            im.draft('RGB', (self.width, self.height))
            img = np.asarray(im.convert('RGB').resize((self.width, self.height), Image.BILINEAR))
        y = np.asarray(rec['target'], np.float32)
        mask = np.asarray(rec['valid'], np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img, y, mask = img[:, ::-1], y[::-1], mask[::-1]
            roll = np.random.randint(self.width)
            img, y, mask = np.roll(img, roll, 1), np.roll(y, roll), np.roll(mask, roll)
            img = (np.power(img.astype(np.float32)/255, np.random.uniform(.8, 1.2))*255).clip(0, 255)
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()/255,
                torch.from_numpy(y.copy()), torch.from_numpy(mask.copy()), idx)


class CornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.column_encoder = ColumnEncoder('resnet34', False, img_h=512, img_w=1024, n_cols=256)
        self.project = nn.Conv1d(self.column_encoder.out_channels, 128, 1)
        self.context = nn.LSTM(128, 128, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=.1)
        self.head = nn.Conv1d(256, 1, 1)
        nn.init.constant_(self.head.bias, -3)

    def forward(self, x):
        f = F.relu(self.project(self.column_encoder(x, 256)))
        # A duplicated ring provides context across the panorama seam.
        f = torch.cat([f, f, f], -1).transpose(1, 2)
        f, _ = self.context(f)
        logits = self.head(f[:, 256:512].transpose(1, 2))
        logits = F.interpolate(torch.cat([logits[..., -1:], logits, logits[..., :1]], -1),
                               size=1032, mode='linear', align_corners=False)
        return logits[:, 0, 4:-4]


def peak_positions(prob, threshold=.2, separation_deg=3):
    prob = np.asarray(prob)
    width = len(prob)
    candidates = np.flatnonzero((prob >= threshold) & (prob > np.roll(prob, 1)) &
                               (prob >= np.roll(prob, -1)))
    chosen = []
    for u in candidates[np.argsort(-prob[candidates])]:
        if all(abs((u-v+width/2) % width-width/2)*360/width >= separation_deg for v in chosen):
            chosen.append(int(u))
    return chosen


def match_corners(pred, gt, width=1024, tolerance_deg=3):
    """Maximum-cardinality one-to-one circular matching, with localization error."""
    neighbors = []
    for p in pred:
        distances = np.abs((np.asarray(gt)-p+width/2) % width-width/2)*360/width
        neighbors.append([(int(g), float(distances[g])) for g in np.argsort(distances)
                          if distances[g] <= tolerance_deg])
    owners = {}
    def augment(p, seen):
        for g, d in neighbors[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owners or augment(owners[g][0], seen):
                owners[g] = (p, d)
                return True
        return False
    for p in range(len(pred)):
        augment(p, set())
    return len(owners), len(pred)-len(owners), len(gt)-len(owners), [v[1] for v in owners.values()]
