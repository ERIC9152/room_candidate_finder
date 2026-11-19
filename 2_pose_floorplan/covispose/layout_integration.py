"""Frozen-pose layout adapter. All local geometry is in camera heights."""
import hashlib
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point
from .corner_fit import refine_corner_polygon, boundary_polygon
from .manhattan_fit import manhattan_polygon, merge_collinear, direction_error
from .geometry import polygon_to_column_distance
from .scene_graph import to_world

FITTING = dict(split_penalty=.1, trigger_deg=2., manhattan_weight=.3,
               merge_turn_deg=20., merge_penalty=.05)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ring_resample(values, width):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or width < 1:
        raise ValueError('Expected a nonempty column vector and positive width')
    columns = (np.arange(width) + .5) * len(values) / width - .5
    return np.interp(columns, np.arange(len(values)), values, period=len(values))


def valid_polygon(poly):
    if poly is None or not np.isfinite(poly).all() or len(poly) < 3:
        return False
    shape = Polygon(poly)
    return shape.is_valid and shape.area > 0 and shape.contains(Point(0, 0))


def fit_layout(bon, columns):
    bon = np.asarray(bon, float)
    report = dict(parameters=FITTING, initial_corner_count=len(columns), fallback_reasons=[])
    if bon.ndim != 1 or not np.isfinite(bon).all() or np.any((bon <= 0) | (bon >= np.pi/2)):
        report.update(status='invalid_boundary', final_corner_count=0,
                      fallback_reasons=['nonfinite_or_out_of_range_boundary'])
        return None, report
    poly, split = refine_corner_polygon(bon, columns, penalty=.1, trigger_deg=2.)
    report['split'] = split
    if poly is None:
        report['fallback_reasons'].append('corner_fit:' + split['status'])
        report['fallback_radius_clipped_columns'] = int(np.sum((1/np.tan(bon) < .1) | (1/np.tan(bon) > 20)))
        poly = boundary_polygon(bon)
    else:
        poly, manhattan = manhattan_polygon(bon, poly, weight=.3)
        report['manhattan'] = manhattan
        if manhattan['status'] != 'ok':
            report['fallback_reasons'].append('manhattan:' + manhattan['status'])
        poly, merge = merge_collinear(bon, poly, weight=.3, turn_deg=20.,
                                     vertex_penalty=.05, refit_initial=False)
        report['merge'] = merge
        if merge['status'] != 'ok':
            report['fallback_reasons'].append('merge:' + merge['status'])
    if not valid_polygon(poly):
        report['fallback_reasons'].append('invalid_final_polygon')
        report.update(status='invalid_polygon', final_corner_count=0)
        return None, report
    report.update(status='fallback' if report['fallback_reasons'] else 'ok',
                  final_corner_count=len(poly), direction_error_deg=direction_error(poly),
                  projection_mae_deg=float(np.degrees(abs(np.arctan2(1,
                      polygon_to_column_distance(poly, len(bon))) - bon)).mean()))
    return poly, report


def local_to_world(poly, theta, center, scale, mirror=False):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('units_per_camera_height must be positive and finite')
    return to_world(np.asarray(poly) * scale * [1, -1 if mirror else 1], theta, center)


def world_to_local(poly, theta, center, scale, mirror=False):
    c, s = np.cos(theta), np.sin(theta)
    return ((np.asarray(poly) - center) @ np.array([[c, -s], [s, c]])) / scale * [1, -1 if mirror else 1]


def require_joint_complete(checkpoint, status):
    if Path(checkpoint).name != 'best.pt':
        raise ValueError('Formal integration requires validation-selected best.pt; smoke is forbidden')
    state = json.loads(Path(status).read_text(encoding='utf-8'))
    if state.get('state') != 'complete':
        raise RuntimeError('Joint experiment is not complete: ' + str(state))


class LayoutAdapter:
    """One prediction per image identity/content and frozen checkpoint hashes."""
    def __init__(self, source, cache, device='cpu',
                 corner='outputs/corners_rtx1070_v1/best.pt',
                 floor='outputs/floor_boundary_rtx1070_v1/best.pt',
                 joint='outputs/joint_layout_transformer_v1/best.pt',
                 status='outputs/joint_layout_transformer_v1/experiment_status.json'):
        import torch
        from .corners import CornerNet
        from .floor_boundary import FloorBoundaryNet
        from .joint_layout import JointLayoutNet
        if source not in ('independent_bilstm', 'joint_transformer'):
            raise ValueError(source)
        self.source, self.device = source, device
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        self.models = []
        paths = [joint] if source == 'joint_transformer' else [corner, floor]
        if source == 'joint_transformer':
            require_joint_complete(joint, status)
        self.weights = {str(Path(p).resolve()): sha256(p) for p in paths}
        self.configs = []
        for i, p in enumerate(paths):
            blob = torch.load(p, map_location='cpu')
            if source == 'joint_transformer':
                model = JointLayoutNet(**blob['config']['architecture'])
                self.threshold = float(blob['metrics']['corner_threshold'])
            else:
                model = CornerNet() if i == 0 else FloorBoundaryNet()
                if i == 0: self.threshold = float(blob['metrics']['threshold'])
            model.load_state_dict(blob['model'])
            self.models.append(model.to(device).eval())
            self.configs.append(blob.get('config', {}))

    def predict(self, path):
        import torch
        from PIL import Image
        from .corners import peak_positions
        identity = dict(image=str(Path(path).resolve()), image_sha256=sha256(path),
                        weights=self.weights, source=self.source, resize=[1024, 512],
                        threshold=self.threshold, nms_deg=3., adapter_version=1)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = self.cache / (key + '.json')
        if cached.exists(): return json.loads(cached.read_text(encoding='utf-8'))
        with Image.open(path) as im:
            original = list(im.size)
            im.draft('RGB', (1024, 512))
            a = np.asarray(im.convert('RGB').resize((1024, 512), Image.BILINEAR))
        x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()[None].div_(255).to(self.device)
        with torch.no_grad():
            if self.source == 'joint_transformer':
                pred = self.models[0](x)
                b, p = pred['bon'], pred['corner_logits'].sigmoid()
            else:
                p, b = self.models[0](x).sigmoid(), self.models[1](x)
        b, p = b[0].float().cpu().numpy(), p[0].float().cpu().numpy()
        finite = np.isfinite(b).all() and np.isfinite(p).all()
        status = ('nonfinite_prediction' if not finite else
                  'invalid_output_shape' if b.shape != (1024,) or p.shape != (1024,) else
                  'invalid_boundary' if np.any((b <= 0) | (b >= np.pi/2)) else
                  'invalid_probability' if np.any((p < 0) | (p > 1)) else 'ok')
        result = dict(identity, original_size=original, config=self.configs,
                      n_cols=len(b), status=status,
                      fallback_reason=None if status == 'ok' else status,
                      bon=b.tolist() if status == 'ok' else None,
                      corner_probability=p.tolist() if status == 'ok' else None,
                      corner_columns=peak_positions(p, self.threshold, 3) if status == 'ok' else [])
        cached.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
        return result
