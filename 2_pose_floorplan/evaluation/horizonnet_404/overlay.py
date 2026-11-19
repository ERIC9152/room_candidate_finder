import json, sys, os
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '/home/claude/pkg/2_pose_floorplan/third_party/HorizonNet')
from dataset import cor_2_1d
W, H = 1024, 512
G = '/home/claude/hneval/gtflat'; P = '/home/claude/hneval/runs/all/horizonnet'; I = '/home/claude/hneval/data/img'
def bounds(cor):
    c = cor.copy(); c[0::2, 0] = c[1::2, 0]
    vc, vf = cor_2_1d(c, H, W)
    return (vc / np.pi + .5) * H - .5, (vf / np.pi + .5) * H - .5
def draw(scene, pano, out, title=''):
    im = Image.open(f'{I}/{pano}.png').convert('RGB'); d = ImageDraw.Draw(im)
    gt = np.loadtxt(f'{G}/{scene}/{pano}.txt').reshape(-1, 2)
    dt = np.asarray(json.load(open(f'{P}/{pano}.json'))['uv']) * [W, H]
    for cor, col in ((gt, (0, 255, 0)), (dt, (255, 0, 255))):
        for row in bounds(cor):
            pts = [(x, float(y)) for x, y in enumerate(row)]
            d.line(pts, fill=col, width=3)
    d.rectangle([0, 0, 520, 22], fill=(0, 0, 0)); d.text((4, 4), f'{title}  green=GT  magenta=HorizonNet', fill=(255, 255, 255))
    im.save(out)
if __name__ == '__main__':
    rows = {r['pano']: r for r in json.load(open('out/rows_diag.json', encoding='utf-8'))}
    os.makedirs('fig', exist_ok=True)
    for h in sys.argv[1:]:
        rs = sorted([r for r in rows.values() if r['house'] == h], key=lambda r: r['iou2d'])
        for r in (rs[len(rs)//2], rs[-1]):
            draw(r['scene'], r['pano'], f"fig/{h}_{r['pano'][9:18]}.png", f"{r['scene']} {r['pano'][:8]} 2DIoU {r['iou2d']:.2f} best scale {r['best_scale']:.2f}")
            print(f"fig/{h}_{r['pano'][9:18]}.png")
