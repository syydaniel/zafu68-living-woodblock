"""Layered ("multiplane") camera moves for the still prints: near objects are their own layers, so the camera can pass between
them. A per-pixel parallax pushes every pixel by its own depth, a rubber sheet: object edges stretch, and fine woodblock hatching
moves by sub-pixel amounts and blinks between sharp and soft. Here:
  - objects clearly nearer than what surrounds them (the petrified columns, the near trunks, the gate stone; found where the depth
    jumps, not on the smooth ground) are cut out as separate rigid layers, their mattes refined on the picture's own edges;
  - behind them the plate is filled by LaMa, and its depth by the local far depth, so the background layer is smooth and can be
    moved per pixel without tearing (the ground stays one continuous plane);
  - each object moves rigidly by its own depth (no stretching), composited over the background;
  - every frame is rendered at twice the output size and reduced, so the hatching does not blink.
  prep:    .venv_ai/bin/python work/v18/multiplane.py prep NAME   ->  work/v18/mp/NAME_{bg.png, fg.npz, bgd.npy}
  use:     MP = Multiplane(NAME, move=(dx, dy, dz), shift=(sx, sy), dur); MP.frame(t_local) -> float32 RGB 1080p"""
import sys, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V18 = ROOT / 'work/v18'; MPD = V18 / 'mp'; MPD.mkdir(exist_ok=True)
W, H = 1920, 1080
SS = 2                                                                   # render at 2x, reduce


def guided(I, p, r=4, eps=1e-3):
    """He et al. guided filter (grey guide): p's edges snap to I's edges"""
    bf = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = bf(I), bf(p); cIp = bf(I * p) - mI * mp; vI = bf(I * I) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return bf(a) * I + bf(b)


# which parts of each print are cut out as rigid layers: (d_min, d_max, x0, x1, y0, y1) in fractions of the picture, one entry per
# object; chosen by looking at the depth maps (an automatic "nearer than its surroundings" test takes whole canopies in the forest)
SPECS = {
    'K18_petrified': [(0.38, 1.0, 0.14, 0.86, 0.0, 0.795),                         # the columns (behind the fence)
                      (0.45, 1.0, 0.18, 0.79, 0.795, 0.868)],                      # the fence ring in front of them
    'K18_forest': [(0.80, 1.0, 0.325, 0.405, 0.0, 1.0),                            # the near trunk on the left, down to its roots
                   (0.45, 0.75, 0.735, 0.80, 0.0, 0.86),                           # the trunk on the right (above the bushes)
                   (0.68, 1.0, 0.70, 1.00, 0.0, 0.30)],                            # the near leaves top right
    'K18_gate_donghu': [(0.60, 0.80, 0.20, 0.80, 0.70, 0.885)],                     # the 浙江农林大学 stone
}


def prep(name, bg_sigma=0.025):
    img = np.asarray(Image.open(ROOT / f'codex/out/{name}.png').convert('RGB'))
    h, w = img.shape[:2]
    d = np.asarray(Image.open(V18 / f'depth/{name}_d.png')).astype(np.float32)
    d = cv2.resize(d / d.max(), (w, h), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:h, 0:w]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    layers, keep = [], np.zeros((h, w), np.uint8)
    for (d0, d1, x0, x1, y0, y1) in SPECS[name]:
        m = ((d >= d0) & (d <= d1) & (xx >= x0 * w) & (xx < x1 * w) & (yy >= y0 * h) & (yy < y1 * h)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1: continue
        big = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 0.002 * w * h]  # every real object in the box, no specks
        m = np.isin(lab, big).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        a = np.clip(guided(g, m.astype(np.float32), 3, 2e-3), 0, 1)
        a = np.where(cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0, a, 0)
        a[(yy >= y1 * h)] = a[(yy >= y1 * h)] * 0                                   # cut flat at the bottom limit
        dk = float(np.median(d[m > 0]))
        ys, xs = np.where(a > 0.02); bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        layers.append(dict(d=dk, box=(int(bx0), int(by0), int(bx1), int(by1)), a=a[by0:by1, bx0:bx1].astype(np.float16)))
        keep |= (a > 0.05).astype(np.uint8)
    hole = cv2.dilate(keep, np.ones((13, 13), np.uint8))
    from simple_lama_inpainting import SimpleLama
    bg = np.asarray(SimpleLama()(Image.fromarray(img), Image.fromarray(hole * 255)))[:h, :w]
    # the plate's depth: the far depth under the cut-outs, then smoothed hard, so the plate moves as one gently curved surface
    R = max(15, int(0.04 * w)); far = cv2.erode(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1)))
    bgd = np.where(hole > 0, far, d)
    bgd = cv2.GaussianBlur(bgd, (0, 0), bg_sigma * w)
    Image.fromarray(bg).save(MPD / f'{name}_bg.png')
    np.save(MPD / f'{name}_bgd.npy', bgd.astype(np.float16))
    np.savez_compressed(MPD / f'{name}_fg.npz', **{f'a{i}': L['a'] for i, L in enumerate(layers)},
                        meta=json.dumps([dict(d=L['d'], box=L['box']) for L in layers]))
    vis = img.copy(); vis[keep > 0] = (vis[keep > 0] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    Image.fromarray(np.hstack([cv2.resize(vis, (960, 540)), cv2.resize(bg, (960, 540))])).save(MPD / f'{name}_check.jpg', quality=88)
    print(name, f'{len(layers)} objects cut out', [round(L['d'], 2) for L in layers], f'hole {hole.mean() * 100:.1f}%')


class Multiplane:
    """the print as a background plate (moved per pixel by its smooth depth) and rigid object layers (each moved by its own depth),
    composited at 2x and reduced; same move parameters as scenes.Parallax (move=(dx, dy, dz), shift=(sx, sy), dur)"""
    def __init__(self, name, move=(0, 0, 0.05), shift=(18, 6), dur=4.0, fg_gain=1.0, over=1.0, bg_flat=0.0):
        self.move, self.shift, self.dur, self.fg_gain, self.over = move, shift, dur, fg_gain, over       # over: a margin so the edges never run out
        Ws, Hs = W * SS, H * SS
        src = np.asarray(Image.open(ROOT / f'codex/out/{name}.png').convert('RGB')).astype(np.float32) / 255
        h0, w0 = src.shape[:2]
        up = lambda x, interp=cv2.INTER_LANCZOS4: cv2.resize(x, (Ws, Hs), interpolation=interp)
        self.img = np.clip(up(src), 0, 1)
        self.bg = np.clip(up(np.asarray(Image.open(MPD / f'{name}_bg.png').convert('RGB')).astype(np.float32) / 255), 0, 1)
        self.bgd = up(np.load(MPD / f'{name}_bgd.npy').astype(np.float32), cv2.INTER_LINEAR)
        m_ = float(self.bgd.mean()); self.bgd = m_ + (1 - bg_flat) * (self.bgd - m_)      # bg_flat: the plate bends less, the cut-outs carry the depth
        z = np.load(MPD / f'{name}_fg.npz'); meta = json.loads(str(z['meta']))
        sx, sy = Ws / w0, Hs / h0
        self.objs = []
        for i, mt in enumerate(meta):
            x0, y0, x1, y1 = mt['box']
            a = np.zeros((Hs, Ws), np.float32)
            X0, Y0, X1, Y1 = int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))
            a[Y0:Y1, X0:X1] = cv2.resize(z[f'a{i}'].astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
            self.objs.append((mt['d'], a))
        self.objs.sort(key=lambda o: o[0])                              # far first
        yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32); self.xx, self.yy = xx, yy

    def _uv(self, t_local):
        return float(np.clip(t_local / self.dur, -0.4, 1.4)) if self.dur > 0 else 0.0

    def _rigid(self, img, d, u):
        """an object at depth d: the same move as the plate would give that depth, applied to the whole object"""
        dx, dy, dz = self.move; Ws, Hs = W * SS, H * SS
        z = (1 + dz * u * (0.6 + 0.8 * d) * self.fg_gain) * self.over
        tx = (dx + self.shift[0] * (d - 0.5) * self.fg_gain) * u * SS; ty = (dy + self.shift[1] * (d - 0.5) * self.fg_gain) * u * SS
        M_ = np.array([[1 / z, 0, Ws / 2 - Ws / 2 / z - tx], [0, 1 / z, Hs / 2 - Hs / 2 / z - ty]], np.float32)   # dst -> src
        return cv2.warpAffine(img, M_, (Ws, Hs), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def frame(self, t_local, layers_only=False):
        u = self._uv(t_local); dx, dy, dz = self.move; Ws, Hs = W * SS, H * SS
        d = self.bgd
        z = (1 + dz * u * (0.6 + 0.8 * d)) * self.over
        sxm = (self.xx - Ws / 2) / z + Ws / 2 - (dx + self.shift[0] * (d - 0.5)) * u * SS
        sym = (self.yy - Hs / 2) / z + Hs / 2 - (dy + self.shift[1] * (d - 0.5)) * u * SS
        out = cv2.remap(self.bg, sxm, sym, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for dk, a in self.objs:
            prem = self._rigid(np.dstack([self.img * a[..., None], a]), dk, u)
            out = out * (1 - prem[..., 3:4]) + prem[..., :3]
        return cv2.resize(np.clip(out, 0, 1), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)


if __name__ == '__main__':
    if sys.argv[1] == 'prep':
        prep(sys.argv[2])
