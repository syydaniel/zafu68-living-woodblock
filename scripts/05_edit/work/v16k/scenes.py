"""Code scenes in the film's one woodblock print style. Every scene returns float32 RGB 1080p frames frame(t).
  Names     16.6-27.4  the library print (H3 clean) printed plate by plate: 墨 (天目林学院) -> +赭石 (浙江林学院) -> +石绿, +石青
                       (浙江农林大学); the name is movable type on a vertical 题签 slip; year seals; ends EXACTLY on AI-2's first frame
  Parallax  any        a keyframe print as a 2.5D move (depth layers), for still shots and for AI shots whose clip does not exist"""
import math, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'work/v14'))
import paper as P                                                  # work/v14/paper.py: type_mask, seal, over, Ink
from woodblock2 import PIG, PAPER, KEYS
W, H = 1920, 1080


def ease(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def ramp(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))
def out_cubic(x): x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3


def load(p, size=(W, H)):
    return np.asarray(Image.open(p).convert('RGB').resize(size, Image.LANCZOS), np.float32) / 255


_paper = None
def paper():
    global _paper
    if _paper is None:
        p = load(ROOT / 'data/v14/paper.png'); _paper = (p / p.reshape(-1, 3).mean(0) * PAPER).astype(np.float32)
    return _paper.copy()


def separate(img):
    """per-pixel densities of the four inks (墨 石青 石绿 赭石) that reprint img on paper; via a 33^3 table"""
    LOGT = np.stack([np.log(PIG[k] / PAPER) for k in KEYS], 1); A = np.vstack([LOGT, 0.03 * np.eye(4)])
    n = 33; g = np.linspace(0, 1, n)
    cols = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    dens = np.array([nnls(A, np.concatenate([np.log(np.clip(c, 0.03, 1) / PAPER), np.zeros(4)]))[0] for c in cols]).reshape(n, n, n, 4)
    x = np.clip(img, 0, 1) * (n - 1); i0 = np.floor(x).astype(int).clip(0, n - 2); f = x - i0
    D = 0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = (f[..., 0] if dr else 1 - f[..., 0]) * (f[..., 1] if dg else 1 - f[..., 1]) * (f[..., 2] if db else 1 - f[..., 2])
                D = D + w[..., None] * dens[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
    return D.astype(np.float32), LOGT.astype(np.float32)


class Names:
    T0, T1 = 16.6, 27.4
    # plate k reaches full strength at these times (ink, blue, green, ochre order of KEYS = ink, blue, green, ochre)
    PLATE_T = {'ink': (16.7, 17.6), 'ochre': (20.25, 21.0), 'green': (23.85, 24.6), 'blue': (26.2, 26.9)}

    def __init__(self, final=ROOT / 'libtv_jobs/refs/K_ai2_library_first_clean_1280x720.png'):
        self.final = load(final)
        self.D, self.LOGT = separate(self.final)
        self.paper = paper()
        # 题签: a vertical paper slip at the upper right; names in movable type down it
        self.slip_x, self.slip_y, self.slip_w, self.slip_h = 1640, 60, 150, 820
        self.names = [('天目林学院', 18.44, 21.3), ('浙江林学院', 21.67, 24.9), ('浙江农林大学', 25.6, 99)]
        self.px = 112
        self.masks = {}
        for name, *_ in self.names:
            for ch in name:
                for r in (1.0, 0.55, 0.0):
                    if (ch, r) not in self.masks: self.masks[(ch, r)] = P.type_mask(ch, self.px, r, seed=ord(ch))
        self.seals = [(P.seal('一九五八', 72), 19.75), (P.seal('一九六六', 72), 22.75), (P.seal('二〇一〇', 72), 26.75)]
        # where the mountains end and the (future) campus begins, traced on H3 (x, y) at 1080p
        poly = np.array([(0, 300), (400, 300), (600, 340), (1000, 330), (1400, 370), (1920, 400)], np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); self.xx = xx
        foot = np.interp(np.arange(W), poly[:, 0], poly[:, 1])[None, :]
        self.mtn = np.clip((foot - yy) / 70 + 0.5, 0, 1).astype(np.float32)
        self.drawing = load(ROOT / 'work/v16k/K_build_drawing_1080.png')
        r = np.random.default_rng(19); n = cv2.resize(r.random((H // 24 + 2, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        self.noise = (n - 0.5) * 2
        self.build = None                                                # set by the edit when AI-B exists

    def plates(self, t, D=None):
        D = self.D if D is None else D
        out = np.zeros_like(D)
        for j, k in enumerate(KEYS):
            a, b = self.PLATE_T[k]
            u = ease(ramp(t, a, b))
            if u > 0: out[..., j] = D[..., j] * u
        return self.paper * np.exp(out @ self.LOGT.T)

    def frame(self, t):
        # 1958: only the faint ink Tianmu above the site (the campus did not exist yet)
        mtn = self.mtn
        D0 = self.D.copy(); D0[..., 0] = D0[..., 0] * 1.5 + D0[..., 1] * 0.3; D0[..., 1:] = 0
        base = self.paper * np.exp((D0 * (mtn[..., None] * 0.8 * ease(ramp(t, 16.7, 17.6)))) @ self.LOGT.T)
        # 1966-2001: the construction drawing is drawn on, left to right, with a ragged ink edge
        u = ramp(t, 20.3, 22.9)
        if u > 0:
            front = -200 + 2400 * ease(u) + 90 * self.noise
            m = np.clip((front - self.xx) / 60, 0, 1)[..., None]
            base = base * (1 - m) + self.drawing * m
        img = base
        # today: the campus grows out of the drawing (AI-B), or, until it exists, is printed plate by plate onto it
        if t >= 23.8:
            if self.build is not None:
                img = self.build(t)
            else:
                w = ease(ramp(t, 23.9, 26.9))
                pl = self.paper * np.exp((self.D * w) @ self.LOGT.T)
                img = self.drawing * (1 - w) + pl * w
        fin = ease(ramp(t, self.T1 - 0.35, self.T1))
        if fin > 0: img = img * (1 - fin) + self.final * fin
        # slow push-in common to the whole print (2 %), centred on the library
        z = 1.0 + 0.02 * (1 - ease(ramp(t, self.T0, self.T1)))
        if z > 1.0001:
            M = np.float32([[z, 0, (1 - z) * 960], [0, z, (1 - z) * 560]]); img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        # the slip and its type (fades away as the full-colour print lands)
        sa = ease(ramp(t, 17.4, 18.2)) * (1 - ease(ramp(t, 26.9, 27.3)))
        if sa > 0:
            x0, y0, sw, sh = self.slip_x, self.slip_y, self.slip_w, self.slip_h
            sl = self.paper[y0:y0 + sh, x0:x0 + sw]
            img[y0 + 6:y0 + sh + 6, x0 + 6:x0 + sw + 6] *= 1 - 0.18 * sa                     # its shadow on the print
            img[y0:y0 + sh, x0:x0 + sw] = img[y0:y0 + sh, x0:x0 + sw] * (1 - sa) + sl * sa
            for name, t_in, t_out in self.names:
                rough = {0: 1.0, 1: 0.55, 2: 0.0}[[n[0] for n in self.names].index(name)]
                n = len(name); step = (sh - 60) / 6
                for i, ch in enumerate(name):
                    ti = t_in + 0.16 * i
                    a = ease(ramp(t, ti, ti + 0.12)) * (1 - ease(ramp(t, t_out, t_out + 0.25))) * sa
                    if a <= 0: continue
                    m = self.masks[(ch, rough)]
                    sc = 1.25 - 0.25 * out_cubic(ramp(t, ti, ti + 0.12))                     # pressed in
                    cx = x0 + sw / 2; cy = y0 + 40 + step * (i + 0.5) + (step * (6 - n) / 2)
                    P.over(img, m[..., None] * 0 + 1, cx - m.shape[1] / 2, cy - m.shape[0] / 2, alpha=0) if False else None
                    mm = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                    hh, ww = mm.shape; X = int(cx - ww / 2); Y = int(cy - hh / 2)
                    xa, ya, xb, yb = max(X, 0), max(Y, 0), min(X + ww, W), min(Y + hh, H)
                    cov = mm[ya - Y:yb - Y, xa - X:xb - X][..., None] * a
                    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - cov) + P.INK * cov
            for k, (s, ts) in enumerate(self.seals):
                if t < ts: continue
                u = ramp(t, ts, ts + 0.09)
                P.over(img, s, self.slip_x - 96, self.slip_y + 150 + 110 * k, alpha=u * sa, scale=1.3 - 0.3 * out_cubic(u))
        return np.clip(img, 0, 1).astype(np.float32)


class Parallax:
    """a keyframe print as a 2.5D move: per-pixel shift by disparity (near moves more), with a common push"""
    def __init__(self, png, depth_png=None, move=(0, 0, 0.04), dur=4.0, shift=(18, 6)):
        self.img = load(png)
        if depth_png and Path(depth_png).exists():
            d = np.asarray(Image.open(depth_png)).astype(np.float32); self.d = cv2.resize(d / max(d.max(), 1), (W, H))
        else: self.d = np.full((H, W), 0.5, np.float32)
        self.d = cv2.GaussianBlur(self.d, (0, 0), 3)
        self.move, self.dur, self.shift = move, dur, shift
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); self.xx, self.yy = xx, yy

    def frame(self, t_local):
        u = ease(t_local / self.dur) if self.dur > 0 else 0
        dx, dy, dz = self.move
        z = 1 + dz * u * (0.6 + 0.8 * self.d)                               # near layers grow faster
        sx = (self.xx - W / 2) / z + W / 2 - (dx + self.shift[0] * (self.d - 0.5)) * u
        sy = (self.yy - H / 2) / z + H / 2 - (dy + self.shift[1] * (self.d - 0.5)) * u
        return cv2.remap(self.img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


if __name__ == '__main__':
    import subprocess
    nm = Names(); out = str(ROOT / 'work/v16k/names_test.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in range(int(16.6 * 25), int(27.4 * 25) + 1):
        t = f / 25; im = nm.frame(t); enc.stdin.write((im * 255 + 0.5).astype(np.uint8).tobytes())
        if f in (430, 470, 520, 610, 660, 685): Image.fromarray((im * 255).astype(np.uint8)).save(out.replace('.mp4', f'_{f}.jpg'), quality=88)
    enc.stdin.close(); enc.wait(); print('->', out)
