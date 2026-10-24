"""Full film of work/v14: past printed on paper, today real. Code scenes + real footage (+ LibTV assets when present),
one grade, one paper, one typeface; every join has its own motivated transition.

  .venv/bin/python work/v14/edit14.py [t0 t1]        -> renders/v14/R7_picture.mov (ProRes 422 HQ, 1080p, no audio)
LibTV assets (tools/libtv.py) are picked up automatically from data/libtv/<name>/ when they exist (A1_inkdrop, A2_map, A3_tree,
A4_seedplate, A5_season, A6_ascend); otherwise the code version is used."""
import sys, math, json, glob, os, subprocess
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import map_coordinates
import importlib.util as ilu

ROOT = Path(__file__).resolve().parents[2]
def _load(name, path):
    s = ilu.spec_from_file_location(name, path); m = ilu.module_from_spec(s); s.loader.exec_module(m); return m
C = _load('comp', ROOT / 'work/v14/comp.py')
P = C.P
W, H, FPS = C.W, C.H, C.FPS
ease, ramp, out_cubic = C.ease, C.ramp, C.out_cubic
HOME = os.environ['HOME']
SJ = f'{HOME}/Desktop/Ori_Pre/浙江农林大学校园十景（删油橄榄.mp4'; SJC = (0, 151, 1920, 778)
YX = f'{HOME}/Desktop/Ori_Pre/银杏终稿.mp4'; XY = f'{HOME}/Desktop/Ori_Pre/雪韵·农林.mp4'
SW = f'{HOME}/Desktop/Ori_Pre/DSCF9176.MOV'; D85 = f'{HOME}/Desktop/Ori_Pre/DJI_0285.MP4'
BAMBOO = f'{HOME}/Movies/ZAFU/renders/v3b_bamboo.mov'
INKC = C.INKC


def libtv(name, ext):
    fs = sorted(glob.glob(str(ROOT / f'data/libtv/{name}/*.{ext}')))
    return fs[0] if fs else None


# ------------------------------------------------------------------ the seed (one drawing used everywhere)
def draw_seed(ink, cx, cy, s=1.0, rot=15, open_=0.0, w=3.0, prog=1.0, key='ink'):
    for half in (1, -1):
        th = np.linspace(0, math.pi, 28) if half > 0 else np.linspace(math.pi, 2 * math.pi, 28)
        r = math.radians(rot + half * 9 * open_)
        pts = [(cx + s * (56 * math.cos(q) * math.cos(r) - 38 * math.sin(q) * math.sin(r)),
                cy - half * 6 * open_ * s - s * (56 * math.cos(q) * math.sin(r) + 38 * math.sin(q) * math.cos(r))) for q in th]
        ink.line(pts, w, prog=prog, taper=False, key=key)
    # the hilum and a hint of the embryo inside: a short inner arc
    pts = [(cx + s * (34 * math.cos(q)), cy + s * (14 * math.sin(q) - 4)) for q in np.linspace(math.radians(200), math.radians(340), 14)]
    ink.line(pts, w * 0.6, prog=prog, key=key, alpha=0.7)


# ------------------------------------------------------------------ 0.0 - 5.6  一粒种子会如何成长？
class Open:
    def __init__(self):
        r = np.random.default_rng(4)
        th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
        self.edge = 1 + 0.035 * np.convolve(r.normal(size=720), np.ones(61) / 61, 'same') * 7.8 + 0.012 * np.convolve(r.normal(size=720), np.ones(9) / 9, 'same') * 3
        yy, xx = np.mgrid[0:H // 2, 0:W // 2]
        self.rho = np.hypot(xx * 2 - 960, yy * 2 - 560).astype(np.float32)
        self.ang = ((np.arctan2(yy * 2 - 560, xx * 2 - 960) + 2 * math.pi) % (2 * math.pi) / (2 * math.pi) * 720).astype(np.int32) % 720
        self.video = libtv('A1_inkdrop', 'mp4')
    def frame(self, t):
        img = P.paper(0.5); ink = P.Ink()
        if t >= 0.9:                                     # the drop blooms: darker wet edge, paler middle, fibres at the rim
            R = 150 * (1 - math.exp(-(t - 0.9) * 2.0))
            e = R * self.edge[self.ang]
            cov = np.clip((e - self.rho) / 7.0, 0, 1) * (0.30 + 0.70 * np.clip((self.rho - e * 0.5) / (e * 0.5 + 1e-3), 0, 1) ** 2.5)
            dry = 1 - 0.55 * ease(ramp(t, 2.6, 3.6))
            cov = cv2.resize(cov.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * 0.62 * dry
            img[:] = img * (1 - cov[..., None]) + P.INK * cov[..., None]
        if 0.45 <= t < 0.92:                             # the falling drop
            y = -30 + (560 + 30) * ease(ramp(t, 0.45, 0.9)) ** 1.6
            for k in range(6): ink.line([(960, y - 3 * k), (960, y - 3 * k - 4)], 9 - k, taper=False, alpha=0.9 - 0.12 * k)
        fall = ramp(t, 4.5, 5.6) ** 2.2
        draw_seed(ink, 960, 560 + 1100 * fall, s=1.25, w=3.2, prog=ease(ramp(t, 2.2, 3.3)))
        ink.onto(img, {'ink': P.INK})
        return img


# ------------------------------------------------------------------ 5.6 - 11.4  1958年，它落在了钱王故里、天目山下
class Map:
    X0, X1, Z0, Z1 = -36000.0, 8000.0, -16000.0, 8750.0
    def __init__(self):
        m = json.load(open(ROOT / 'data/v14/terrain_h.json'))
        Hm = np.fromfile(ROOT / 'data/v14/terrain_h.f32', np.float32).reshape(m['nz'], m['nx'])
        acc = np.fromfile(ROOT / 'data/v14/terrain_acc.f32', np.float32).reshape(m['nz'], m['nx'])
        RES = (m['x1'] - m['x0']) / (m['nx'] - 1)
        xs = np.linspace(self.X0, self.X1, W); zs = np.linspace(self.Z0, self.Z1, H)
        X, Z = np.meshgrid(xs, zs)
        I = (X - m['x0']) / RES; J = (Z - m['z0']) / RES
        h = map_coordinates(Hm, [J, I], order=1).astype(np.float32)
        a = map_coordinates(acc, [J, I], order=1).astype(np.float32)
        px = (self.X1 - self.X0) / W
        gy, gx = np.gradient(h, px)
        az, alt = math.radians(315), math.radians(42)
        slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
        shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
        hill = np.clip((h - 60) / 500, 0, 1)
        wash = np.clip((0.75 - shade) * 0.55, 0, 1) * (0.25 + 0.75 * hill) + 0.10 * np.clip(h / 1500, 0, 1)
        u = h / 100.0; fw = np.abs(np.gradient(u)[0]) + np.abs(np.gradient(u)[1]) + 1e-4
        cont = np.clip(1 - np.abs(np.mod(u + 0.5, 1) - 0.5) / (fw * 0.8), 0, 1) * (np.clip((h - 40) / 120, 0, 1)) * 0.22
        riv = np.clip((np.log(a + 1) - math.log(320)) / 1.2, 0, 1)
        riv = cv2.GaussianBlur(riv, (0, 0), 1.0) * 0.55
        base = P.paper(0.8)
        base = base * (1 - wash[..., None]) + P.INK * wash[..., None]
        base = base * (1 - cont[..., None]) + P.INK * cont[..., None]
        stone_blue = np.array([0.22, 0.36, 0.46], np.float32)
        base = base * (1 - riv[..., None]) + stone_blue * riv[..., None]
        ai = libtv('A2_map', 'png') or libtv('A2_map', 'jpg') or libtv('A2_map', 'webp')
        if ai:                                           # the painted map, laid onto the same paper (multiply)
            im = np.asarray(Image.open(ai).convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
            base = P.paper(0.8) * (im / np.maximum(im.reshape(-1, 3).max(0) * 0.98, 1e-3)).clip(0, 1)
        self.base = base.astype(np.float32)
        self.L = ((0 - self.X0) / (self.X1 - self.X0) * W, (0 - self.Z0) / (self.Z1 - self.Z0) * H)     # the campus (landing)
        self.S = ((-31200 - self.X0) / (self.X1 - self.X0) * W, (-13550 - self.Z0) / (self.Z1 - self.Z0) * H)   # West Tianmu
        self.slips = {'临安': C.paper_slip('临安'), '天目山': C.paper_slip('天目山')}
    def frame(self, t):
        z = 1 + 0.32 * ease(ramp(t, 9.6, 11.6)) ** 1.5
        lx, ly = self.L
        M = np.float32([[z, 0, (1 - z) * lx], [0, z, (1 - z) * ly]])
        img = cv2.warpAffine(self.base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ink = P.Ink()
        tp = lambda x, y: (lx + (x - lx) * z, ly + (y - ly) * z)
        land = 7.22
        if t < land:                                     # the seed falls from the sky onto 临安, shrinking
            u = ramp(t, 5.4, land) ** 1.7
            cx, cy = 960 + (lx - 960) * u, -120 + (ly + 120) * u
            draw_seed(ink, cx, cy, s=1.4 - 1.2 * u, w=3.0 - 1.6 * u)
        else:
            cx, cy = tp(lx, ly)
            draw_seed(ink, cx, cy, s=0.2, w=1.6)
            for k in range(3):                           # the ripple (circle 1)
                tr = ramp(t, land + 0.22 * k, land + 0.22 * k + 1.7)
                if 0 < tr < 1:
                    R = (30 + 300 * out_cubic(tr)) * z
                    ink.line([(cx + R * math.cos(q), cy + R * math.sin(q) * 0.92) for q in np.linspace(0, 2 * math.pi, 90)], 2.2 * (1 - tr) + 0.6, taper=False, alpha=1 - tr)
        ink.onto(img, {'ink': P.INK})
        for name, (x, y), ts in (('临安', (lx + 40, ly + 36), 8.06), ('天目山', (self.S[0] + 40, self.S[1] + 46), 8.48)):
            if t >= ts:
                sx, sy = tp(x, y); u = ramp(t, ts, ts + 0.45); spring = 1 - math.exp(-6 * u) * math.cos(9 * u)
                C.over(img, self.slips[name], sx, sy - (1 - spring) * 14, ease(ramp(t, ts, ts + 0.12)), -1.5 + (1 - spring) * -5)
        return img


# ------------------------------------------------------------------ 30.6 - 42.6  roots, the rising gold, the tree, its rings
class RootsTree:
    def __init__(self):
        self.R = C.Roots()
        b = np.fromfile(ROOT / 'data/v14/tree.bin', np.float32).reshape(-1, 12)
        self.crown = b[(b[:, 10] > 1.5) & (b[:, 10] < 2.5)]
        self.leaves = b[b[:, 10] > 2.5]
        r = np.random.default_rng(5)
        self.fruit = self.leaves[r.choice(len(self.leaves), 70, replace=False)]
        self.leafr = r.uniform(5, 10, len(self.leaves)); self.leafa = r.uniform(0.18, 0.4, len(self.leaves))
        rr = np.random.default_rng(8)
        self.rings = [(k, 1 + 0.05 * rr.normal(), rr.uniform(0, 6.28)) for k in range(1, 41)]
        self.tree_ai = libtv('A3_tree', 'png') or libtv('A3_tree', 'jpg') or libtv('A3_tree', 'webp')
    def frame(self, t):
        if t < 37.8:
            return self.R.frame(t)
        dy = 930 * ease(ramp(t, 37.8, 39.9))
        img = P.paper(0.0); ink = P.Ink()
        Pp0 = lambda x, y, z: (960 + x * 72 - z * 18, 150 - y * 72 + dy)
        def Pp(x, y, z):
            if y > 3.2: x = x * 1.28 + 0.5 * math.sin(y * 0.9) * (y - 3.2) / 4
            return Pp0(x, y, z)
        ink.line([(x, 150 + dy + 2 * math.sin(x / 50)) for x in range(-20, 1960, 12)], 2.0)
        for a, b_, w in self.R.hatch:
            ya = a[1] + dy
            if ya < H + 30: ink.line([(a[0], ya), (b_[0], b_[1] + dy)], w, alpha=0.4)
        lay = ink.layer('ochre')
        for s in self.R.segs:                             # the roots, complete
            a = Pp(*s[0:3]); bb = Pp(*s[4:7])
            if min(a[1], bb[1]) > H + 20: continue
            cv2.line(lay, (int(a[0] * P.SS * 16), int(a[1] * P.SS * 16)), (int(bb[0] * P.SS * 16), int(bb[1] * P.SS * 16)), float(min(1.0, 0.35 + 0.65 * s[9])), max(1, int(round(s[8] * 1.7 * P.SS))), cv2.LINE_AA, shift=4)
        lay = ink.layer('ink')
        for s in self.crown:                              # trunk and crown: already there, revealed by the camera
            a = Pp(*s[0:3]); bb = Pp(*s[4:7])
            if max(a[1], bb[1]) < -20: continue
            cv2.line(lay, (int(a[0] * P.SS * 16), int(a[1] * P.SS * 16)), (int(bb[0] * P.SS * 16), int(bb[1] * P.SS * 16)), float(min(1.0, 0.4 + 0.6 * s[9])), max(1, int(round(s[8] * 1.9 * P.SS))), cv2.LINE_AA, shift=4)
        gl = ink.layer('leaf')
        for k, s in enumerate(self.leaves):               # 叶茂: soft 石绿 dabs
            x, y = Pp(*s[4:7])
            if -20 < y < H + 20:
                cv2.ellipse(gl, (int(x * P.SS), int(y * P.SS)), (int(self.leafr[k] * P.SS), int(self.leafr[k] * 0.6 * P.SS)), (k * 37) % 180, 0, 360, float(self.leafa[k]), -1, cv2.LINE_AA)
        for side in (-1, 1):                              # the trunk: tapered, flaring into its roots
            pts = []
            for yv in np.linspace(-0.05, 3.4, 40):
                w = 0.30 + 0.18 * (1 - yv / 3.4) + 0.55 * math.exp(-max(yv, 0) / 0.35)
                pts.append(Pp0(side * w, yv, 0))
            ink.line(pts, 2.6, taper=False)
        for k in range(9):                                # bark: a few vertical engraved strokes
            xo = -0.25 + 0.5 * k / 8
            ink.line([Pp0(xo + 0.02 * math.sin(yv * 3 + k), yv, 0) for yv in np.linspace(0.3, 3.2, 20)], 1.0, alpha=0.45)
        gold = ink.layer('gold')                          # the sap rising along the trunk, then 硕果满枝
        g = ease(ramp(t, 37.6, 39.6))
        ink.line([(960, 150 + dy), (960, 150 + dy - 3.4 * 72 * g)], 3.0, key='gold', alpha=1 - 0.7 * ease(ramp(t, 39.8, 40.6)))
        fr = ease(ramp(t, 40.9, 41.4))
        if fr > 0:
            for s in self.fruit:
                x, y = Pp(*s[0:3]); cv2.circle(gold, (int(x * P.SS), int(y * P.SS)), int(5.5 * P.SS * (0.6 + 0.4 * out_cubic(fr))), float(fr * 0.95), -1, cv2.LINE_AA)
        ink.onto(img, {'ink': P.INK, 'ochre': P.OCHRE, 'gold': P.GOLD, 'leaf': np.array([0.30, 0.50, 0.36], np.float32)})
        # the rings: a detail circle opens from the trunk (a botanical plate's cross-section)
        rr = ease(ramp(t, 41.4, 42.25))
        if rr > 0:
            cx, cy = 960 + 0 * rr, (150 + dy - 1.7 * 72) * (1 - rr) + 540 * rr
            R = 30 + 430 * rr
            yy, xx = np.ogrid[0:H, 0:W]
            inside = (((xx - cx) ** 2 + (yy - cy) ** 2) < R * R).astype(np.float32)[..., None]
            img[:] = img * (1 - 0.08 * rr) * (1 - inside) + P.paper(0.0) * inside
            ink2 = P.Ink()
            for k, e, ph in self.rings:
                rk = R * k / 41.0
                ink2.line([(cx + rk * (1 + 0.02 * e * math.sin(q * 3 + ph)) * math.cos(q), cy + rk * (1 + 0.02 * e * math.cos(q * 2 + ph)) * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 120)],
                          1.3 if k % 5 else 1.9, key='ochre', taper=False, alpha=0.85)
            ink2.line([(cx + R * math.cos(q), cy + R * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 160)], 3.4, taper=False)
            ink2.onto(img, {'ink': P.INK, 'ochre': P.OCHRE})
        return img


# ------------------------------------------------------------------ 66.8 - 69.4  我们研究一粒种子的生长 (a botanical plate)
class Research:
    def __init__(self):
        self.ai = libtv('A4_seedplate', 'png') or libtv('A4_seedplate', 'jpg') or libtv('A4_seedplate', 'webp')
        self.slips = {k: C.paper_slip(k, 24) for k in ('种皮', '子叶', '胚芽', '胚根')}
        self.digits = {str(k): C.glyph_row(str(k), 20, INKC) for k in range(6)}; self.mm = C.glyph_row('毫米', 20, INKC)
    def frame(self, t):
        img = P.paper(0.15); ink = P.Ink()
        cx, cy = 960, 500
        pr = ease(ramp(t, 66.3, 67.3))
        kid = lambda s: [(cx + s * 330 * math.cos(q) - s * 40 * math.exp(-((q - 1.5 * math.pi) / 0.45) ** 2),
                          cy + s * 215 * math.sin(q) - s * 55 * math.exp(-((q - 0.5 * math.pi) / 0.35) ** 2)) for q in np.linspace(0, 2 * math.pi, 220)]
        ink.line(kid(1.0), 3.6, prog=pr, taper=False); ink.line(kid(0.965), 1.6, prog=pr, taper=False)       # 种皮 (double)
        ink.line([(cx + 30, cy - 200), (cx + 18, cy + 150)], 1.8, prog=ease(ramp(t, 66.6, 67.4)))             # the line between the 子叶
        ax = [(cx - 190 + 40 * math.sin(u * 2.6), cy + 120 - 230 * u) for u in np.linspace(0, 1, 30)]        # the embryo axis
        ink.line(ax, 2.6, prog=ease(ramp(t, 66.8, 67.5)))
        for side in (-1, 1):                                                                                   # 胚芽: two tiny leaves
            leaf = [(ax[-1][0] + side * 34 * math.sin(u * math.pi) * (0.3 + u), ax[-1][1] - 50 * u) for u in np.linspace(0, 1, 16)]
            ink.line(leaf, 1.8, prog=ease(ramp(t, 67.1, 67.6)))
        for k in range(5):                                                                                       # hatching in the 子叶
            yk = cy - 140 + 70 * k; ink.line([(cx + 70, yk), (cx + 240 - 30 * abs(k - 2), yk + 18)], 1.0, alpha=0.5, prog=ease(ramp(t, 67.2, 67.7)))
        # a ruler, 5 mm
        x0, y0 = 650, 860
        ink.line([(x0, y0), (x0 + 600, y0)], 1.6, taper=False, prog=ease(ramp(t, 66.8, 67.3)))
        for k in range(51):
            L = 22 if k % 10 == 0 else (14 if k % 5 == 0 else 8)
            ink.line([(x0 + 12 * k, y0), (x0 + 12 * k, y0 - L)], 1.2, taper=False, prog=ease(ramp(t, 66.9 + k * 0.006, 67.0 + k * 0.006)))
        ink.onto(img, {'ink': P.INK})
        if self.ai:
            im = np.asarray(Image.open(self.ai).convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
            img = P.paper(0.15) * (im / np.maximum(im.reshape(-1, 3).max(0) * 0.98, 1e-3)).clip(0, 1)
        for k in range(6):
            if t > 67.3: C.over(img, self.digits[str(k)], x0 + 120 * k - 6, y0 + 10, ease(ramp(t, 67.3, 67.7)))
        if t > 67.3: C.over(img, self.mm, x0 + 640, y0 - 12, ease(ramp(t, 67.3, 67.7)))
        lab = [('种皮', (cx + 250, cy - 210), (cx + 300, cy - 300), 67.25), ('子叶', (cx + 160, cy + 40), (cx + 420, cy + 60), 67.45),
               ('胚芽', (ax[-1][0], ax[-1][1] - 30), (cx - 430, cy - 260), 67.65), ('胚根', (ax[0][0], ax[0][1]), (cx - 470, cy + 210), 67.85)]
        ink2 = P.Ink()
        for name, a, b, ts in lab:
            ink2.line([a, b], 1.1, taper=False, prog=ease(ramp(t, ts, ts + 0.25)))
        ink2.onto(img, {'ink': P.INK})
        for name, a, b, ts in lab:
            if t >= ts + 0.15:
                C.over(img, self.slips[name], b[0] - (90 if b[0] < cx else -10), b[1] - 30, ease(ramp(t, ts + 0.15, ts + 0.3)))
        return img


# ------------------------------------------------------------------ 101.2 - 108  浙江农林大学
class Title:
    def __init__(self):
        lk = Image.open(ROOT / 'data/logo/A04_lockup.png').convert('RGBA')
        w = 1260; h = int(lk.height * w / lk.width)
        self.lock = np.asarray(lk.resize((w, h), Image.LANCZOS), np.float32) / 255
        self.years = C.glyph_row('1958 · 2026', 30, INKC)
    def frame(self, t):
        img = P.paper(0.2); ink = P.Ink()
        cx, cy = 960, 470
        for k in range(1, 15):                             # circle 4: thin rings behind the name
            u = ease(ramp(t, 101.0 + k * 0.05, 101.9 + k * 0.05))
            if u <= 0: continue
            R = 70 + k * 40
            ink.line([(cx + R * math.cos(q), cy + R * math.sin(q) * 0.94) for q in np.linspace(0, 2 * math.pi, 180)], 1.0, key='ochre', taper=False, alpha=0.30 * u, prog=u)
        ink.onto(img, {'ochre': P.OCHRE})
        u = ramp(t, 101.2, 101.45)
        if u > 0:
            sc = 1.03 - 0.03 * out_cubic(u)
            C.over(img, self.lock, cx - self.lock.shape[1] * sc / 2, cy - self.lock.shape[0] * sc / 2, u, 0.0, sc)
        if t > 102.3:
            C.over(img, self.years, cx - self.years.shape[1] / 2, cy + 170, ease(ramp(t, 102.3, 102.9)))
        return img


# ------------------------------------------------------------------ the film
def film():
    beat = lambda k: 48.0 + k * 0.725
    op, mp, pb, rt, rs, ti = Open(), Map(), C.PaperB(), RootsTree(), Research(), Title()
    G = C.Clip
    S = [
        (0.0, 5.6, op.frame, None, None, None),                                                                     # 0
        (5.6, 11.4, mp.frame, None, None, None),                                                                    # 1
        (11.4, 27.4, pb.frame, None, None, None),                                                                   # 2
        (27.4, 30.6, G(ROOT / 'footage/orbit_library', 1 / 30), (960, 540, 1.0), (960, 660, 1.2), dict(exp=0.05, con=1.06, temp=0.2)),   # 3
        (30.6, 42.6, rt.frame, None, None, None),                                                                   # 4
        (42.6, 44.2, G(BAMBOO, 0.3, scale=1920), (960, 540, 1.0), (960, 540, 1.06), dict(exp=0.0, con=1.04, sat=0.95, temp=0.15)),       # 5
        (44.2, 45.8, G(SJ, 77.0, crop=SJC), (1080, 600, 1.35), (1080, 640, 1.55), dict(exp=0.05, temp=0.1)),                            # 6
        (45.8, 47.92, G(BAMBOO, 2.2, scale=1920), (960, 540, 1.04), (960, 500, 1.12), dict(con=1.04, sat=0.95, temp=0.15)),             # 7
        (47.92, beat(6), G(ROOT / 'footage/dive_library', 2100 / 30, speed=2.6), (960, 540, 1.55), (960, 540, 1.0), dict(exp=0.15, con=1.12, sat=1.08, temp=0.4)),   # 8
        (beat(6), beat(7), G(SJ, 147.6, crop=SJC), (960, 389, 1.0), (960, 389, 1.08), dict(sat=0.95, temp=0.1)),                        # 9 荷
        (beat(7), beat(8), G(SJ, 250.7, crop=SJC), (960, 389, 1.02), (960, 389, 1.1), dict(temp=0.05)),                                 # 10 银杏
        (beat(8), beat(9), G(ROOT / 'footage/topdown_lake', 1200 / 30), (960, 540, 1.15), (960, 540, 1.25), dict(con=1.08, sat=1.05, temp=0.15)),   # 11
        (beat(9), beat(10), G(XY, 0.4), (960, 540, 1.0), (980, 540, 1.08), dict(exp=0.05, temp=0.35)),                                  # 12
        (beat(10), beat(11), G(XY, 4.3), (960, 540, 1.0), (960, 540, 1.08), dict(temp=0.35)),                                           # 13
        (beat(11), beat(13), G(SJ, 4.0, crop=SJC), (960, 389, 1.0), (960, 389, 1.08), dict(exp=0.05, con=1.05, temp=0.2)),             # 14 天目
        (beat(13), 59.6, G(ROOT / 'footage/topdown_lake', 14.0), (960, 540, 1.0), (960, 540, 1.1), dict(con=1.05, temp=0.2)),         # 15 鹿鸣岛 (summer, own DJI)
        (59.6, 66.8, G(SJ, 10.5, speed=0.5, crop=SJC), (960, 389, 1.0), (1000, 380, 1.1), dict(exp=0.05, con=1.04, temp=0.05)),        # 16 mist
        (66.8, 69.4, rs.frame, None, None, None),                                                                   # 17
        (69.4, 73.0, G(SJ, 213.6, crop=SJC), (960, 389, 1.0), (960, 389, 1.08), dict(exp=0.05, temp=0.25)),                            # 18 院士林
        (73.0, 80.0, G(SJ, 110.8, speed=0.5, crop=SJC), (960, 420, 1.0), (980, 430, 1.12), dict(exp=0.05, temp=0.2)),                  # 19 校训石
        (80.0, 87.8, G(YX, 53.0), (960, 540, 1.0), (960, 540, 1.08), dict(con=1.05, temp=0.1)),                                         # 20 avenue
        (87.8, 90.8, G(SW, 27.5), (960, 540, 1.0), (960, 540, 1.06), dict(exp=0.05, temp=0.25)),                                         # 21 swans
        (90.8, 92.0, G(XY, 35.6), (960, 540, 1.0), (960, 540, 1.04), dict(temp=0.3)),                                                    # 22 snow island
        (92.0, 93.8, G(YX, 17.5), (960, 540, 1.0), (960, 540, 1.05), dict(con=1.05, temp=0.1)),                                          # 23 autumn island
        (93.8, 101.2, G(D85, 47.6, reverse=True), (960, 540, 1.1), (960, 520, 1.0), dict(exp=0.05, con=1.04, temp=0.25)),              # 24 rising (reversed)
        (101.2, 108.0, ti.frame, None, None, None),                                                                 # 25
    ]
    trans = {0: ('whip', 0.6, dict(dx=0, dy=-1000)), 1: ('zoom', 0.6), 2: ('print', 1.3), 3: ('whip', 0.5, dict(dx=0, dy=-1100)),
             4: ('zoom', 0.6), 5: ('whip', 0.4, dict(dx=-1200, dy=0)), 6: ('whip', 0.4, dict(dx=0, dy=1000)), 7: ('zoom', 0.5),
             8: ('zoom', 0.7), 13: ('luma', 0.8), 14: ('whip', 0.5, dict(dx=0, dy=-1000)), 15: ('ink', 1.1), 16: ('ink', 1.0),
             17: ('ink', 1.0, dict(cx=0.40, cy=0.46)), 20: ('luma', 0.7), 22: ('ink', 0.9), 23: ('whip', 0.5, dict(dx=0, dy=1000)), 24: ('ink', 1.2)}
    L = C.Label
    labels = [L('图书馆', 27.9, 30.45, 120, 880), L('东湖校区', 48.7, beat(6) - 0.05, 120, 880), L('银杏', beat(7) + 0.05, beat(8) - 0.02, 120, 880),
              L('天目山', beat(11) + 0.15, beat(13) - 0.05, 120, 880), L('鹿鸣岛', beat(13) + 0.15, 59.55, 120, 880),
              L('院士林', 69.9, 72.95, 120, 880), L('浙江农林大学', 81.2, 87.6, 120, 880)]
    ct, cap, icap, RT = C.char_times, C.caption, C.ink_caption, C.RiseText
    texts = [RT('一粒种子会如何成长？', ct('u01', '一粒种子会如何成长？'), W / 2, 250, px=62, wght=600, color=INKC, t_out=5.0, shadow=False),
             RT('1958', [5.8, 5.9, 6.04, 6.14], 330, 170, px=120, wght=500, color=INKC, t_out=9.2, shadow=False),
             icap('1958年，它落在了钱王故里、天目山下', [5.8] * 5 + ct('u02', '它落在了乾王故里、天目山下'), 11.2),
             icap('于是，它开始萌芽、生长', ct('u03', '于是，它开始萌芽、生长'), 15.4),
             icap('曾经，它叫', ct('u04', '曾经，它叫'), 20.0), icap('后来，它叫', ct('u05', '后来，它叫'), 23.6), icap('今天，它是', ct('u06', '今天，它是'), 27.2),
             cap('u07', '在这片肥沃的大地上，', 30.45),
             icap('它扎根近70年，', [30.12, 30.6, 30.84, 31.02, 31.2, 31.4, 31.68], 31.95),
             icap('只为积蓄向上生长的力量', ct('u07', '只为继续向上生长的力量'), 34.9),
             icap('像一棵树，根深叶茂，硕果满枝', ct('u08', '像一棵树，根深夜貌，朔果满枝'), 41.3),
             cap('u09', '像一株竹', 44.1),
             cap('u10', '整个校园就是一座植物园', 51.4), cap('u11', '在这里，一草一木，皆有学问', 56.1), cap('u11', '一山一水，皆是课堂', 59.45),
             RT('博览古今', ct('u12', '博览古今'), 1660, 40, px=66, wght=600, color=(40, 34, 28, 235), vertical=True, t_out=66.4, shadow=False, gap=0.12),
             RT('学参天地', ct('u12', '学参天地'), 1560, 40, px=66, wght=600, color=(40, 34, 28, 235), vertical=True, t_out=66.4, shadow=False, gap=0.12),
             icap('在这里，我们研究一粒种子的生长', ct('u13', '在这里，我们研究一粒种子的生长'), 69.1),
             cap('u13', '我们也守护一座森林的未来', 72.8),
             cap('u15', '这是一所大学', 82.05), cap('u15', '坐落在国家森林公园里的大学', 85.6),
             cap('u16', '欢迎你来这里扎根', 90.3), cap('u16', '欢迎你在这里生长', 93.4),
             cap('u17', '他会成全你的梦想', ct('u17', '它会成全你的梦想') and 96.2), cap('u17', '他会托举你的未来', 98.8)]
    seals = [Seal('肯干', 77.3, 1180, 250), Seal('实干', 78.15, 1420, 250), Seal('能干', 79.0, 1660, 250)]
    return 0.0, 108.0, S, labels + seals, texts, trans


class Seal:
    """a big 朱印 stamped onto the picture (the motto's three 干)"""
    def __init__(self, text, ts, x, y, px=170):
        self.img = P.seal(text, px); self.ts, self.x, self.y = ts, x, y
    def draw(self, img, t):
        if t < self.ts or t > 79.95: return
        u = ramp(t, self.ts, self.ts + 0.09)
        P.over(img, self.img, self.x, self.y, alpha=u * (1 - ease(ramp(t, 79.6, 79.95))), scale=1.35 - 0.35 * out_cubic(u))


# ------------------------------------------------------------------ streaming renderer (shots generated with handles, freed when done)
def render(t_a=0.0, t_b=108.0, out=None):
    t0, t1, S, labels, texts, trans = film()
    t_a, t_b = max(t0, t_a), min(t1, t_b)
    out = out or ROOT / 'renders/v14/R7_picture.mov'
    pre = [0.0] * len(S); post = [0.0] * len(S)
    for i, tr in trans.items():
        d = tr[1] / 2 + 0.04; post[i] = max(post[i], d); pre[i + 1] = max(pre[i + 1], d)
    starts = [int(round((s[0] - t0) * FPS)) for s in S]
    cache = {}

    def shot_frames(i):
        if i in cache: return cache[i]
        a, b, clip, k0, k1, g = S[i][:6]
        fa = int(round((a - pre[i] - t0) * FPS)); fb = int(round((b + post[i] - t0) * FPS)); n = fb - fa
        fr = {}
        if callable(clip):
            for k in range(n): fr[fa + k] = clip(t0 + (fa + k) / FPS).astype(np.float16)
        else:
            sh = pre[i] * clip.speed
            c2 = C.Clip(clip.src, clip.t0 - sh if not clip.reverse else clip.t0 + sh, clip.speed, clip.fps_src if clip.files else None, clip.crop, clip.scale, clip.reverse)
            raw = c2.frames(n); na = int(round((b - a) * FPS)); off = int(round(pre[i] * FPS))
            for k, f in enumerate(raw):
                u = ease(min(max((k - off) / max(na - 1, 1), 0), 1))
                cx, cy, z = [k0[j] + (k1[j] - k0[j]) * u for j in range(3)]
                fr[fa + k] = (C.grade(C.reframe(f, cx, cy, z), **g) * C.BAND).astype(np.float16)
        cache[i] = fr; print(f'  shot {i} {a:.2f}-{b:.2f}: {len(fr)} frames', flush=True)
        return fr

    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', str(out)], stdin=subprocess.PIPE)
    fa, fb = int(round((t_a - t0) * FPS)), int(round((t_b - t0) * FPS))
    for f in range(fa, fb):
        t = t0 + f / FPS
        i = max(k for k in range(len(S)) if starts[k] <= f or k == 0)
        for k in list(cache):
            if k < i - 1: del cache[k]
        img = shot_frames(i).get(f)
        for j, tr in trans.items():
            bt = S[j][1]; d = tr[1]
            if bt - d / 2 <= t < bt + d / 2:
                A = shot_frames(j).get(f); B = shot_frames(j + 1).get(f)
                if A is not None and B is not None:
                    img = C.TRANS[tr[0]](A.astype(np.float32), B.astype(np.float32), (t - (bt - d / 2)) / d, **(tr[2] if len(tr) > 2 else {}))
        if img is None:
            fr = shot_frames(i); img = fr[min(fr, key=lambda k: abs(k - f))]
        img = np.array(img, np.float32, copy=True)
        for L_ in labels: L_.draw(img, t)
        for T in texts: T.draw(img, t)
        enc.stdin.write((np.clip(img, 0, 1) * 65535 + 0.5).astype('<u2').tobytes())
        if f % 250 == 0: print(f'frame {f} t={t:.2f}', flush=True)
    enc.stdin.close(); enc.wait(); print('->', out)


if __name__ == '__main__':
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; b = float(sys.argv[2]) if len(sys.argv) > 2 else 108.0
    o = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    render(a, b, o)
