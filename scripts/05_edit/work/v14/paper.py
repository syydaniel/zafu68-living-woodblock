"""Paper world: the past printed on 宣纸 in ONE Chinese print tradition: woodblock line (雕版线刻), movable
type (活字) and 朱印. 2D, drawn at 2x and filtered down, so every stroke is clean at 4K too.
Used by work/v14/comp.py (segment B: 萌芽, the three names, the first real picture, the cut into the earth)."""
import math, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1920, 1080
SS = 2                                                     # supersampling for ink
SERIF = str(ROOT / 'data/fonts/NotoSerifSC[wght].ttf')
INK = np.array([0.155, 0.13, 0.105], np.float32)           # 墨 (warm black)
OCHRE = np.array([0.42, 0.26, 0.13], np.float32)           # 赭石
CINNABAR = np.array([0.70, 0.17, 0.12], np.float32)        # 朱砂
GOLD = np.array([0.80, 0.58, 0.20], np.float32)
_paper = np.asarray(Image.open(ROOT / 'data/v14/paper.png').convert('RGB'), np.float32)[:H, :W] / 255
_paper = _paper * np.array([1.035, 1.035, 1.035], np.float32)
_yy, _xx = np.mgrid[0:H, 0:W]
_vig = (1 - 0.07 * (((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]
_rng = np.random.default_rng(11)
_grain = cv2.GaussianBlur(_rng.random((H * SS, W * SS)).astype(np.float32), (0, 0), 1.2)
_grain = 0.8 + 0.2 * (_grain - _grain.min()) / (_grain.max() - _grain.min())


def ease(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def out_cubic(x): x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3
def ramp(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))


def paper(age=0.0):
    """the sheet; age 1 = yellowed (1958), 0 = clean"""
    tint = np.array([1.0, 0.975 - 0.03 * age, 0.94 - 0.09 * age], np.float32)
    return (_paper * tint * _vig).copy()


class Ink:
    """an ink layer drawn at 2x: polylines (with draw-on) and masks; composited onto paper with paper-grain texture"""
    def __init__(self):
        self.m = {}                                            # colour key -> coverage (2x)
    def layer(self, key):
        if key not in self.m: self.m[key] = np.zeros((H * SS, W * SS), np.float32)
        return self.m[key]
    def line(self, pts, width, key='ink', prog=1.0, alpha=1.0, taper=True):
        """pts (n,2) in 1x pixels; draws the first prog of its length; width in 1x px, tapering at both ends"""
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2 or prog <= 0: return
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]); tot = L[-1]
        if tot <= 0: return
        lim = tot * min(prog, 1.0)
        lay = self.layer(key)
        for i in range(len(pts) - 1):
            if L[i] >= lim: break
            a, b = pts[i], pts[i + 1]
            if L[i + 1] > lim: b = a + (b - a) * (lim - L[i]) / (L[i + 1] - L[i])
            u = (L[i] + L[i + 1]) / 2 / tot
            w = width * (min(1.0, 0.35 + 4 * min(u, 1 - u)) if taper else 1.0)
            cv2.line(lay, tuple((a * SS * 16).astype(int)), tuple((b * SS * 16).astype(int)), alpha, max(1, int(round(w * SS))), cv2.LINE_AA, shift=4)
    def blit(self, mask1x, x, y, key='ink', alpha=1.0, scale=1.0):
        """a 1x coverage mask (float) placed with its top-left at (x, y), upsampled to 2x"""
        m = cv2.resize(mask1x, None, fx=SS * scale, fy=SS * scale, interpolation=cv2.INTER_CUBIC) * alpha
        lay = self.layer(key); h, w = m.shape
        X, Y = int(round(x * SS)), int(round(y * SS))
        xa, ya, xb, yb = max(X, 0), max(Y, 0), min(X + w, W * SS), min(Y + h, H * SS)
        if xa >= xb or ya >= yb: return
        lay[ya:yb, xa:xb] = np.maximum(lay[ya:yb, xa:xb], m[ya - Y:yb - Y, xa - X:xb - X])
    def onto(self, img, colors):
        for key, lay in self.m.items():
            cov = cv2.resize(np.clip(lay, 0, 1) * _grain, (W, H), interpolation=cv2.INTER_AREA)[..., None]
            c = colors.get(key, INK)
            img[:] = img * (1 - cov) + c * cov
        return img


# ------------------------------------------------------------------ movable type
_font_cache = {}
def font(px, wght):
    k = (int(px), wght)
    if k not in _font_cache:
        f = ImageFont.truetype(SERIF, int(px))
        try: f.set_variation_by_axes([wght])
        except Exception: pass
        _font_cache[k] = f
    return _font_cache[k]


def type_mask(ch, px, rough, seed=0):
    """a printed character as a 1x coverage mask: rough 1 = worn wood type (uneven ink, ragged edge, slight bleed), 0 = clean"""
    S = 4; f = font(px * S, 700 if rough > 0.5 else 600)
    im = Image.new('L', (int(px * 1.25 * S), int(px * 1.3 * S)), 0)
    ImageDraw.Draw(im).text((px * 0.12 * S, px * 0.02 * S), ch, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    if rough > 0:
        r = np.random.default_rng(seed)
        n1 = cv2.GaussianBlur(r.random(a.shape).astype(np.float32), (0, 0), 2.0 * S)
        n1 = (n1 - n1.mean()) / (n1.std() + 1e-6)
        a = cv2.GaussianBlur(a, (0, 0), 0.9 * S * rough)                      # bleed
        a = np.clip((a - 0.5 + 0.10 * rough * n1) * (6 - 3 * rough) + 0.5, 0, 1)   # ragged edge
        n2 = cv2.GaussianBlur(r.random(a.shape).astype(np.float32), (0, 0), 1.2 * S)
        n2 = (n2 - n2.min()) / (n2.max() - n2.min() + 1e-6)
        a *= 1 - rough * 0.35 * (n2 > 0.62)                                  # dry patches where the block took less ink
    return cv2.resize(a, (a.shape[1] // S, a.shape[0] // S), interpolation=cv2.INTER_AREA)


def seal(text, px, color=CINNABAR):
    """a 朱印 (阴文: the characters left white in red) as an RGBA float image"""
    S = 4; s = int(px * S)
    n = len(text); cols = 1 if n <= 2 else 2; rows = math.ceil(n / cols)
    im = Image.new('L', (s, s), 0); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.05), fill=255)
    f = font(s / rows * 0.78, 700)
    order = [(c, r) for c in range(cols - 1, -1, -1) for r in range(rows)]   # read right column first, top to bottom
    for k, ch in enumerate(text):
        c, r = order[k]
        d.text((s * (c + 0.5) / cols - s / rows * 0.39, s * r / rows + s / rows * 0.04), ch, font=f, fill=0)
    a = np.asarray(im, np.float32) / 255
    rr = np.random.default_rng(len(text) + px)
    a *= (cv2.GaussianBlur(rr.random(a.shape).astype(np.float32), (0, 0), 3) > 0.42).astype(np.float32) * 0.25 + 0.75
    a = cv2.resize(a, (px, px), interpolation=cv2.INTER_AREA)
    rgba = np.zeros((px, px, 4), np.float32); rgba[..., :3] = color; rgba[..., 3] = a * 0.93
    return rgba


def over(img, rgba, x, y, alpha=1.0, scale=1.0):
    if scale != 1.0: rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = rgba.shape[:2]; x, y = int(round(x - w / 2)), int(round(y - h / 2))
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa >= xb or ya >= yb: return
    s = rgba[ya - y:yb - y, xa - x:xb - x]; a = s[..., 3:4] * alpha
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + s[..., :3] * a
