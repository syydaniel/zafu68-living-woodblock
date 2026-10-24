"""Designed water-print woodblock (江南水印木刻) of one real frame, built from semantic layers of THAT frame, so it registers
with the real picture by construction. Each layer is cut the way a printmaker would cut it:
  sky       paper, a faint ochre glow wiped up from the ridge
  mountains 石青 (+石绿 nearer), graded from the ridge down into paper mist (山脚留白), ink ridge line
  buildings paper walls, 赭石 roofs, 石青 shade in two steps, fine ink outline from the real edges
  trees     石绿/赭石 in three tone steps from the real light, ink only in the deepest shade, knife flecks (刀痕)
  lake      石青 wiped darker toward the viewer, tree reflections, horizontal carved ripple cuts
Plates get edge pooling, straight wood grain and 1-2 px misregistration, then overprint on 宣纸.
  .venv_ai/bin/python tools/woodblock2.py <real.png> <disparity.png> <labels.npy> <out_dir>
labels: 1 trees, 2 autumn trees, 3 mountains, 4 buildings, 5 lake, 6 sky (work/v17/masks.py)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from skimage.segmentation import slic, find_boundaries
from skimage import graph

ROOT = Path(__file__).resolve().parents[1]
PAPER = np.array([0.935, 0.905, 0.835], np.float32)
PIG = {'ink':   np.array([0.17, 0.155, 0.14], np.float32),     # 墨
       'blue':  np.array([0.40, 0.52, 0.66], np.float32),      # 石青
       'green': np.array([0.47, 0.63, 0.49], np.float32),      # 石绿
       'ochre': np.array([0.85, 0.58, 0.33], np.float32)}      # 赭石
KEYS = list(PIG)
SHIFT = {'ink': (0, 0), 'green': (1.5, -1.0), 'blue': (-1.2, 1.0), 'ochre': (1.0, 1.4)}


def sstep(a, b, x): x = np.clip((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def grain(shape, seed, period):
    """straight plank grain: long fibres along x, gently waving"""
    h, w = shape; r = np.random.default_rng(seed)
    n = cv2.GaussianBlur(r.random((h, w)).astype(np.float32), (0, 0), sigmaX=60, sigmaY=1.3)
    n = (n - n.mean()) / (n.std() + 1e-6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = cv2.resize(r.random((h // 64 + 2, w // 64 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * 40
    rings = np.sin(2 * np.pi * (yy + warp) / period)
    return 0.6 * np.tanh(n * 0.8) + 0.4 * rings                     # about -1..1


def from_top(mask, sc, above=None):
    """distance (px) from the mask's upper boundary (pixels whose upper neighbour is outside, or inside `above`)"""
    m = mask.astype(bool)
    up = np.zeros_like(m); up[1:] = m[:-1]
    seeds = m & ~up
    if above is not None:
        a = np.zeros_like(m); a[1:] = above[:-1]; seeds |= m & a
    d = cv2.distanceTransform((~seeds).astype(np.uint8), cv2.DIST_L2, 5)
    return cv2.GaussianBlur(d, (0, 0), 2 * sc)


def poly_lines(mask_edges, min_len, sc):
    cs, _ = cv2.findContours(mask_edges.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(np.float32) for c in cs if len(c) >= min_len * sc]


def carve_lines(H, W, polys, width, rng, skip=0.07):
    """key-block line: drawn at 2x, width swells and thins, the knife lifts now and then"""
    lay = np.zeros((H * 2, W * 2), np.float32)
    for P in polys:
        P = cv2.approxPolyDP(P.reshape(-1, 1, 2), 0.6, False)[:, 0, :] if len(P) > 4 else P
        n = len(P)
        if n < 2: continue
        ph = rng.uniform(0, 6.28)
        for i in range(n - 1):
            if rng.random() < skip: continue
            u = i / max(n - 2, 1)
            w = width * (0.75 + 0.35 * np.sin(ph + 9 * u)) * min(1, 0.4 + 5 * min(u, 1 - u))
            a, b = P[i] * 2, P[i + 1] * 2
            cv2.line(lay, tuple((a * 16).astype(int)), tuple((b * 16).astype(int)), 1.0, max(1, int(round(w * 2))), cv2.LINE_AA, shift=4)
    return cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA)


def knife_flecks(H, W, where, theta, rng, sc, density=0.5, length=(3, 8), width=1.3):
    cut = np.zeros((H * 2, W * 2), np.float32); step = max(2, int(8 * sc))
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ys = (ys + rng.integers(0, step, ys.shape)).clip(0, H - 1); xs = (xs + rng.integers(0, step, xs.shape)).clip(0, W - 1)
    for y, x in zip(ys.ravel(), xs.ravel()):
        if where[y, x] < rng.random() / density: continue
        a = theta[y, x] + rng.normal(0, 0.3); L = rng.uniform(*length) * sc
        c, s = np.cos(a) * L / 2, np.sin(a) * L / 2
        cv2.line(cut, (int((x - c) * 32), int((y - s) * 32)), (int((x + c) * 32), int((y + s) * 32)), 1.0, max(1, int(round(width * 2 * sc))), cv2.LINE_AA, shift=4)
    return cv2.resize(cut, (W, H), interpolation=cv2.INTER_AREA)


def woodblock(rgb, disp, lab, seed=1958):
    rng = np.random.default_rng(seed)
    H, W = lab.shape; sc = H / 1080
    rgb8 = (rgb * 255).astype(np.uint8)
    g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    simp = cv2.pyrMeanShiftFiltering(cv2.resize(rgb8, (W // 2, H // 2), interpolation=cv2.INTER_AREA), int(7 * sc), 18)
    gs = cv2.resize(cv2.cvtColor(simp, cv2.COLOR_RGB2GRAY), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat = hsv[..., 0] * 2, hsv[..., 1] / 255
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    P = {k: np.zeros((H, W), np.float32) for k in KEYS}
    L = {i: lab == i for i in range(9)}

    # sky: paper, faint warm glow wiped up from the ridge
    ridge = np.where((~L[6]).any(0), (~L[6]).argmax(0), H).astype(np.float32)
    up = np.clip(ridge[None] - yy, 0, None)
    P['ochre'] += L[6] * 0.10 * (1 - sstep(0, 140 * sc, up))

    # mountains: two distance bands, each graded from its own ridge into paper mist
    far = L[3] & (disp < 0.012); near = L[3] & ~far
    for m, base_b, base_g, ab in ((far, 0.52, 0.0, L[6]), (near, 0.62, 0.55, L[6] | far)):
        if not m.any(): continue
        dist = from_top(m, sc, ab)
        fall = 1 - 0.62 * sstep(0, 190 * sc, dist)                       # bokashi: strong at the ridge, mist below
        vshade = np.clip((0.52 - cv2.GaussianBlur(gs, (0, 0), 4 * sc)) * 1.6, 0, 0.4)                        # the real valley shading, lightly
        P['blue'] += m * (base_b * fall + vshade * fall)
        P['green'] += m * base_g * fall
        P['ink'] += m * 0.10 * sstep(0.30, 0.18, gs) * fall

    # buildings: paper walls, ochre roofs, blue shade in two steps, glass darker
    b = L[4]
    roof = b & (((hue < 18) | (hue > 330)) & (sat > 0.22))
    P['ochre'] += roof * 1.05; P['ink'] += roof * 0.08
    shade = b & ~roof
    step1 = (gs < 0.70).astype(np.float32); step2 = (gs < 0.52).astype(np.float32)
    P['blue'] += shade * (0.22 * step1 + 0.30 * step2)
    P['ink'] += shade * 0.28 * (gs < 0.35)

    # trees: crown by crown (SLIC cells that follow the real crowns), each cell flat in one of three tones,
    # lit side lighter and a darker crescent on the shadow side, as a printmaker cuts a forest
    veg = L[1] | L[2]
    gr = L[1]
    seg = slic(rgb, n_segments=int(3400 * sc * sc), compactness=9, mask=gr, start_label=1, channel_axis=-1)
    # merge neighbouring cells of similar colour: clumps of different sizes, not a honeycomb
    rag = graph.rag_mean_color(rgb, seg, mode='similarity' if False else 'distance')
    seg = graph.cut_threshold(seg, rag, 0.045) + 1
    seg[~gr] = 0
    n = seg.max() + 1
    idx = seg.ravel()
    cnt = np.bincount(idx, minlength=n).astype(np.float32) + 1e-6
    mlum = np.bincount(idx, weights=gs.ravel(), minlength=n) / cnt
    cx = np.bincount(idx, weights=xx.ravel(), minlength=n) / cnt; cy = np.bincount(idx, weights=yy.ravel(), minlength=n) / cnt
    rad = np.sqrt(cnt / np.pi)
    tone = mlum[seg]
    lightv = sstep(0.355, 0.385, tone); darkv = sstep(0.265, 0.235, tone)
    midv = np.clip(1 - lightv - darkv, 0, 1)
    lx, ly = -0.55, -0.83                                                    # sun from the upper left (golden backlight)
    sh = np.clip(((xx - cx[seg]) * -lx + (yy - cy[seg]) * -ly) / (rad[seg] + 1e-6), -1, 1)
    bd = cv2.distanceTransform((~find_boundaries(seg, mode='inner')).astype(np.uint8), cv2.DIST_L2, 3)
    rim = (bd < 2.2 * sc) & (sh > 0.15)
    shade = 0.14 * sh + 0.22 * rim
    P['green'] += gr * (0.55 * lightv + 0.92 * midv + 1.08 * darkv + 0.5 * shade)
    P['ochre'] += gr * (0.24 * lightv + 0.07 * midv)
    P['blue'] += gr * (0.10 * midv + 0.58 * darkv + 0.4 * np.clip(shade, 0, None))
    P['ink'] += gr * (0.12 * darkv + 0.12 * rim)
    darkv = darkv * gr
    au = L[2]
    P['ochre'] += au * 1.12; P['green'] += au * 0.10; P['ink'] += au * 0.10 * (gs < 0.25)
    # lawn: flat light green, sun-warmed, two steps; path: paper
    lw = L[7]
    lsun = sstep(0.30, 0.36, cv2.GaussianBlur(gs, (0, 0), 5 * sc))
    P['green'] += lw * (0.66 - 0.10 * lsun); P['ochre'] += lw * (0.05 + 0.13 * lsun); P['blue'] += lw * 0.12 * (1 - lsun)
    # far trees fade a little toward the mountains (aerial perspective, kept print-flat)
    aerial = sstep(0.08, 0.35, disp)
    vl = veg | L[7]
    for k in ('green', 'ochre', 'ink'): P[k] = np.where(vl, P[k] * (0.72 + 0.28 * aerial), P[k])
    P['blue'] += vl * 0.12 * (1 - aerial)

    # lake: wiped darker toward the viewer, reflections, ripple cuts
    lk = L[5]
    if lk.any():
        dist = from_top(lk, sc)
        depthv = sstep(0, 260 * sc, dist)
        refl = sstep(0.24, 0.12, cv2.GaussianBlur(g, (0, 0), 3 * sc))  # dark reflections of the shore trees
        P['blue'] += lk * (0.42 + 0.55 * depthv + 0.25 * refl)
        P['green'] += lk * 0.35 * refl
        P['ink'] += lk * (0.22 * depthv + 0.18 * refl)
        rip = np.zeros((H * 2, W * 2), np.float32)
        ys, xs = np.nonzero(lk[::int(5 * sc), ::int(23 * sc)])
        for y, x in zip(ys * int(5 * sc), xs * int(23 * sc)):
            if rng.random() > 0.35: continue
            Lr = rng.uniform(10, 46) * sc * (1.2 - 0.6 * depthv[y, x]); x0 = x + rng.uniform(-10, 10) * sc
            cv2.line(rip, (int(x0 * 32), int(y * 32)), (int((x0 + Lr) * 32), int((y + rng.normal(0, 0.4)) * 32)), 1.0, max(1, int(round(2.0 * sc))), cv2.LINE_AA, shift=4)
        rip = cv2.resize(rip, (W, H), interpolation=cv2.INTER_AREA) * lk
        for k in ('blue', 'green', 'ink'): P[k] *= (1 - 0.7 * rip)

    # knife flecks in the deep shade of trees and on the near mountains
    gx = cv2.Sobel(cv2.GaussianBlur(gs, (0, 0), 6 * sc), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(gs, (0, 0), 6 * sc), cv2.CV_32F, 0, 1)
    J = [cv2.GaussianBlur(a, (0, 0), 10 * sc) for a in (gx * gx, gy * gy, gx * gy)]
    theta = 0.5 * np.arctan2(2 * J[2], J[0] - J[1]) + np.pi / 2
    fl = knife_flecks(H, W, (gr * darkv * 0.7).astype(np.float32), theta, rng, sc, density=0.3, length=(4, 10), width=1.1)
    for k in ('green', 'blue', 'ink'): P[k] *= (1 - 0.65 * fl)

    # water-print character: soft cut edge, pigment pooling at the edges, grain, misregistration
    for j, k in enumerate(KEYS):
        p = cv2.GaussianBlur(P[k], (0, 0), 0.8 * sc)
        pool = np.clip(p - cv2.GaussianBlur(p, (0, 0), 6 * sc), 0, None)
        p = p + 0.8 * pool
        p = p * (1 + 0.05 * grain((H, W), 50 + j, [34, 27, 41, 30][j] * sc))
        dx, dy = SHIFT[k]; P[k] = shift(np.clip(p, 0, 2.2), dx * sc, dy * sc)

    paper = np.asarray(Image.open(ROOT / 'data/v14/paper.png').convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
    paper = paper / paper.reshape(-1, 3).mean(0) * PAPER
    out = paper.copy()
    for k in KEYS: out = out * np.exp(P[k][..., None] * np.log(PIG[k] / PAPER)[None, None])

    # key block: layer boundaries (ridges, roofs, shore) + building detail + tree-crown edges in the dark steps
    bnd = np.zeros((H, W), bool)
    for a, bb in ((6, 3), (3, 1), (3, 4), (1, 5), (2, 5), (4, 1), (4, 2), (3, 2), (7, 5), (7, 1), (8, 7), (8, 1)):
        ma, mb = L[a].astype(np.uint8), L[bb].astype(np.uint8)
        bnd |= (cv2.dilate(ma, np.ones((3, 3), np.uint8)) > 0) & (mb > 0)
    bnd &= ~((L[3] & (disp > 0.03)) & False)
    lines = carve_lines(H, W, poly_lines(bnd, 25, sc), 1.7 * sc, rng)
    det = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY), (0, 0), 1.2 * sc), 60, 150) > 0
    det &= cv2.erode(L[4].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lines = np.maximum(lines, carve_lines(H, W, poly_lines(det, 16, sc), 1.0 * sc, rng, skip=0.12) * 0.8)
    folds = cv2.Canny((cv2.GaussianBlur(gs, (0, 0), 3 * sc) * 255).astype(np.uint8), 25, 60) > 0
    folds &= cv2.erode(L[3].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    lines = np.maximum(lines, carve_lines(H, W, poly_lines(folds, 70, sc), 1.2 * sc, rng, skip=0.15) * 0.45)
    crown = find_boundaries(seg, mode='inner') & (sh > 0.0) & gr & (disp > 0.12)
    lines = np.maximum(lines, carve_lines(H, W, poly_lines(crown, 6, sc), 0.9 * sc, rng, skip=0.25) * 0.35)
    lines = np.clip(lines, 0, 1)
    out = out * (1 - 0.85 * lines[..., None]) + PIG['ink'] * 0.85 * lines[..., None]
    return np.clip(out, 0, 1), P


def edges_struct(rgb8, scale=0.5, lo=30, hi=70, min_len=25):
    s = cv2.resize(rgb8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s = cv2.bilateralFilter(s, 9, 40, 7)
    s = cv2.pyrMeanShiftFiltering(s, 8, 18)
    e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (0, 0), 1.5), lo, hi)
    n, labc, st, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_len
    return keep[labc]


def gate(real8, test8, tol=3):
    """alignment gate on MAJOR structure (long edges: ridges, roofs, shore, library):
    recall = share of the real's major edges found in the print; precision = share of the print's major edges found in the real"""
    er_all, et_all = edges_struct(real8), edges_struct(test8)
    er_maj, et_maj = edges_struct(real8, lo=45, hi=110, min_len=60), edges_struct(test8, lo=45, hi=110, min_len=60)
    dt = cv2.distanceTransform((~et_all).astype(np.uint8), cv2.DIST_L2, 3)
    dr = cv2.distanceTransform((~er_all).astype(np.uint8), cv2.DIST_L2, 3)
    rec = float((dt[er_maj] <= tol).mean()); pre = float((dr[et_maj] <= tol).mean())
    return rec, pre, er_maj, et_maj


if __name__ == '__main__':
    src, dsp, labp, od = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]); od.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255
    H, W = rgb.shape[:2]
    d = np.asarray(Image.open(dsp)).astype(np.float32); d = cv2.resize(d / d.max(), (W, H))
    lab = np.load(labp)
    wb, P = woodblock(rgb, d, lab)
    wb8 = (wb * 255).astype(np.uint8); rgb8 = (rgb * 255).astype(np.uint8)
    Image.fromarray(wb8).save(od / 'woodblock.png')
    rec, pre, er, et = gate(rgb8, wb8)
    ov = np.full(er.shape + (3,), 255, np.uint8); ov[er] = (40, 120, 220); ov[et] = (220, 60, 40); ov[er & et] = (20, 20, 20)
    sm = lambda a: cv2.resize(a, (960, 540), interpolation=cv2.INTER_AREA)
    Image.fromarray(np.hstack([sm(rgb8), sm(wb8), sm(ov)])).save(od / 'pair.jpg', quality=90)
    json.dump({'recall_major': rec, 'precision_major': pre}, open(od / 'gate.json', 'w'))
    print(f'gate  recall(major) {rec:.3f}  precision(major) {pre:.3f}  -> {"PASS" if min(rec, pre) >= 0.85 else "FAIL"}')
