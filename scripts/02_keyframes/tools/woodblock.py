"""Jiangnan water-print woodblock (江南水印木刻) made FROM a real frame, so every edge sits where the real edge is.
Pipeline: edge-preserving simplify -> colour clusters -> separation into 4 printing plates (墨 石绿 石青 赭石) + paper ->
per-plate water-print gradients, wood grain, knife marks, edge pooling, 1-2 px misregistration -> overprint on 宣纸 ->
key block (carved ink line) traced on the cluster boundaries. An image model may only add surface texture afterwards (gate: edge overlap).
  .venv_ai/bin/python tools/woodblock.py <real.png> <disparity.png> <out_dir>
writes woodblock.png, plates/*.png, pair.jpg (real | woodblock | edge overlay) and prints the edge-overlap scores."""
import sys, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1958)

# pigments as they look printed once (d = 1) on the paper; T = transmittance relative to paper
PAPER = np.array([0.935, 0.905, 0.835], np.float32)
PIG = {'ink':   np.array([0.17, 0.155, 0.14], np.float32),     # 墨
       'green': np.array([0.50, 0.66, 0.52], np.float32),      # 石绿
       'blue':  np.array([0.42, 0.55, 0.68], np.float32),      # 石青
       'ochre': np.array([0.86, 0.60, 0.36], np.float32)}      # 赭石
KEYS = list(PIG)
LOGT = np.stack([np.log(PIG[k] / PAPER) for k in KEYS], 1)     # (3, 4)
SHIFT = {'ink': (0, 0), 'green': (1.5, -1.0), 'blue': (-1.2, 1.0), 'ochre': (1.0, 1.4)}   # misregistration, px at 1080p


def to_lin(x): return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def separate(rgb):
    """densities of the 4 plates that reprint this colour on paper (non-negative, small preferred)"""
    t = np.clip(rgb, 0.02, 1.0)
    y = np.log(t / PAPER)
    A = np.vstack([LOGT, 0.06 * np.eye(4)]); b = np.concatenate([y, np.zeros(4)])
    d, _ = nnls(A.astype(np.float64), b.astype(np.float64))
    return d


def plate_texture(shape, angle, period, seed):
    """wood grain: rings warped by low-frequency noise, along the plank direction"""
    h, w = shape
    r = np.random.default_rng(seed)
    n = cv2.resize(r.random((h // 16 + 2, w // 16 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), 6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ca, sa = np.cos(angle), np.sin(angle)
    v = -sa * xx + ca * yy
    g = np.sin(2 * np.pi * (v + 55 * n) / period)
    g = np.sign(g) * np.abs(g) ** 0.6
    fine = cv2.GaussianBlur(r.random((h, w)).astype(np.float32), (0, 0), 0.8)
    return (0.5 + 0.5 * g) * 0.75 + 0.25 * (fine - fine.min()) / (fine.max() - fine.min() + 1e-6)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edges_struct(rgb8, scale=0.5):
    """structure-level edges (texture suppressed), for the alignment gate"""
    s = cv2.resize(rgb8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    s = cv2.bilateralFilter(s, 9, 40, 7)
    s = cv2.pyrMeanShiftFiltering(s, 8, 18)
    g = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 30, 70)
    n, lab, st, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 25
    return keep[lab]


def overlap(e_ref, e_test, tol=3):
    """share of edges within tol px of the other set: (recall of the real's edges, precision of the print's edges, F)"""
    dr = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 3)
    dt = cv2.distanceTransform((~e_test).astype(np.uint8), cv2.DIST_L2, 3)
    rec = float((dt[e_ref] <= tol).mean()); pre = float((dr[e_test] <= tol).mean())
    return rec, pre, 2 * rec * pre / (rec + pre + 1e-9)


def woodblock(rgb, disp, k=18, cfg=None):
    cfg = cfg or {}
    H, W = rgb.shape[:2]; sc = H / 1080
    rgb8 = (rgb * 255).astype(np.uint8)
    # 1. simplify into big printable shapes, edges kept where they are
    small = cv2.resize(rgb8, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    ms = cv2.pyrMeanShiftFiltering(small, int(10 * sc), 22)
    ms = cv2.pyrMeanShiftFiltering(ms, int(6 * sc), 16)
    simp = cv2.resize(ms, (W, H), interpolation=cv2.INTER_CUBIC)
    simp = cv2.ximgproc.jointBilateralFilter(rgb8, simp, 9, 25, 5) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(simp, 9, 25, 5)
    # 2. colour clusters (Lab + a little depth so far ridges and near trees separate)
    lab = cv2.cvtColor(simp, cv2.COLOR_RGB2LAB).astype(np.float32)
    feat = np.dstack([lab[..., 0] * 1.0, lab[..., 1] * 1.6, lab[..., 2] * 1.6, disp * 60]).reshape(-1, 4)
    sub = feat[rng.choice(len(feat), 120000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, cen = cv2.kmeans(sub, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    dist = ((feat[:, None, :] - cen[None]) ** 2).sum(-1)
    lab_id = dist.argmin(1).reshape(H, W).astype(np.int32)
    lab_id = cv2.medianBlur(lab_id.astype(np.uint8), 5).astype(np.int32)
    # 3. one flat print colour per cluster: mean colour -> plate densities, quantized to carved levels
    lin = to_lin(simp.astype(np.float32) / 255)
    dens = np.zeros((k, 4), np.float32); col = np.zeros((k, 3), np.float32)
    LEVELS = np.array([0, 0.35, 0.7, 1.05, 1.5])
    for c in range(k):
        m = lab_id == c
        if not m.any(): continue
        mean = (simp[m].astype(np.float32) / 255).mean(0); col[c] = mean
        d = separate(mean)
        d = LEVELS[np.abs(d[:, None] - LEVELS[None]).argmin(1)]          # carved: a few tone steps per plate
        dens[c] = d
    # 4. plate density maps; tone inside a region follows the real light, compressed (water-print 擦色)
    lum = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    lum_lo = cv2.GaussianBlur(lum, (0, 0), 18 * sc)
    reg_mean = np.zeros(k, np.float32)
    for c in range(k):
        m = lab_id == c
        if m.any(): reg_mean[c] = lum_lo[m].mean()
    shade = np.clip(1 + 1.4 * (reg_mean[lab_id] - lum_lo), 0.75, 1.35)    # darker than the region mean -> more pigment
    plates = {}
    for i, key in enumerate(KEYS):
        p = dens[lab_id, i] * shade
        p = cv2.GaussianBlur(p, (0, 0), 0.7 * sc)                           # cut edge, slightly soft on wet paper
        pool = np.clip(p - cv2.GaussianBlur(p, (0, 0), 5 * sc), 0, None)    # pigment pools at the plate edge
        p = p + 0.9 * pool
        plates[key] = p
    # 5. art direction: sky top left as paper, mountain foot fades into mist (留白), lake gets horizontal wiped gradient
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    far = disp < cfg.get('far', 0.16)
    if far.any():
        top = np.where(far.any(0), far.argmax(0), H).astype(np.float32)                # first far pixel per column = ridge
        dist = np.clip(yy - top[None], 0, None)
        fade = 1 - 0.55 * np.clip(dist / (110 * sc), 0, 1) ** 1.3
        for key in KEYS: plates[key] = np.where(far, plates[key] * fade, plates[key])
    sky = (disp < 0.035) & (yy < H * 0.25)
    for key in KEYS: plates[key] = np.where(sky, plates[key] * np.clip(yy / (H * 0.12), 0, 1) * 0.6, plates[key])
    # 6. knife marks: short cuts following the local contours in vegetation, leaving paper flecks (刀痕)
    gx = cv2.Sobel(cv2.GaussianBlur(lum, (0, 0), 6 * sc), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(lum, (0, 0), 6 * sc), cv2.CV_32F, 0, 1)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 10 * sc); Jyy = cv2.GaussianBlur(gy * gy, (0, 0), 10 * sc); Jxy = cv2.GaussianBlur(gx * gy, (0, 0), 10 * sc)
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2                      # along the contour
    veg = (plates['green'] > 0.3) & (disp > cfg.get('far', 0.16))
    cuts = np.zeros((H * 2, W * 2), np.float32)
    step = int(9 * sc)
    for y in range(0, H, step):
        for x in range(0, W, step):
            jx, jy = x + rng.integers(0, step), y + rng.integers(0, step)
            if jx >= W or jy >= H or not veg[jy, jx] or rng.random() > 0.55: continue
            a = theta[jy, jx] + rng.normal(0, 0.25); L = rng.uniform(4, 11) * sc
            p0 = np.array([jx - np.cos(a) * L / 2, jy - np.sin(a) * L / 2]); p1 = np.array([jx + np.cos(a) * L / 2, jy + np.sin(a) * L / 2])
            cv2.line(cuts, tuple((p0 * 2 * 16).astype(int)), tuple((p1 * 2 * 16).astype(int)), 1.0, max(1, int(round(1.6 * sc))), cv2.LINE_AA, shift=4)
    cuts = cv2.resize(cuts, (W, H), interpolation=cv2.INTER_AREA)
    for key in ('green', 'ink', 'ochre'): plates[key] = plates[key] * (1 - 0.75 * cuts)
    # 7. wood grain + mottling per plate, misregistration
    for j, key in enumerate(KEYS):
        g = plate_texture((H, W), angle=[0.02, -0.04, 0.05, 0.0][j], period=[38, 26, 44, 31][j] * sc, seed=100 + j)
        plates[key] = plates[key] * (0.80 + 0.32 * g)
        dx, dy = SHIFT[key]; plates[key] = shift(plates[key], dx * sc, dy * sc)
    # 8. overprint on paper
    paper = np.asarray(Image.open(ROOT / 'data/v14/paper.png').convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
    paper = paper / paper.reshape(-1, 3).mean(0) * PAPER
    out = paper.copy()
    for key in KEYS:
        out = out * np.exp(plates[key][..., None] * np.log(PIG[key] / PAPER)[None, None])
    # 9. key block: carved ink line on the region boundaries with real contrast, plus building detail
    bnd = np.zeros((H, W), np.uint8)
    for dy, dx in ((0, 1), (1, 0)):
        a = lab_id[:H - dy, :W - dx]; b = lab_id[dy:, dx:]
        dl = np.abs(cen[a][..., :3] - cen[b][..., :3]).sum(-1)
        bnd[:H - dy, :W - dx] |= ((a != b) & (dl > cfg.get('line_contrast', 26))).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(bnd, 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= int(30 * sc)
    bnd = keep[cc].astype(np.uint8)
    white = (cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)[..., 1] < 50) & (lum > 0.55)
    det = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY), (0, 0), 1.0), 40, 110) > 0
    det &= cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    line = cv2.GaussianBlur(np.maximum(bnd.astype(np.float32), det.astype(np.float32) * 0.8), (0, 0), 0.6 * sc)
    wobble = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3)
    line = np.clip(line * 1.7, 0, 1) * (wobble > 0.47)                               # the knife skips: broken line
    out = out * (1 - 0.82 * line[..., None]) + PIG['ink'] * 0.82 * line[..., None]
    return np.clip(out, 0, 1), plates, lab_id


if __name__ == '__main__':
    src, dsp, od = sys.argv[1], sys.argv[2], Path(sys.argv[3]); od.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255
    H, W = rgb.shape[:2]
    d = np.asarray(Image.open(dsp)).astype(np.float32); d = cv2.resize(d / d.max(), (W, H))
    wb, plates, lab_id = woodblock(rgb, d)
    Image.fromarray((wb * 255).astype(np.uint8)).save(od / 'woodblock.png')
    (od / 'plates').mkdir(exist_ok=True)
    for kk, p in plates.items(): Image.fromarray((np.clip(1 - p / 1.6, 0, 1) * 255).astype(np.uint8)).save(od / 'plates' / f'{kk}.png')
    er = edges_struct((rgb * 255).astype(np.uint8)); ew = edges_struct((wb * 255).astype(np.uint8))
    rec, pre, f = overlap(er, ew)
    ov = np.zeros(er.shape + (3,), np.uint8) + 255; ov[er] = (40, 120, 220); ov[ew] = (220, 60, 40); ov[er & ew] = (30, 30, 30)
    sm = lambda a: cv2.resize(a, (960, 540), interpolation=cv2.INTER_AREA)
    pair = np.hstack([sm((rgb * 255).astype(np.uint8)), sm((wb * 255).astype(np.uint8)), sm(ov)])
    Image.fromarray(pair).save(od / 'pair.jpg', quality=90)
    json.dump({'recall': rec, 'precision': pre, 'F': f}, open(od / 'overlap.json', 'w'))
    print(f'edge overlap  recall {rec:.3f}  precision {pre:.3f}  F {f:.3f}')
