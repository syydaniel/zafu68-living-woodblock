"""3D pigment-particle transition: picture A becomes a cloud of particles placed by its depth (Depth Anything V2), the camera
keeps moving forward through it, the particles gather at a few structure points of picture B and B bleeds outward from there.
  render     fine, crisp grains (one per 2x2 px); the sheet under them turns to bare paper
  render_v1  particles paired with B's inside 32 px tiles by brightness, each taking on its colour in B on the way
  frames = render(A, B, n, move=(dz, dy))   A, B: float32 RGB 1080p; returns a list of n frames"""
import numpy as np, cv2, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

W, H = 1920, 1080
_M = {}


def depth(img):
    if 'm' not in _M:
        dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        _M['p'] = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf')
        _M['m'] = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf').to(dev).eval(); _M['dev'] = dev
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((1036, 588))
    x = _M['p'](images=im, return_tensors='pt').to(_M['dev'])
    with torch.no_grad(): d = _M['m'](**x).predicted_depth[0].float().cpu().numpy()
    d = cv2.resize(d, (W, H)); d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.GaussianBlur(d, (0, 0), 2)


def centres(B, k=4, min_d=260):
    """where the picture should first appear: the k strongest structure points of B (blurred gradient peaks), well apart"""
    g = cv2.cvtColor((np.clip(B, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 40)
    yy, xx = np.mgrid[0:H, 0:W]; mag *= np.exp(-(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H / 2) / (H * 0.55)) ** 2))   # prefer the middle
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(mag), mag.shape); out.append((float(x), float(y)))
        mag[np.hypot(xx - x, yy - y) < min_d] = 0
    return np.array(out, np.float32)


class Bleed:
    """ink bleed: B appears at the gathering points first, then spreads outward like ink soaking into xuan paper:
    a ragged fibrous front, a slightly darker wet rim, a faint tide line left behind"""
    def __init__(self, pts, seed=3):
        r = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.d = np.min(np.stack([np.hypot(xx - x, yy - y) for x, y in pts]), 0)        # distance to the nearest gathering point
        n1 = cv2.GaussianBlur(r.random((H // 12 + 2, W // 12 + 2)).astype(np.float32), (0, 0), 3)
        n1 = cv2.resize((n1 - n1.mean()) / (n1.std() + 1e-6), (W, H), interpolation=cv2.INTER_CUBIC)
        fib = cv2.GaussianBlur(r.random((H, W)).astype(np.float32), (0, 0), sigmaX=4, sigmaY=0.9)
        fib = (fib - fib.mean()) / (fib.std() + 1e-6)
        self.noise = n1 * 46 + fib * 7                                                    # soft lobes + fine paper fibres at the front
        self.rmax = float(self.d.max()) + 200
    def mask(self, u):
        """u 0..1: 0 nothing, 1 the whole frame; returns (alpha, wet rim)"""
        R = 40 + (self.rmax) * (u ** 1.4)
        x = (R - (self.d + self.noise)) / 34
        a = cv2.GaussianBlur(np.clip(0.5 + x, 0, 1).astype(np.float32), (0, 0), 2.0)
        rim = np.exp(-((x - 0.4) / 1.1) ** 2) * (1 - u ** 3)
        return a.astype(np.float32), rim.astype(np.float32)


def render_v1(A, B, n, move=(7.5, 1.6), seed=58, K=2, gather_k=4):
    rng = np.random.default_rng(seed)
    dA = depth(A); dB = depth(B)
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    ys = (ys + rng.integers(0, 2, ys.shape)).ravel(); xs = (xs + rng.integers(0, 2, xs.shape)).ravel()
    N = len(xs); cA = A[ys, xs]; cB = B[ys, xs]
    f = 0.5 * H / np.tan(np.radians(40) / 2)
    ZA = 1 / (1 / 80 + (1 / 6 - 1 / 80) * dA[ys, xs]); XA = (xs - W / 2) / f * ZA; YA = (ys - H / 2) / f * ZA
    lumA = cA @ np.array([0.2126, 0.7152, 0.0722]); lumB = cB @ np.array([0.2126, 0.7152, 0.0722])
    tile = (ys // 32) * (W // 32 + 1) + xs // 32
    kA = np.lexsort((lumA + rng.random(N) * 1e-3, tile)); kB = np.lexsort((lumB + rng.random(N) * 1e-3, tile))
    pair = np.empty(N, int); pair[kA] = kB
    bx, by, cBp = xs[pair].astype(np.float32), ys[pair].astype(np.float32), cB[pair]
    near = dA[ys, xs]
    pts = centres(B, gather_k)
    # each particle gathers at the gathering point nearest to its place in B, then is released into the bleed
    cid = np.argmin(np.stack([np.hypot(bx - x, by - y) for x, y in pts]), 0)
    gx, gy = pts[cid, 0], pts[cid, 1]
    rr = np.sqrt(rng.random(N)) * 38; th = rng.random(N) * 6.283
    gx = gx + rr * np.cos(th); gy = gy + rr * np.sin(th) * 0.8                  # a dense knot, not a single point
    rel = 0.02 + 0.22 * (1 - near) + 0.06 * rng.random(N)                      # near particles leave first
    arr = 0.42 + 0.10 * rng.random(N)                                          # all gathered by about 0.5
    sw = rng.normal(0, 1, (N, 2)).astype(np.float32); ph = rng.random(N) * 6.28
    bleed = Bleed(pts)

    def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

    def pos(M):
        cam = ease(np.clip(M / 0.6, 0, 1)); Zc = ZA - move[0] * cam; Zs = np.maximum(Zc, 1.2)
        ax = W / 2 + f * XA / Zs; ay = H / 2 + f * (YA - move[1] * cam) / Zs
        tau = np.clip((M - rel) / (arr - rel), 0, 1); e = ease(tau) ** 1.3; bell = np.sin(np.pi * tau)
        x = ax + (gx - ax) * e + 30 * bell * sw[:, 0] * np.cos(ph + 6 * M)
        y = ay + (gy - ay) * e + 30 * bell * sw[:, 1] * np.sin(ph + 6 * M)
        # gathered particles swirl slowly in their knot
        sp = np.clip((M - arr) / 0.1, 0, 1)
        x = x + sp * 10 * np.cos(ph + 9 * M); y = y + sp * 8 * np.sin(ph + 9 * M)
        col = cA + (cBp - cA) * ease((M - rel) / 0.5)[:, None]
        a = np.where((Zc > 1.2) | (e > 0.5), 1.0, np.clip((Zc - 0.4) / 0.8, 0, 1))
        return x, y, col, a, tau

    def splat(x, y, col, a, size):
        acc = np.zeros((4, H * W), np.float32)
        ok = (x > -2) & (x < W + 1) & (y > -2) & (y < H + 1) & (a > 0.01)
        x, y, col, a, size = x[ok], y[ok], col[ok], a[ok], size[ok]
        for ox in (0.0, 1.0):
            for oy in (0.0, 1.0):
                px, py = x + ox * size, y + oy * size
                x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int); fx = px - x0; fy = py - y0
                for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                    xi, yi = x0 + dx, y0 + dy; m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                    idx = yi[m] * W + xi[m]; ww = (w * a)[m]
                    acc[3] += np.bincount(idx, ww, H * W)
                    for c in range(3): acc[c] += np.bincount(idx, ww * col[m, c], H * W)
        return acc[:3].T.reshape(H, W, 3), acc[3].reshape(H, W)

    paper = None
    out = []
    for fi in range(n):
        acc = 0; cov = 0
        for k in range(K):
            M = np.clip((fi + (k + 0.5) / K - 0.5) / max(n - 1, 1), 0, 1)
            x, y, col, a, tau = pos(M)
            ac, cv = splat(x, y, col, a, 1.0 + 0.6 * np.sin(np.pi * tau)); acc = acc + ac; cov = cov + cv
        acc /= K; cov /= K
        colr = acc / np.maximum(cov, 1e-6)[..., None]; o = np.clip(cov * 1.15, 0, 1)[..., None]
        M = fi / max(n - 1, 1)
        if paper is None: paper = np.full((H, W, 3), [0.935, 0.905, 0.835], np.float32)
        bg = A * (1 - ease(M / 0.85)) + paper * ease(M / 0.85)               # A fades to paper slowly under the particles
        frame = bg * (1 - o) + colr * o
        wa = ease(M / 0.3); frame = A * (1 - wa) + frame * wa
        # B appears inside each knot, then bleeds outward over the paper; the knots dissolve into the ink
        ub = np.clip((M - 0.5) / 0.5, 0, 1)
        if ub > 0:
            al, rim = bleed.mask(ub)
            wet = B * (1 - 0.18 * rim[..., None])
            frame = frame * (1 - al[..., None]) + wet * al[..., None]
        out.append(np.clip(frame, 0, 1).astype(np.float32))
    return out


def render(A, B, n, move=(7.5, 1.6), seed=58, K=3, gather_k=4, A_seq=None, B_seq=None):
    """picture A lifts off as fine, crisp pigment grains (one per 2x2 px, its own colour, placed in depth) while
    the camera keeps moving; under them the sheet turns to bare paper; the grains stream to a few gathering points and vanish as they
    arrive (a small warm glow there); picture B appears exactly at those points and bleeds outward like ink on xuan paper.
    No averaged colour blobs: grains are drawn as separate dots, B is revealed sharp.
    A_seq / B_seq (optional, n frames each): the two shots keep moving under the effect (A until it has lifted, B as it bleeds in)."""
    rng = np.random.default_rng(seed)
    dA = depth(A)
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    ys = (ys + rng.integers(0, 2, ys.shape)).ravel(); xs = (xs + rng.integers(0, 2, xs.shape)).ravel()
    N = len(xs); cA = A[ys, xs]
    f = 0.5 * H / np.tan(np.radians(40) / 2)
    ZA = 1 / (1 / 80 + (1 / 6 - 1 / 80) * dA[ys, xs]); XA = (xs - W / 2) / f * ZA; YA = (ys - H / 2) / f * ZA
    pts = centres(B, gather_k)
    cid = np.argmin(np.stack([np.hypot(xs - x, ys - y) for x, y in pts]), 0)
    gx, gy = pts[cid, 0].astype(np.float32), pts[cid, 1].astype(np.float32)
    near = dA[ys, xs]
    rel = 0.03 + 0.20 * (1 - near) + 0.08 * rng.random(N)          # near grains lift first
    arr = 0.34 + 0.14 * rng.random(N)                              # all arrive between 0.34 and 0.48
    sw = rng.normal(0, 1, (N, 2)).astype(np.float32); ph = rng.random(N) * 6.28
    keep = rng.random(N) < 0.55                                    # only half the grains travel (a lighter cloud)
    bleed = Bleed(pts)
    paper = np.full((H, W, 3), [0.935, 0.905, 0.835], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.zeros((H, W), np.float32)
    for x, y in pts: glow = np.maximum(glow, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 26.0 ** 2)))

    def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

    def grains(M):
        cam = ease(np.clip(M / 0.6, 0, 1)); Zc = ZA - move[0] * cam; Zs = np.maximum(Zc, 1.2)
        ax = W / 2 + f * XA / Zs; ay = H / 2 + f * (YA - move[1] * cam) / Zs
        tau = np.clip((M - rel) / (arr - rel), 0, 1); e = ease(tau) ** 1.25; bell = np.sin(np.pi * tau)
        x = ax + (gx - ax) * e + 22 * bell * sw[:, 0] * np.cos(ph + 6 * M)
        y = ay + (gy - ay) * e + 22 * bell * sw[:, 1] * np.sin(ph + 6 * M)
        lifted = np.clip((M - rel) / 0.06, 0, 1)                      # a grain shows only once it has lifted off the sheet
        a = lifted * (1 - np.clip((tau - 0.82) / 0.18, 0, 1)) * keep * np.where(Zc > 1.2, 1.0, np.clip((Zc - 0.4) / 0.8, 0, 1))
        return x, y, a

    def splat(x, y, a):
        acc = np.zeros((4, H * W), np.float32)
        ok = (x > -1) & (x < W - 1) & (y > -1) & (y < H - 1) & (a > 0.02)
        x, y, a, col = x[ok], y[ok], a[ok], cA[ok]
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy; m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            idx = yi[m] * W + xi[m]; ww = (w * a)[m]
            acc[3] += np.bincount(idx, ww, H * W)
            for c in range(3): acc[c] += np.bincount(idx, ww * col[m, c], H * W)
        cov = acc[3].reshape(H, W); col = (acc[:3] / np.maximum(acc[3], 1e-6)).T.reshape(H, W, 3)
        return col, np.clip(cov, 0, 1)

    out = []
    for fi in range(n):
        M = fi / max(n - 1, 1)
        col = 0; cov = 0
        for k in range(K):                                            # sub-frames: short motion trails, not smears
            Mk = np.clip((fi + (k + 0.5) / K - 0.5) / max(n - 1, 1), 0, 1)
            c, v = splat(*grains(Mk)); col = col + c * v[..., None]; cov = cov + v
        col = col / np.maximum(cov, 1e-6)[..., None]; cov = np.clip(cov / K, 0, 1)[..., None]
        # the sheet under the grains: A while it is still whole, paper once its grains have lifted
        left = np.clip((M - 0.02 - 0.24 * (1 - dA)) / 0.24, 0, 1)[..., None]       # the sheet pales over ~0.7 s
        Af = A if A_seq is None else A_seq[fi]; Bf = B if B_seq is None else B_seq[fi]
        sheet = Af * (1 - left) + paper * left
        frame = sheet * (1 - cov * 0.9) + col * cov * 0.9
        g = np.clip((M - 0.26) / 0.10, 0, 1) * np.clip((0.60 - M) / 0.12, 0, 1)
        frame = np.clip(frame + (1 - frame) * (glow * 0.28 * g)[..., None] * np.array([1.0, 0.93, 0.75], np.float32), 0, 1)
        ub = np.clip((M - 0.37) / 0.63, 0, 1)
        if ub > 0:
            al, rim = bleed.mask(ub)
            frame = frame * (1 - al[..., None]) + (Bf * (1 - 0.16 * rim[..., None])) * al[..., None]
        if fi == 0: frame = Af.copy()
        if fi == n - 1: frame = Bf.copy()
        out.append(np.clip(frame, 0, 1).astype(np.float32))
    return out
