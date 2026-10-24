"""Shot list, transitions and Python preview renderer of the film (0-110 s, 1920x1080, 25 fps), built on edit17:
  - the narration is laid line by line (CUE_ONSET); captions and the movable-type names follow its words
  - no seals and no label slips anywhere
  - 今天，它是浙江农林大学 lands on today's 东湖 gate (a generated print based on a photo), ginkgo leaves drift through it
  - 像一株竹 is the bamboo hero render (the shoot bursts out and races up), graded into the print palette
  - 博览古今 opens on the 木化石 (petrified wood) with the library behind, 学参天地 on the misty library
  - 守护一座森林的未来 is a forest print with light shafts and drifting motes
  - title: the 标准字 in the emblem's green, the years larger; the closing card is a growing ginkgo cross-section (Title19)
  - the opening (AI-1, O1, AI-G, S1) is retimed by its own motion: slow where the picture moves a lot, quick through the still
    paper passages, the speed curve smoothed, so there is no jolt
  .venv_ai/bin/python work/v16k/edit18.py [t0 t1 [out.mov]]   (Python preview; the cut is assembled in DaVinci from export18)"""
import sys, math, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import importlib.util as ilu
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
s = ilu.spec_from_file_location('edit17', ROOT / 'work/v16k/edit17.py'); E17 = ilu.module_from_spec(s); sys.modules['edit17'] = E17; s.loader.exec_module(E17)
E, C, P, S16, PL = E17.E, E17.C, E17.P, E17.S16, E17.PL
Chain = E17.Chain                                            # export18 recognises the era long take by this class
W, H, FPS = 1920, 1080, 25
END = 110.0                                                   # film length (s): the birthday ending holds the last chord
ease, ramp, out_cubic = C.ease, C.ramp, C.out_cubic
kf = E.kf
V18 = ROOT / 'work/v18'
sys.path.insert(0, str(V18)); import bamboo_look as BL
from multiplane import Multiplane
C.TRANS['particles3d'] = lambda A, B, u, **kw: E17.T_pigment(A, B, u, wave='centre')     # preview stand-in (DaVinci gets the 3D render)

# ------------------------------------------------------------------ the narration: where each line's speech begins in the film
CUE_ONSET = {'u01': 1.2, 'u02': 5.8, 'u03': 11.6, 'u04': 16.6, 'u05': 20.23, 'u06': 23.83, 'u07': 28.2, 'u08': 37.8, 'u09': 43.4,
             'u10': 48.2, 'u11': 52.55, 'u12': 59.61, 'u13': 65.8, 'u14': 73.4, 'u15': 81.0, 'u16': 87.8, 'u17': 94.3, 'u18': 103.98}
TIMING = json.load(open(ROOT / 'vo/audition/zwt2_timing.json'))     # u18: 浙江农林大学 生日快乐
WORDS = json.load(open(ROOT / 'vo/zwt/zwt_words.json'))
ONSET_IN_SEG = {'u01': 0.0}                                   # speech begins 0.05 s into every cut line (u01 starts the file)


def cues():
    """[[u, film second of the cut line's first sample]] for audio/v3/mix_final.py"""
    return [[u, round(t - ONSET_IN_SEG.get(u, 0.05), 3)] for u, t in CUE_ONSET.items()]


def word_times():
    """the WT structure of work/v14 (captions, the question, 博览古今 学参天地, the names) for the narration, in film seconds;
    whisper's word times run ~0.35 s early, so each line's words are shifted so that its first word starts with its speech"""
    wt = {}
    for u, t0 in CUE_ONSET.items():
        ws = WORDS[u]; d = ONSET_IN_SEG.get(u, 0.05) - ws[0][1]
        seg0 = t0 - ONSET_IN_SEG.get(u, 0.05)
        wt[u] = {'film_start': t0, 'dur': TIMING[u]['dur'], 'words': [[w, round(seg0 + a + d, 3), round(seg0 + b + d, 3)] for w, a, b in ws if w]}
    return wt


C.WT.clear(); C.WT.update(word_times())
# in the captions 它 is written 他 (the university spoken of as a person)
for _u in E17.LINES: E17.LINES[_u] = [l_.replace('它', '他') for l_ in E17.LINES[_u]]
# every line is cut at its punctuation into phrases, each phrase starting
# where the voice comes out of a pause (work/v18/align_chunks.py); characters spread evenly inside their phrase
CHUNKS = json.load(open(ROOT / 'work/v18/chunks.json')) if (ROOT / 'work/v18/chunks.json').exists() else None
if CHUNKS:
    for u_, rows_ in CHUNKS.items():
        ws_ = []
        for ph_, a_, b_, ts_ in rows_:
            for k_, (c_, t_) in enumerate(zip(ph_, ts_)):
                ws_.append([c_, t_, ts_[k_ + 1] if k_ + 1 < len(ts_) else b_])
        C.WT[u_]['words'] = ws_


def char_t(u, text, ch_index):
    return C.char_times(u, text)[ch_index]


def phrase_times(u, phrase):
    """one time per character of `phrase` inside line u: found in whisper's characters by its best-matching window (whisper may
    mishear one character, e.g. 学餐 for 学参), each character spread inside its word"""
    seq = []
    for w, a, b in C.WT[u]['words']:
        cs = [c for c in w if c not in E17.PUNCT]
        for k, c in enumerate(cs): seq.append((c, a + (b - a) * k / max(len(cs), 1)))
    n = len(phrase); best, bi = -1, 0
    for i in range(len(seq) - n + 1):
        sc = sum(seq[i + k][0] == phrase[k] for k in range(n))
        if sc > best: best, bi = sc, i
    return [seq[bi + k][1] for k in range(n)]


def smooth_luma(A, B, u):
    """a soft 'bright first' dissolve (the preview's stand-in for DaVinci's cross dissolve): wide luminance band + a plain fade"""
    l = C._lum(B) * 0.7 + C._N * 0.3
    m = C._ss(1 - u * 1.6, 1.5 - u * 1.6, l)[..., None]
    w = u * u * (3 - 2 * u)
    m = 0.55 * m + 0.45 * w
    return A * (1 - m) + B * m


C.TRANS['luma'] = smooth_luma


# ------------------------------------------------------------------ retiming by motion
MOTION = json.load(open(V18 / 'motion.json'))


def motion_map(m, fps, s0, s1, dur, p=0.6, floor=1.2, smax=4.0, smin=0.55, sigma=0.35, start_slow=0.0, end_slow=0.0):
    """film-local time -> source time for one clip: each source frame gets film time in proportion to (its motion)^p, smoothed,
    clamped between smin and smax playback speed; start_slow / end_slow lengthen the first / last 0.6 s (eases a join)"""
    i0, i1 = int(round(s0 * fps)), int(round(s1 * fps))
    mm = np.asarray(m[i0 + 1:i1 + 1], np.float64)
    if len(mm) < 2: return np.array([0.0, dur]), np.array([s0, s1])
    w = np.maximum(mm, floor) ** p
    k = max(1, int(sigma * fps)); g = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2); g /= g.sum()
    w = np.convolve(np.pad(w, 3 * k, mode='edge'), g, 'valid')
    idx = np.arange(len(w)) / fps
    w *= 1 + start_slow * np.exp(-idx / 0.6) + end_slow * np.exp(-(idx[-1] - idx) / 0.6)
    for _ in range(30):
        c = dur * fps / w.sum()                                   # film seconds per unit weight per source frame
        w = np.clip(w, 1.0 / (c * smax), 1.0 / (c * smin))
    c = dur * fps / w.sum()
    dt = c * w / fps                                              # film seconds spent on each source frame
    F = np.concatenate([[0.0], np.cumsum(dt)]); S = s0 + np.arange(len(F)) / fps
    S[-1] = s1; F *= dur / F[-1]
    return F, S


CAD = json.load(open(ROOT / 'work/v18/cadence.json')) if (ROOT / 'work/v18/cadence.json').exists() else {}


def true_index(steps, n):
    """the generated clips move in groups of four frames (one step short, the next long; T1b repeats a frame): each frame is put
    back at its true time, the cumulative motion with the cadence smoothed out (work/v18/cadence.py); still clips stay as they are"""
    st = np.asarray(steps[1:n], np.float64)
    if len(st) < 9: return np.arange(n, dtype=np.float64)
    ms = np.convolve(np.pad(st, 4, mode='edge'), np.ones(9) / 9, 'valid')
    w = np.clip(ms / 0.15, 0, 1)
    r = np.clip(st / np.maximum(ms, 1e-6), 0.1, 2.5)
    step = 1 + w * (r - 1)
    tau = np.concatenate([[0.0], np.cumsum(step)])
    return tau * (n - 1) / tau[-1]


class AIShotMap(E.AIShot):
    """an AI shot whose source time follows motion_map (or `knots`: [(film-local s, source s), ...] through a monotone curve) instead
    of a straight line. Outside its slot the clip keeps going (handles for dissolves and transitions): the source time runs on at the
    end speeds, and where the file has no more frames the edge frame keeps the camera move of its first / last frames (a similarity
    fitted to them), so no dissolve ever freezes one side"""
    def __init__(self, name, a, b, first, last=None, src=(0.0, None), knots=None, **mapkw):
        super().__init__(name, a, b, first, last, src=src)
        self.F = None; self.edge = {}
        if self.path.exists():
            if knots is not None:
                from scipy.interpolate import PchipInterpolator
                kf_, ks_ = zip(*knots); pc = PchipInterpolator(kf_, ks_)
                self.F = np.linspace(0, b - a, int(round((b - a) * 100)) + 1); self.S = pc(self.F)
            else:
                pr = MOTION[self.path.stem]
                self.F, self.S = motion_map(pr['m'], pr['fps'], self.s0, self.s1, b - a, **mapkw)
            w = min(0.3, self.F[-1] / 3)
            self.v0 = (np.interp(w, self.F, self.S) - self.S[0]) / w
            self.v1 = (self.S[-1] - np.interp(self.F[-1] - w, self.F, self.S)) / w
    def src_time(self, t):
        if self.F is None: return self.s0
        x = t - self.a
        if x < 0: return float(self.S[0] + x * self.v0)
        if x > self.F[-1]: return float(self.S[-1] + (x - self.F[-1]) * self.v1)
        return float(np.interp(x, self.F, self.S))
    def _edges(self):
        """first and last usable frame: a generated clip's very first (or last) frame is often the reference picture itself and
        jumps against the next one; such a frame is skipped"""
        if not hasattr(self, 'k01'):
            n = len(self.raw); g = lambda i: cv2.cvtColor(cv2.resize(self.raw[i], (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            d = lambda i, j: float(np.abs(g(i) - g(j)).mean())
            k0 = 1 if d(0, 1) > 2.0 * d(1, 2) + 1.0 else 0
            k1 = n - 2 if d(n - 1, n - 2) > 2.0 * d(n - 2, n - 3) + 1.0 else n - 1
            self.k01 = (k0, k1)
        return self.k01
    def _edge_motion(self, end):
        """per-second similarity (log scale, angle, dx, dy about the centre) of the clip's first (end=0) or last (end=1) 5 frames"""
        if end not in self.edge:
            k0_, k1_ = self._edges(); k = 5
            i0, i1 = (k0_, k0_ + k) if end == 0 else (k1_ - k, k1_)
            g0 = cv2.cvtColor(self.raw[i0], cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(self.raw[i1], cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 12); par = (0.0, 0.0, 0.0, 0.0)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
                ok = st.ravel() == 1
                if ok.sum() >= 10:
                    A, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if A is not None:
                        hh, ww = g0.shape; c = np.array([ww / 2, hh / 2])
                        sc = math.hypot(A[0, 0], A[1, 0]); th = math.atan2(A[1, 0], A[0, 0]); d = A[:, :2] @ c + A[:, 2] - c
                        dt = k / self.fps; par = (math.log(sc) / dt, th / dt, d[0] / dt, d[1] / dt)
            self.edge[end] = par
        return self.edge[end]
    def raw_at(self, st):
        """the clip's picture at source second st (native size, BGR uint8), continued past either end"""
        k0_, k1_ = self._edges(); T0, T1 = k0_ / self.fps, k1_ / self.fps
        if T0 <= st <= T1:
            k = st * self.fps; i0 = int(math.floor(k)); f = k - i0; i1 = min(i0 + 1, k1_)
            return self.raw[i0] if f < 0.02 else cv2.addWeighted(self.raw[i0], 1 - f, self.raw[i1], f, 0)
        end = 0 if st < T0 else 1; tau = st - T0 if st < T0 else st - T1
        ls, th, dx, dy = self._edge_motion(end)
        img = self.raw[k0_ if end == 0 else k1_]; hh, ww = img.shape[:2]
        sc = math.exp(ls * tau); a_ = th * tau; R = sc * np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]])
        c = np.array([ww / 2, hh / 2]); M_ = np.hstack([R, (c + np.array([dx, dy]) * tau - R @ c)[:, None]])
        return cv2.warpAffine(img, M_, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    def zoom_at(self, t):
        zr = getattr(self, 'zoom_ramp', None)
        if zr: return zr[2] + (zr[3] - zr[2]) * ease(ramp(t, zr[0], zr[1]))
        return getattr(self, 'zoom', 1.0)
    def _flow(self, i):
        """optical flow raw[i] -> raw[i + 1] at the clip's own size (cached)"""
        if not hasattr(self, 'flows'): self.flows = {}
        if i not in self.flows:
            g0 = cv2.cvtColor(self.raw[i], cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(self.raw[i + 1], cv2.COLOR_BGR2GRAY)
            self.flows[i] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)
            if len(self.flows) > 10: self.flows.pop(next(iter(self.flows)))
        return self.flows[i]
    def _soft(self, i):
        """raw frame i, softened if it is one of the cadence's over-sharp frames"""
        sg = self.sig[i] if hasattr(self, 'sig') else 0.0
        if sg <= 0.05: return self.raw[i]
        if not hasattr(self, 'softc'): self.softc = {}
        if i not in self.softc:
            self.softc[i] = cv2.GaussianBlur(self.raw[i], (0, 0), sg)
            if len(self.softc) > 12: self.softc.pop(next(iter(self.softc)))
        return self.softc[i]
    def _between(self, i0, f):
        """a real in-between frame (both neighbours warped along the flow), not a cross-fade of two positions; a plain blend
        where the flow does not explain the change (fades, cuts inside the clip)"""
        i1 = min(i0 + 1, len(self.raw) - 1)
        R = self._soft
        if i1 == i0: return R(i0)
        Fl = self._flow(i0); hh, ww = Fl.shape[:2]
        if not hasattr(self, 'grid'):
            gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32); self.grid = (gx, gy)
        gx, gy = self.grid
        a = cv2.remap(R(i0), gx - f * Fl[..., 0], gy - f * Fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        b = cv2.remap(R(i1), gx + (1 - f) * Fl[..., 0], gy + (1 - f) * Fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        r0 = R(i0).astype(np.float32); r1 = R(i1).astype(np.float32)
        if np.abs(a - b).mean() > 1.6 * np.abs(r0 - r1).mean() + 4.0:          # only where the flow clearly fails
            return cv2.addWeighted(R(i0), 1 - f, R(i1), f, 0)
        return np.clip((a if f < 0.5 else b) + 0.5, 0, 255).astype(np.uint8)   # the nearest frame moved along the flow (two warped
                                                                                # frames averaged lose their fine detail where they disagree)
    def pic(self, t):
        """the clip at film time t (native size, BGR uint8): flow in-betweens, and where it plays fast a half-frame shutter
        (several in-betweens averaged), so a sped-up shot moves evenly instead of stepping"""
        st = self.src_time(t)
        v = abs(self.src_time(t + 0.02) - self.src_time(t - 0.02)) / 0.04            # source seconds per film second
        span = v * 0.5 / 25                                                          # half a film frame of source time
        n_s = span * self.fps
        k0_, k1_ = self._edges(); T0, T1 = k0_ / self.fps, k1_ / self.fps
        if not hasattr(self, 'tau'):
            cd = CAD.get(self.path.stem); n = len(self.raw)
            self.tau = true_index(cd['steps'], n) if cd and len(cd['steps']) == n else np.arange(n, dtype=np.float64)
            self.sig = [0.0] * n                                                      # no softening: detail is evened out by _steady
        def at(x):
            if T0 <= x <= T1:
                kk = float(np.interp(x * self.fps, self.tau, np.arange(len(self.raw))))
                i0 = int(math.floor(kk)); return self._between(min(i0, k1_), kk - i0)
            return self.raw_at(x)
        if n_s <= 1.2: return self._steady(at(st), st)                              # below 2.5x: in-betweens alone
        m = int(min(5, math.ceil(n_s) + 1))
        acc = sum(at(st + span * (j / (m - 1) - 0.5)).astype(np.float32) for j in range(m))
        return np.clip(acc / m + 0.5, 0, 255).astype(np.uint8)                       # (its motion blur is meant: not re-sharpened)
    def _steady(self, img, st):
        """detail steadied: the clip's frames are not equally sharp (the cadence, and an in-between is softer than a whole frame),
        so the fine detail of every output frame is scaled to the median sharpness of the source frames around it"""
        if not hasattr(self, 'lv'):
            lv = np.array([cv2.Laplacian(cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() for f_ in self.raw])
            self.lv = np.array([np.median(lv[max(0, i - 6):i + 7]) for i in range(len(lv))])
        k_ = int(np.clip(round(st * self.fps), 0, len(self.raw) - 1))
        v = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
        g = float(np.clip(math.sqrt(self.lv[k_] / max(v, 1e-6)), 0.72, 1.65))
        if abs(g - 1) < 0.04: return img
        f32 = img.astype(np.float32); base = cv2.GaussianBlur(f32, (0, 0), 1.2)
        return np.clip(base + (f32 - base) * g + 0.5, 0, 255).astype(np.uint8)
    def frame(self, t):
        if self.F is None: return super().frame(t)
        st = self.src_time(t); z_ = self.zoom_at(t); key = (round(st * 1000), round(z_ * 1e4))
        if key not in self.cache:
            img = self.pic(t)
            if z_ > 1.0:
                hh, ww = img.shape[:2]; ch_, cw_ = int(hh / z_), int(ww / z_)
                img = img[(hh - ch_) // 2:(hh - ch_) // 2 + ch_, (ww - cw_) // 2:(ww - cw_) // 2 + cw_]
            self.cache[key] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
            if len(self.cache) > 6: self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


class Chain18(Chain):
    """the era long take, its segments optionally motion-retimed: seg = (name, a, b, first, last, src[, mapkw])"""
    def __init__(self, segs, xfades=None):
        self.segs = []; self.xf = xfades or {}
        for sg in segs:
            name, a, b, first, last, src = sg[:6]; mk = sg[6] if len(sg) > 6 else None
            sh = AIShotMap(name, a, b, first, last, src=src, **(mk if mk is not None else dict(p=0.0)))
            self.segs.append((a, b, sh))
    def _at(self, t):
        if t < self.segs[0][0]: return self.segs[0][2].frame(t)
        if t >= self.segs[-1][1]: return self.segs[-1][2].frame(t)
        for a, b, sh in self.segs:
            if t < b: return sh.frame(max(t, a))


# ------------------------------------------------------------------ pictures, overlays and transitions defined here
def load_depth(name):
    d = np.asarray(Image.open(V18 / f'depth/{name}_d.png')).astype(np.float32)
    return cv2.GaussianBlur(cv2.resize(d / max(d.max(), 1), (W, H)), (0, 0), 3)


class Motes:
    """drifting specks of light (pollen, dust, falling leaves seen far off): additive soft dots that rise / fall slowly, a few nearer
    and larger and out of focus; deterministic"""
    def __init__(self, n, seed, color, size=(1.2, 4.5), vel=(6, -14), area=(0, 0, W, H), alpha=0.5, leaves=False):
        r = np.random.default_rng(seed)
        self.p = r.uniform([area[0], area[1]], [area[2], area[3]], (n, 2)); self.z = r.random(n) ** 2
        self.v = np.array(vel, np.float32) * (0.4 + self.z[:, None]); self.ph = r.random(n) * 6.28
        self.sz = size[0] + (size[1] - size[0]) * self.z; self.c = np.array(color, np.float32); self.a = alpha; self.leaves = leaves; self.area = area
    def draw(self, img, tl, fade=1.0):
        if fade <= 0: return img
        lay = np.zeros((H, W), np.float32)
        x0, y0, x1, y1 = self.area
        for (px, py), v, ph, sz, z in zip(self.p, self.v, self.ph, self.sz, self.z):
            x = x0 + (px + v[0] * tl + 14 * math.sin(ph + tl * (0.7 + z))) % (x1 - x0); y = y0 + (py + v[1] * tl) % (y1 - y0)
            cv2.circle(lay, (int(round(x * 16)), int(round(y * 16))), int(round(max(1.0, sz) * 16)), float(0.35 + 0.65 * z), -1, cv2.LINE_AA, shift=4)
        lay = cv2.GaussianBlur(lay, (0, 0), 2.0)[..., None] * self.a * fade                # drawn to 1/16 px: no stepping
        return np.clip(img + (1 - img) * lay * self.c, 0, 1) if not self.leaves else img * (1 - lay) + self.c * lay


def rays(shape, origin, angle_deg, n=7, seed=3, width=60):
    """soft light shafts falling from `origin` at angle_deg (0 = straight down), as a 0..1 mask"""
    h, w = shape; yy, xx = np.mgrid[0:h:4, 0:w:4].astype(np.float32)
    a = math.radians(angle_deg); dx, dy = math.sin(a), math.cos(a)
    px = (xx - origin[0]) * dy - (yy - origin[1]) * dx                       # across the shafts
    along = (xx - origin[0]) * dx + (yy - origin[1]) * dy
    r = np.random.default_rng(seed); m = np.zeros_like(xx)
    for k in range(n):
        c = r.uniform(-w * 0.45, w * 0.45); wd = width * r.uniform(0.5, 1.6); st = r.uniform(0.4, 1.0)
        m += st * np.exp(-((px - c) / wd) ** 2)
    m *= np.clip(along / (h * 0.9), 0, 1) ** 0.6 * np.exp(-np.clip(along, 0, None) / (h * 1.6))
    return cv2.resize(np.clip(m, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)


class Parallax18(S16.Parallax):
    """the 2.5D move at constant speed (no ease in or out): a shot is already moving when it dissolves in and still moving when it
    dissolves out, and the move runs on through the handles (no stop-start at the joins)"""
    def frame(self, t_local, img=None):
        """rendered at twice the size and reduced, so a slow move does not make fine hatching blink sharp and soft; img: draw this
        picture (same size as the print) through the same move, e.g. the print with something composited on it"""
        u = float(np.clip(t_local / self.dur, -0.4, 1.4)) if self.dur > 0 else 0.0
        dx, dy, dz = self.move
        if not hasattr(self, 'xx2'):
            yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32); self.xx2, self.yy2 = xx, yy
            self.d2 = cv2.resize(self.d, (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR)
        src = self.img if img is None else img
        if not hasattr(self, 'up') or img is not None:
            up = cv2.resize(src, (2 * W, 2 * H), interpolation=cv2.INTER_LANCZOS4)
            if img is None: self.up = up
        else: up = self.up
        d = self.d2
        z = 1 + dz * u * (0.6 + 0.8 * d)
        sx = (self.xx2 - W) / z + W - (dx + self.shift[0] * (d - 0.5)) * u * 2
        sy = (self.yy2 - H) / z + H - (dy + self.shift[1] * (d - 0.5)) * u * 2
        out = cv2.remap(up, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return np.clip(cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA), 0, 1).astype(np.float32)


class Planet18(PL.Planet):
    """0-5.6: never still: one continuous push-in from the first frame that grows into the dive (log
    zoom through a monotone cubic), the planet turning; the unroll eases in but keeps some speed at its end, and its last frame is
    AI-1's first frame itself (the decoded frame is the print), so AI-1 follows on a plain cut, already moving"""
    V_END = 0.30                                                            # zoom speed (log / s) handed to AI-1
    def __init__(self):
        super().__init__(ROOT / 'libtv_jobs/out/AI-1_f0.png')
        from scipy.interpolate import CubicHermiteSpline
        s_p = 0.42 * H / (self.Wp / (2 * math.pi) + H); self.s_p = s_p
        L0, Ls = math.log(s_p * 0.76), math.log(s_p)
        self.L = CubicHermiteSpline([0.0, 3.3, 5.6], [L0, Ls, 0.0], [0.05, 0.14, self.V_END])
    def frame(self, t):
        """three sub-frames over half a frame: the planet carries the same motion blur as the clip it cuts to"""
        if t < 2.6: return self._frame(t)
        img = sum(self._frame(t + d) for d in (-0.005, 0.0, 0.005)) / 3
        g = 1 + 0.24 * ease(ramp(t, 4.9, 5.6))                             # the flat print as crisp as AI-1's own frames at the cut
        if g > 1.005:
            base = cv2.GaussianBlur(img, (0, 0), 1.2); img = np.clip(base + (img - base) * g, 0, 1)
        return img
    def _frame(self, t):
        if t >= self.END - 1e-6:                                            # handle after the cut: the flat print keeps the push
            s = math.exp(self.V_END * (t - self.END))
            M_ = np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]], np.float32)
            return cv2.warpAffine(self.flat, M_, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = ramp(t, 2.7, self.T_UNROLL1)
        u = 2.5 * x * x - 1.5 * x ** 3                                      # eases in; at its end still ~a third of its top speed
        k = (1 - u) ** 1.6
        rho0 = self.Wp / (2 * math.pi * max(k, 1e-3)); rho1 = self.Wp / (2 * math.pi)
        s = math.exp(float(self.L(min(max(t, 0.0), self.END))))
        by = (H / 2 + 20 - s * rho1) * (1 - u) + H * u
        piv = np.array([W / 2, by + s * rho0]) if k > 1e-3 else np.array([W / 2, by])
        spin = -260 * (1 - u) ** 2 * t                                     # the turn comes to rest exactly as the print lies flat
        xm, ym = self.warp(k, s, piv, spin)
        xm = np.mod(xm, self.Wp).astype(np.float32); ym = ym.astype(np.float32)
        img = cv2.remap(self.pano, xm, ym, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
        outside = (img[..., 0] < 0)[..., None]
        return np.clip(np.where(outside, self.paper, img), 0, 1).astype(np.float32)


class _Leaves:
    """ginkgo leaves as flat woodblock shapes: a fan on a short stalk, gold to ochre, a thin darker edge"""
    COLS = np.array([[0.95, 0.76, 0.24], [0.90, 0.66, 0.18], [0.97, 0.84, 0.40], [0.86, 0.58, 0.16], [0.93, 0.72, 0.30]], np.float32)
    def __init__(self, n, seed):
        r = np.random.default_rng(seed); self.n = n
        self.size = r.uniform(13, 30, n); self.rot0 = r.uniform(0, 360, n); self.spin = r.uniform(-260, 260, n)
        self.flip = r.uniform(0, 6.28, n); self.flips = r.uniform(2, 7, n)
        self.col = self.COLS[r.integers(0, len(self.COLS), n)] * r.uniform(0.92, 1.05, (n, 1))
        a = np.radians(np.linspace(-132, -48, 13))
        fan = np.stack([np.cos(a), np.sin(a)], 1); fan[6] *= 0.72                # the notch in the middle of the rim
        self.shape = np.concatenate([[[0.0, 0.0]], fan, [[0.0, 0.0]], [[0.04, 0.0], [0.04, 0.55], [-0.04, 0.55], [-0.04, 0.0]]])
    def draw(self, col_l, a_l, xs, ys, t, alpha):
        for i in np.argsort(self.size):
            if alpha[i] <= 0.02: continue
            r_ = np.radians(self.rot0[i] + self.spin[i] * t); fx = math.cos(self.flip[i] + self.flips[i] * t)
            pts = self.shape * self.size[i] * np.array([max(abs(fx), 0.18), 1.0])
            R = np.array([[math.cos(r_), -math.sin(r_)], [math.sin(r_), math.cos(r_)]])
            P = (pts @ R.T + [xs[i], ys[i]]) * 16
            P = P.astype(np.int32)[None]
            c = self.col[i] * (0.82 + 0.18 * abs(fx))
            cv2.fillPoly(col_l, P, tuple(float(v) for v in c), cv2.LINE_AA, shift=4)
            cv2.polylines(col_l, P, True, tuple(float(v) * 0.62 for v in c), 1, cv2.LINE_AA, shift=4)
            cv2.fillPoly(a_l, P, float(alpha[i]), cv2.LINE_AA, shift=4)


_LW = {}
def T_leafwipe(A, B, u, seed=11, n=210, K=5, dur=1.4):
    """a gust of ginkgo leaves crosses the frame left to right; just behind its front the picture has turned to B (a narrow ragged
    edge, hidden by the leaves), the leaves tumbling, near ones larger and faster, motion-blurred over half a frame, a soft shadow
    under them"""
    if seed not in _LW:
        r = np.random.default_rng(seed); L = _Leaves(n, seed + 1)
        dep = r.uniform(0, 1, n); L.size = 20 + 46 * dep ** 1.5
        nz = cv2.GaussianBlur(r.standard_normal((H // 8 + 2, 1)).astype(np.float32), (0, 0), 2)
        nz = cv2.resize(nz, (1, H), interpolation=cv2.INTER_CUBIC)[:, 0] * 60
        _LW[seed] = dict(L=L, y=r.uniform(-60, H + 60, n), off=r.normal(-30, 150, n), sp=0.86 + 0.28 * dep,
                         wob=r.uniform(10, 36, n), ph=r.uniform(0, 6.28, n), nz=nz, dep=dep)
    d = _LW[seed]; L = d['L']
    def front(v): return -0.25 * W + 1.50 * W * (v * v * (3 - 2 * v))
    xf = front(u)
    xx = np.arange(W, dtype=np.float32)[None, :]
    m = np.clip((xf - 40 + d['nz'][:, None] - xx) / 80 + 0.5, 0, 1)
    m = m * m * (3 - 2 * m)
    base = A * (1 - m[..., None]) + B * m[..., None]
    col = np.zeros((H, W, 3), np.float32); acc = np.zeros((H, W), np.float32)
    sh_u = 0.5 / (dur * 25)                                                     # half-frame shutter
    for k in range(K):
        v = float(np.clip(u + (k / (K - 1) - 0.5) * sh_u, 0, 1))
        xs = front(v) * d['sp'] + d['off'] + d['wob'] * np.sin(d['ph'] + 6 * v)
        ys = d['y'] + 50 * np.sin(d['ph'] * 1.7 + 2.5 * v) + 70 * v * (d['dep'] - 0.3)
        alpha = np.clip(1.0 - np.abs(xs - xf) / 700, 0, 1) ** 0.7 * np.clip(min(u, 1 - u) * 7, 0, 1)
        cl = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
        L.draw(cl, al, xs, ys, v * dur, alpha)
        col += cl * al[..., None]; acc += al
    col = col / np.maximum(acc, 1e-6)[..., None]; acc /= K
    sh = np.roll(np.roll(cv2.GaussianBlur(acc, (0, 0), 8), 10, 0), 7, 1)
    out = base * (1 - 0.30 * sh[..., None])
    return np.clip(out * (1 - acc[..., None]) + col * acc[..., None], 0, 1)


_MIST = {}
def T_mist(A, B, u, drift=(90, -50), seed=5):
    """the camera goes down through a bank of mist: warm paper-white, soft large-scale fBm, it thickens from the bottom until the
    frame is veiled (A turns into B under it) and thins away upward (9.9: the ink valley -> into the valley)"""
    if seed not in _MIST:
        r = np.random.default_rng(seed); h4, w4 = (H + 400) // 4, (W + 800) // 4
        def octave(sig):
            n_ = cv2.GaussianBlur(r.standard_normal((h4, w4)).astype(np.float32), (0, 0), sig)
            return n_ / (n_.std() + 1e-6)
        L1 = 0.62 * octave(26) + 0.30 * octave(12) + 0.16 * octave(5)      # broad banks (blurred at 1/4 size: no grid)
        L2 = 0.55 * octave(16) + 0.35 * octave(7)                            # a faster, finer layer in front
        up = lambda a_: cv2.resize(a_, (W + 800, H + 400), interpolation=cv2.INTER_CUBIC)
        _MIST[seed] = (up(L1 / L1.std()), up(L2 / L2.std()))
    N1, N2 = _MIST[seed]
    o1x, o1y = int(400 + drift[0] * (u - 0.5)), int(200 + drift[1] * (u - 0.5))
    o2x, o2y = int(400 + 2.6 * drift[0] * (u - 0.5)), int(200 + 2.2 * drift[1] * (u - 0.5))
    n1 = 0.65 * N1[o1y:o1y + H, o1x:o1x + W] + 0.35 * N2[o2y:o2y + H, o2x:o2x + W]
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    dens = math.sin(math.pi * u) ** 0.75 * (1 + 0.35 * (yy - 0.5) * (1 - 2 * u))      # low first coming in, high last going out
    fog = np.clip(dens * 1.05 + 0.18 * np.tanh(n1) - 0.12, 0, 0.80)                 # the picture always shows through
    sw = np.clip((u - 0.5) * 4.0 - 0.25 * np.tanh(n1) + 0.5, 0, 1); sw = sw * sw * (3 - 2 * sw)
    base = A * (1 - sw[..., None]) + B * sw[..., None]
    mist = np.array([0.945, 0.925, 0.875], np.float32) * (0.985 + 0.03 * np.tanh(n1)[..., None])
    return np.clip(base * (1 - fog[..., None]) + mist * fog[..., None], 0, 1)


_RP = {}
def T_reprint(A, B, u, seed=9):
    """the old picture lifts off the sheet (its lights first, unevenly, like a print being washed off), the new one is printed in its
    place: its ink (the darks) first, then its colour soaks in from the middle (24.0: the 浙江林学院 iron gate -> today's gate)"""
    if seed not in _RP:
        r = np.random.default_rng(seed)
        n_ = cv2.resize(r.standard_normal((H // 24 + 2, W // 24 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        n_ = cv2.GaussianBlur(n_, (0, 0), 6); n_ = (n_ - n_.min()) / (n_.max() - n_.min() + 1e-6)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        R = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H * 0.55) / (H / 2)) ** 2) / 1.42
        nl = cv2.resize(r.standard_normal((6, 9)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        nl = cv2.GaussianBlur(nl, (0, 0), 40); nl = (nl - nl.min()) / (nl.max() - nl.min() + 1e-6)
        _RP[seed] = dict(N=n_, NL=nl, R=R, paper=S16.paper())
    d = _RP[seed]; N, NL, R, paper = d['N'], d['NL'], d['R'], d['paper']
    lA = A @ np.array([0.2126, 0.7152, 0.0722], np.float32); lB = B @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    f = ramp(u, 0.0, 0.52) * 1.7 - 0.35
    ma = np.clip((f - (0.8 * NL + 0.2 * (1 - lA))) / 0.55 + 0.5, 0, 1); ma = ma * ma * (3 - 2 * ma)       # a broad soft wash
    sheet = A * (1 - ma[..., None]) + paper * ma[..., None]
    p = ease(ramp(u, 0.2, 0.62)); ink = np.clip(1.1 - lB * 1.25, 0, 1)
    printed = sheet * (1 - (0.85 * p) * ink[..., None] * (0.85 + 0.15 * N[..., None]))
    c = ramp(u, 0.42, 1.0)
    if c <= 0: return printed
    m = np.clip((c * 1.35 - 0.18 - (0.5 * N + 0.5 * R)) / 0.2 + 0.5, 0, 1); m = (m * m * (3 - 2 * m))[..., None]
    return np.clip(printed * (1 - m) + B * m, 0, 1)


def T_light(A, B, u, warm=(1.0, 0.90, 0.66), peak=0.74):
    """the forest's light swells until the picture swims in warm light, and the next picture comes out of it (a light-dissolve: the
    double image of a plain dissolve is hidden in the brightness)"""
    g = math.sin(math.pi * u) ** 1.4 * peak
    m = ramp(u, 0.34, 0.66); m = m * m * (3 - 2 * m)                       # the change happens inside the brightest part
    base = A * (1 - m) + B * m
    lum = base @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    k = g * (0.55 + 0.45 * np.clip(lum, 0, 1))[..., None]                     # the lights go first, the shadows follow
    return np.clip(base + (1 - base) * k * np.array(warm, np.float32), 0, 1)


_LK = {}
def T_lumakey(A, B, u, seed=4):
    """'bright first': the new picture appears in its own light places first (snow, cloud, mist, sky, white walls), where the old
    picture is light too, then in the mid tones, the shadows last; soft wide edges, a little plain fade under it, so it never draws
    hard contour lines (62.6 古 -> the misty library, 80.0 the gold of the ginkgo avenue)"""
    if seed not in _LK:
        r = np.random.default_rng(seed)
        n_ = cv2.GaussianBlur(cv2.resize(r.standard_normal((H // 16, W // 16)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 20)
        _LK[seed] = (n_ - n_.min()) / (n_.max() - n_.min() + 1e-6)
    wl = np.array([0.2126, 0.7152, 0.0722], np.float32)
    key = 0.55 * (B @ wl) + 0.25 * (A @ wl) + 0.20 * _LK[seed]
    key = cv2.GaussianBlur(key, (0, 0), 8)
    lo, hi = float(np.percentile(key, 2)), float(np.percentile(key, 98)); key = np.clip((key - lo) / (hi - lo + 1e-6), 0, 1)
    th = 1.25 - 1.5 * u
    m = np.clip((key - th) / 0.35 + 0.5, 0, 1); m = m * m * (3 - 2 * m)
    w = u * u * (3 - 2 * u)
    m = (0.7 * m + 0.3 * w)[..., None]
    return np.clip(A * (1 - m) + B * m, 0, 1)


C.TRANS['leafwipe'] = T_leafwipe
C.TRANS['mist'] = T_mist
C.TRANS['light'] = T_light
C.TRANS['lumakey'] = T_lumakey
C.TRANS['reprint'] = T_reprint


class Plants18:
    """一草一木 as a collage: the campus seen from above becomes the first card (it shrinks onto the sheet, still moving),
    then four plant prints slide onto it, opaque cut paper with a paper edge and a soft shadow, each landing on a beat"""
    NAMES = ['K_plant_lotus', 'K_plant_ginkgo', 'K_plant_redleaf_snow', 'K_plant_blossom_snow']
    def __init__(self, base_fn, t0, lands):
        self.base, self.t0, self.lands = base_fn, t0, lands
        self.ims = [S16.load(kf(n)) for n in self.NAMES]; self.paper = S16.paper()
    def _card(self, img, im, s, rot, dy, a, shk, shift=(0.0, 0.0)):
        M_ = cv2.getRotationMatrix2D((W / 2, H / 2), rot, s); M_[1, 2] += dy + shift[1]; M_[0, 2] += shift[0]
        card = cv2.warpAffine(im, M_, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
        m = (card[..., 0] >= 0).astype(np.float32)
        edge = cv2.dilate(m, np.ones((13, 13), np.uint8)) - m
        sh = cv2.GaussianBlur(np.clip(m + edge, 0, 1), (0, 0), 16); sh = np.roll(np.roll(sh, 12, 0), 9, 1)
        img = img * (1 - 0.30 * sh[..., None] * shk)
        img = img * (1 - edge[..., None] * a) + self.paper * edge[..., None] * a
        return img * (1 - m[..., None] * a) + np.where(m[..., None] > 0, card, 0) * a
    def frame(self, t):
        u0 = ease(ramp(t, self.t0, self.t0 + 0.6))
        if u0 <= 0: return self.base(t)
        drop = lambda L: ease(ramp(t, L - 0.4, L))
        later = sum(drop(L) for L in self.lands)
        img = self._card(self.paper.copy(), self.base(t), (1 - 0.11 * u0) * (1 - 0.04 * later), -1.0 * u0 - 0.6 * later, 0, 1.0, u0)
        # each print slides in whole from outside the frame (right, left, below, right) and lands on its beat; it is opaque
        # all the way, so no frame ever shows two pictures through each other
        dirs = [(1, 0), (-1, 0), (0, 1), (1, 0)]
        for k, (im, L) in enumerate(zip(self.ims, self.lands)):
            if t < L - 0.6: continue
            v = ease(ramp(t, L - 0.6, L)); lat = sum(drop(LL) for LL in self.lands[k + 1:])
            s_ = 0.93 * (1 - 0.04 * lat)
            rot = (1 if k % 2 else -1) * (3.0 * (1 - v) + 1.0 * lat)
            dx, dy = dirs[k % 4]
            img = self._card(img, im, s_, rot, 0.0, 1.0, 1.0, shift=(dx * (1 - v) * W * 1.02, dy * (1 - v) * H * 1.05))
        return img


class ResearchSamara18(E.ResearchSamara):
    """the study plate labelled as a botanical plate is: the name in small ink type at the end of its hairline, no slip, no seal"""
    def __init__(self, png):
        super().__init__(png)
        self.slips = {k: C.glyph_row(k, 30, (46, 36, 28, 255)) for k in self.slips}
    def frame(self, t):
        img = self.pl.frame(t - 66.8)
        ink = P.Ink()
        for name, a, b, ts in self.anchor:
            ink.line([a, b], 1.1, taper=False, prog=ease(ramp(t, ts, ts + 0.25)))
        ink.onto(img, {'ink': P.INK})
        for name, a, b, ts in self.anchor:
            if t >= ts + 0.15:
                g = self.slips[name]; gh, gw = g.shape[:2]
                y = b[1] - gh - 2 if b[1] < a[1] else b[1] + 4
                C.over(img, g, b[0] - gw / 2, y, ease(ramp(t, ts + 0.15, ts + 0.35)))
        return img


class MistBand:
    """a low bank of mist drifting across a print (soft fBm, warm paper white), its own speed and direction: a plane in front of the
    picture that needs no cut-out"""
    def __init__(self, y_c=0.8, y_s=0.09, strength=0.24, vx=26.0, seed=12):
        r = np.random.default_rng(seed); h4, w4 = H // 4, (W + 600) // 4
        def oc(sig):
            n_ = cv2.GaussianBlur(r.standard_normal((h4, w4)).astype(np.float32), (0, 0), sig); return n_ / (n_.std() + 1e-6)
        n_ = 0.7 * oc(14) + 0.3 * oc(6)
        self.N = cv2.resize(n_, (W + 600, H), interpolation=cv2.INTER_CUBIC)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        self.band = np.exp(-((yy - y_c) / y_s) ** 2); self.k = strength; self.vx = vx
    def draw(self, img, tl, fade=1.0):
        ox = 300 - self.vx * tl; x0 = int(math.floor(ox)); f = ox - x0
        x0 = int(np.clip(x0, 0, 598))
        n_ = self.N[:, x0:x0 + W] * (1 - f) + self.N[:, x0 + 1:x0 + 1 + W] * f
        a = np.clip(self.band * (0.55 + 0.45 * np.tanh(n_)), 0, 1) * self.k * fade
        return img * (1 - a[..., None]) + np.array([0.95, 0.93, 0.88], np.float32) * a[..., None]


class Gate:
    """24.0-27.4 今天，它是浙江农林大学: today's East Lake gate (a generated print based on a photo), one slow push along the avenue (the
    whole print, supersampled), ginkgo leaves drifting down through the warm light"""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.pl = Parallax18(V18 / 'K18_gate_donghu.png', move=(0, -10, 0.07), dur=b - a + 1.2)
        self.leaves = Motes(26, 21, (0.93, 0.70, 0.22), size=(3, 9), vel=(-18, 46), alpha=0.85, leaves=True)
    def frame(self, t):
        tl = t - self.a + 0.6
        return self.leaves.draw(self.pl.frame(tl), tl, ease(ramp(tl, 0.3, 1.2)))


class Forest:
    """69.4-73.0 我们也守护一座森林的未来: a forest print, a slow walk in (the whole print, supersampled, no cut-out layers, so
    nothing stretches), light shafts breathing, dust in them, a few leaves falling close to the camera"""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.pl = Parallax18(V18 / 'K18_forest.png', move=(-16, 0, 0.10), dur=b - a + 1.2)
        self.rm = rays((H, W), (W * 0.82, -80), 28, n=8, seed=5, width=55)
        self.motes = Motes(70, 9, (1.0, 0.92, 0.65), size=(1.0, 3.5), vel=(-4, -6), alpha=0.55)
        self.leaves = Motes(9, 33, (0.80, 0.70, 0.22), size=(4, 12), vel=(-14, 40), alpha=0.75, leaves=True)
    def frame(self, t):
        tl = t - self.a + 0.6
        img = self.pl.frame(tl)
        b = 0.16 * (0.75 + 0.25 * math.sin(tl * 1.3))
        img = np.clip(img + (1 - img) * self.rm[..., None] * b * np.array([1.0, 0.9, 0.62], np.float32), 0, 1)
        return self.leaves.draw(self.motes.draw(img, tl), tl)


class Petrified:
    """59.6-62.6 我们博览古今: the 木化石 trunks (a generated print based on a photo) with the library behind; a slow truck to the left
    (the whole print, supersampled), and a thin bank of mist drifting the other way over the lake and the platform"""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.pl = Parallax18(V18 / 'K18_petrified.png', move=(-36, 0, 0.05), dur=b - a + 1.4)
        self.mist = MistBand(y_c=0.80, y_s=0.07, strength=0.26, vx=-30.0)
    def frame(self, t):
        tl = t - self.a + 0.7
        return self.mist.draw(self.pl.frame(tl), tl)


class Bamboo:
    """42.6-48.3 像一株竹，坚韧不拔，不断超越: the bamboo hero render (no text): the shoot bursts out of the soil and races up,
    the camera rises with it through the moso forest. Retimed by its own motion (the burst a little slow, the climb fast), and
    graded into the print palette by a fixed colour transfer to the woodblock bamboo"""
    SRC = Path('~/Movies/ZAFU/renders/v18_bamboo.mov')
    def __init__(self, a, b, src=(0.45, 16.0)):
        self.a, self.b = a, b
        cap = cv2.VideoCapture(str(self.SRC)); self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.cap = cap; self.n = n
        prof = MOTION.get('v18_bamboo')
        if prof is None:
            prof = {'fps': self.fps, 'm': self._motion()}; MOTION['v18_bamboo'] = prof
            json.dump(MOTION, open(V18 / 'motion.json', 'w'))
        self.F, self.S = motion_map(prof['m'], self.fps, src[0], src[1], b - a, p=0.55, floor=2.0, smax=4.2, smin=1.0, start_slow=0.8)
        self.v1 = (self.S[-1] - np.interp(self.F[-1] - 0.3, self.F, self.S)) / 0.3
        ref = cv2.cvtColor((S16.load(kf('K_bamboo_up')) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        mid = self._read(int(n * 0.55)); lab = cv2.cvtColor(mid, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.mu_s, self.sd_s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-3
        self.mu_r, self.sd_r = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0) + 1e-3
        self.cache = {}
    def _read(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(i, 0), self.n - 1)); ok, fr = self.cap.read()
        return cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    def _motion(self):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST); prev = None; m = [0.0]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(self.n):
            ok, fr = self.cap.read()
            if not ok: break
            g = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY)
            if prev is not None: m.append(float(np.percentile(np.linalg.norm(dis.calc(prev, g, None), axis=-1), 90)))
            prev = g
        return m
    def _graded(self, i):
        if i not in self.cache:
            self.cache[i] = BL.look(self._read(i).astype(np.float32) / 255)
            if len(self.cache) > 4: self.cache.pop(next(iter(self.cache)))
        return self.cache[i]
    def src_time(self, t):
        x = t - self.a
        if x > self.F[-1]: return min(float(self.S[-1] + (x - self.F[-1]) * self.v1), (self.n - 2) / self.fps)
        return float(np.interp(x, self.F, self.S))
    def frame(self, t):
        k = self.src_time(t) * self.fps; i0 = int(math.floor(k)); f = k - i0
        a = self._graded(i0)
        return a * (1 - f) + self._graded(i0 + 1) * f if f > 0.05 and i0 + 1 < self.n else a


class Title18(E.E.Title):
    """the title: the 标准字 in the emblem's own green, the years '1958 – 2026' larger"""
    GREEN = np.array([0.0, 146 / 255, 71 / 255], np.float32)
    def __init__(self):
        super().__init__()
        w = self.lock.shape[1]
        col = np.arange(w)[None, :, None] > int(0.215 * w)                       # right of the emblem: the calligraphy only
        self.lock[..., :3] = np.where(col, self.GREEN, self.lock[..., :3])     # same coverage (alpha), the emblem's green
        self.years = C.glyph_row('1958 – 2026', 50, E.E.INKC)                   # an en dash between the years


class Title19:
    """the closing card, a birthday card for the 68th anniversary (festive, animated ginkgo leaves at the sides and behind; the last
    line 浙江农林大学 生日快乐). The idea: a tree grows one ring a year. From 101.0 the cross-section of an old ginkgo
    (the generated print K19_title_slice) grows outward from its pith, ring after ring, a thin gold light at the growing edge; the outermost
    ring and the bark are reached on 106.0, the full chord of the score and 生日快乐: a wave of gold light runs out across the rings,
    ginkgo leaves lift from the rim and fall, and 10.25 appears between two gold sprigs. Two ginkgo branches sway at the corners,
    leaves fall behind the name and, out of focus, in front; gold dust rises. The name is printed with the voice (103.98)."""
    T_BURST = 106.0
    def __init__(self):
        Ld = lambda n: np.asarray(Image.open(ROOT / f'codex/out/{n}.png').convert('RGBA'), np.float32) / 255
        sl = np.asarray(Image.open(ROOT / 'codex/out/K19_title_slice.png').convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
        self.slice = sl
        pap = S16.paper(); mref = np.median(sl[:, :90].reshape(-1, 3), 0)               # the card's own paper colour
        self.paper = np.clip(pap / pap.reshape(-1, 3).mean(0) * mref * np.array([1.02, 0.985, 0.94], np.float32), 0, 1)     # a touch warmer
        # normalised radius: 0 at the pith, 1 at the bark (the pith sits up-left of the bark's centre; blend the two centres)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        c0, c1, rx, ry = np.array([906.0, 469.0]), np.array([958.0, 533.0]), 812.0, 683.0
        r = np.hypot((xx - c1[0]) / rx, (yy - c1[1]) / ry)
        for _ in range(3):                                                           # centre moves with the radius
            c = c0 + (c1 - c0) * np.clip(r, 0, 1)[..., None]
            r = np.hypot((xx - c[..., 0]) / rx, (yy - c[..., 1]) / ry)
        rng = np.random.default_rng(19)
        nz = cv2.GaussianBlur(rng.standard_normal((H // 8, W // 8)).astype(np.float32), (0, 0), 2)
        self.r = r + 0.012 * cv2.resize(nz / nz.std(), (W, H), interpolation=cv2.INTER_CUBIC)
        # the name, the years, the date, the sprigs
        lk = Image.open(ROOT / 'data/logo/A04_lockup.png').convert('RGBA'); w = 1260; h = int(lk.height * w / lk.width)
        self.lock = np.asarray(lk.resize((w, h), Image.LANCZOS), np.float32) / 255
        col = np.arange(w)[None, :, None] > int(0.215 * w)
        self.lock[..., :3] = np.where(col, Title18.GREEN, self.lock[..., :3])
        self.years = C.glyph_row('1958 – 2026', 46, E.E.INKC)
        self.date = C.glyph_row('10.25', 80, (0, 120, 62, 255))
        sp = Ld('K19_sprig'); ys, xs = np.where(sp[..., 3] > 0.03); sp = sp[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sc = 150 / sp.shape[1]; sp = cv2.resize(sp, (150, int(sp.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        self.sprig_l, self.sprig_r = sp, sp[:, ::-1].copy()
        # the two branches (rigid, swaying about the corner they come in from)
        self.br = []
        for n, wpx, anchor_, pivot, ph, blur in (('K19_branch_a', 700, (-95, -85), 'tl', 0.0, 0.0), ('K19_branch_b', 760, None, 'br', 1.7, 1.6)):
            b = Ld(n); hh = int(b.shape[0] * wpx / b.shape[1]); b = cv2.resize(b, (wpx, hh), interpolation=cv2.INTER_AREA)
            if blur: b = cv2.GaussianBlur(b, (0, 0), blur)
            pos = anchor_ if anchor_ else (W - wpx + 70, H - hh + 60)
            piv = (pos[0], pos[1]) if pivot == 'tl' else (pos[0] + wpx, pos[1] + hh)
            self.br.append(dict(img=b, pos=pos, piv=piv, ph=ph))
        # single leaves from the sprite sheet
        sh = Ld('K19_leaves'); n_, lab, st, _ = cv2.connectedComponentsWithStats((sh[..., 3] > 0.05).astype(np.uint8), 8)
        self.leaf = []
        for k in range(1, n_):
            if st[k, cv2.CC_STAT_AREA] < 2000: continue
            x0, y0, ww, hh = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
            spr = sh[y0:y0 + hh, x0:x0 + ww].copy(); spr[..., 3] *= (lab[y0:y0 + hh, x0:x0 + ww] == k)
            self.leaf.append(spr)
        r_ = np.random.default_rng(25)
        def flock(n, size, speed, area, blur, seed_t=0.0):
            return [dict(s=self.leaf[r_.integers(len(self.leaf))], size=r_.uniform(*size), x=r_.uniform(*area[0]), y=r_.uniform(*area[1]),
                         vy=r_.uniform(*speed), vx=r_.uniform(-12, 18), ph=r_.uniform(0, 6.28), fl=r_.uniform(1.2, 2.6), rot0=r_.uniform(0, 360),
                         spin=r_.uniform(-50, 50), blur=blur) for _ in range(n)]
        self.back = flock(16, (52, 84), (40, 72), ((150, 1800), (-500, 900)), 0.5)
        self.front = flock(5, (120, 175), (30, 48), ((-60, 1980), (-700, 700)), 6.5)
        for f in self.front:                                                        # keep the big ones off the words
            if 330 < f['x'] < 1590: f['x'] = f['x'] * 0.17 if f['x'] < 960 else 1600 + (f['x'] - 960) * 0.4
        # the leaves that lift off the rim on 生日快乐
        self.burst = []
        for k in range(36):
            a = r_.uniform(0, 2 * math.pi); rr = 0.95
            x0 = c1[0] + rx * rr * math.cos(a); y0 = c1[1] + ry * rr * math.sin(a)
            self.burst.append(dict(s=self.leaf[k % len(self.leaf)], size=r_.uniform(58, 104), x0=x0, y0=y0, dx=math.cos(a) * r_.uniform(180, 380),
                                   dy=math.sin(a) * r_.uniform(90, 200) - r_.uniform(120, 260), ph=r_.uniform(0, 6.28), fl=r_.uniform(2, 3.5),
                                   rot0=r_.uniform(0, 360), spin=r_.uniform(-160, 160), delay=r_.uniform(0, 0.35)))
        self.dust = Motes(70, 31, (1.0, 0.82, 0.42), size=(1.2, 3.6), vel=(4, -22), alpha=0.7)
        # glints: points on the rings that flash as the wave of gold light passes them
        self.glint = [(r_.uniform(0.08, 1.0), r_.uniform(0, 2 * math.pi), r_.uniform(1.5, 4.0)) for _ in range(140)]
        self.c0, self.c1, self.rx, self.ry = c0, c1, rx, ry

    def _leaf(self, img, L, x, y, rot, flip, alpha, size):
        s_ = L['s'] if 's' in L else L
        sc = size / max(s_.shape[:2]); fx = max(abs(math.cos(flip)), 0.22)
        h0, w0 = s_.shape[:2]
        M_ = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), rot, 1.0)
        M_[:, :2] = M_[:, :2] @ np.diag([sc * fx, sc]); M_[0, 2] = x - (M_[0, 0] * w0 / 2 + M_[0, 1] * h0 / 2); M_[1, 2] = y - (M_[1, 0] * w0 / 2 + M_[1, 1] * h0 / 2)
        pad = int(size)
        x0_, y0_ = int(x - pad), int(y - pad); x1_, y1_ = int(x + pad), int(y + pad)
        if x1_ < 0 or y1_ < 0 or x0_ >= W or y0_ >= H: return
        M_[0, 2] -= x0_; M_[1, 2] -= y0_
        spr = cv2.warpAffine(s_, M_, (x1_ - x0_, y1_ - y0_), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        if L.get('blur', 0) > 0.3: spr = cv2.GaussianBlur(spr, (0, 0), L['blur'])
        spr[..., :3] *= 0.86 + 0.14 * abs(math.cos(flip))                         # the side turned away is a little darker
        C.over(img, spr, x0_, y0_, alpha)

    def _fall(self, img, flock, t, span):
        for L in flock:
            tt = t - 100.6
            y = (L['y'] + L['vy'] * tt) % span[1] - 120; x = L['x'] + L['vx'] * tt + 30 * math.sin(L['ph'] + 0.9 * tt)
            self._leaf(img, L, x, y, L['rot0'] + L['spin'] * tt, L['ph'] + L['fl'] * tt, 0.95, L['size'])

    def frame(self, t):
        u = ramp(t, 100.8, self.T_BURST); R = 0.30 + 0.75 * (0.35 * u + 0.65 * u * u * (3 - 2 * u))   # a young trunk at first, grows steadily, lands on 106.0
        m = np.clip((R - self.r) / 0.018 + 0.5, 0, 1)
        img = self.paper * (1 - m[..., None]) + self.slice * m[..., None]
        if R < 1.06:                                                                # the growing edge: a thin warm light
            band = np.exp(-((self.r - R) / 0.014) ** 2) * (1 - ramp(R, 1.0, 1.06))
            img = img + (1 - img) * (band * 0.55)[..., None] * np.array([1.0, 0.86, 0.5], np.float32)
        tb = t - self.T_BURST
        if tb > -0.05:                                                              # 生日快乐: gold light runs out over the rings
            rp = tb / 1.3; k = np.exp(-((self.r - rp) / 0.09) ** 2) * 0.62 * (1 - ramp(tb, 0.0, 1.9))
            glow = 0.10 * math.exp(-max(tb, 0) / 2.5) * ramp(tb, -0.05, 0.25)
            img = img + (1 - img) * (k + glow * np.exp(-self.r ** 2 / 0.5))[..., None] * np.array([1.0, 0.84, 0.46], np.float32)
        # a slow push on the whole card (2x supersampled)
        z = 1.0 + 0.028 * ramp(t, 101.0, 110.0)
        if z > 1.0005:
            up = cv2.resize(img, (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR)
            M_ = np.array([[z, 0, W * (1 - z)], [0, z, H * (1 - z)]], np.float32)
            img = cv2.resize(cv2.warpAffine(up, M_, (2 * W, 2 * H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_AREA)
        img = np.clip(img, 0, 1).astype(np.float32)
        self._fall(img, self.back, t, (W, H + 240))                                   # leaves behind the name
        for B in self.burst:                                                        # leaves lifting off the rim
            tt = tb - B['delay']
            if tt <= 0: continue
            e = 1 - math.exp(-tt / 0.55)
            x = B['x0'] + B['dx'] * e + 20 * math.sin(B['ph'] + 2 * tt); y = B['y0'] + B['dy'] * e + 0.5 * 160 * max(tt - 0.6, 0) ** 2
            a = ramp(tt, 0.0, 0.2) * (1 - ramp(tt, 3.0, 4.0))
            self._leaf(img, B, x, y, B['rot0'] + B['spin'] * tt, B['ph'] + B['fl'] * tt, a, B['size'])
        img = self.dust.draw(img, t - 100.6, 0.6 + 0.7 * ramp(t, 105.8, 106.6))
        if tb > 0:                                                                  # glints flash as the wave passes
            lay = np.zeros((H, W), np.float32); rp = tb / 1.3
            for rr, aa, sz in self.glint:
                f = math.exp(-((rr - rp) / 0.05) ** 2)
                if f < 0.03: continue
                c = self.c0 + (self.c1 - self.c0) * min(rr, 1.0)
                x = c[0] + self.rx * rr * math.cos(aa); y = c[1] + self.ry * rr * math.sin(aa)
                cv2.circle(lay, (int(x * 16), int(y * 16)), int(sz * 16), f, -1, cv2.LINE_AA, shift=4)
                cv2.line(lay, (int((x - 4 * sz) * 16), int(y * 16)), (int((x + 4 * sz) * 16), int(y * 16)), 0.5 * f, 1, cv2.LINE_AA, shift=4)
                cv2.line(lay, (int(x * 16), int((y - 4 * sz) * 16)), (int(x * 16), int((y + 4 * sz) * 16)), 0.5 * f, 1, cv2.LINE_AA, shift=4)
            lay = cv2.GaussianBlur(lay, (0, 0), 1.2) * (1 - ramp(tb, 1.2, 1.9))
            img = np.clip(img + (1 - img) * lay[..., None] * np.array([1.0, 0.9, 0.6], np.float32), 0, 1)
        # the name, printed with the voice; the years; the date between its sprigs on 生日快乐
        cx, cy = 960, 455
        a = ramp(t, 103.95, 104.3)
        if a > 0:
            sc = 1.03 - 0.03 * out_cubic(a)
            C.over(img, self.lock, cx - self.lock.shape[1] * sc / 2, cy - self.lock.shape[0] * sc / 2, a, 0.0, sc)
        a = ease(ramp(t, 104.9, 105.5))
        if a > 0:
            yh = cy + 172; C.over(img, self.years, cx - self.years.shape[1] / 2, yh + 10 * (1 - a), a)
            ln = int(170 * ease(ramp(t, 105.0, 105.7))); yl = int(yh + self.years.shape[0] * 0.5)
            if ln > 2:
                lay = np.zeros((H, W), np.float32)
                for x0 in (cx - self.years.shape[1] // 2 - 26, cx + self.years.shape[1] // 2 + 26):
                    d_ = -1 if x0 < cx else 1
                    cv2.line(lay, (x0 * 16, yl * 16), ((x0 + d_ * ln) * 16, yl * 16), 1.0, 2, cv2.LINE_AA, shift=4)
                img[:] = img * (1 - 0.55 * lay[..., None] * a) + np.array([0.62, 0.48, 0.24], np.float32) * 0.55 * lay[..., None] * a
        a = ease(ramp(t, self.T_BURST, self.T_BURST + 0.5))
        if a > 0:
            yd = cy + 250; C.over(img, self.date, cx - self.date.shape[1] / 2, yd + 14 * (1 - a), a)
            g = ease(ramp(t, self.T_BURST + 0.15, self.T_BURST + 0.8))
            if g > 0:
                ys_ = yd + self.date.shape[0] * 0.5 - self.sprig_l.shape[0] / 2
                C.over(img, self.sprig_l, cx - self.date.shape[1] / 2 - 18 - self.sprig_l.shape[1] * (0.6 + 0.4 * g), ys_, g, 0.0, 0.6 + 0.4 * g)
                C.over(img, self.sprig_r, cx + self.date.shape[1] / 2 + 18 - self.sprig_r.shape[1] * (1 - (0.6 + 0.4 * g)) * 0, ys_, g, 0.0, 0.6 + 0.4 * g)
        # the branches at the corners, swaying (more after the gust of 生日快乐), then the out-of-focus leaves in front
        for Bq in self.br:
            gust = 2.2 * math.exp(-max(tb, 0) / 1.2) * ramp(tb, 0.0, 0.3)
            ang = 1.1 * math.sin(0.8 * (t - 100) + Bq['ph']) + 0.4 * math.sin(2.1 * (t - 100) + 2 * Bq['ph']) + gust * math.sin(3.2 * max(tb, 0))
            b = Bq['img']; hh, ww = b.shape[:2]; px, py = Bq['piv']
            M_ = cv2.getRotationMatrix2D((px - Bq['pos'][0], py - Bq['pos'][1]), ang, 1.0)
            rot = cv2.warpAffine(b, M_, (ww, hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            C.over(img, rot, Bq['pos'][0], Bq['pos'][1], ease(ramp(t, 100.9, 101.9)))
        self._fall(img, self.front, t, (W, H + 900))
        return np.clip(img, 0, 1).astype(np.float32)


class Motto18(E.Motto):
    """the real calligraphy of the motto stone carved in, right to left, as the voice says 求真敬业. The characters are carved into
    the print at fixed places first and the print then moves as one picture, so they cannot drift a pixel against the stone"""
    def __init__(self, t_carve):
        super().__init__(); self.tc = t_carve
        self.st = Parallax18(kf('K_motto_stone'), move=(0, 0, 0.06), dur=7.0)        # constant-speed push (no stop at the joins)
        self.base = self.st.img
    def frame(self, t):
        img = self.base.copy(); done = 0
        for k, (x, y, m) in enumerate(self.g):
            u = ease(ramp(t, self.tc + 0.25 * k, self.tc + 0.3 + 0.25 * k))
            if u <= 0: continue
            done += 1
            X, Y = int(round(x)), int(round(y))
            mm = np.clip(m * 2.4, 0, 1); mm = cv2.dilate(mm, np.ones((2, 2), np.uint8)); hh, ww = mm.shape
            rim = np.clip(cv2.dilate(mm, np.ones((5, 5), np.uint8)) - mm, 0, 1)
            sh = np.roll(np.roll(cv2.GaussianBlur(mm, (0, 0), 1.5), 3, 0), 3, 1)
            reg = img[Y:Y + hh, X:X + ww]
            reg[:] = reg * (1 - 0.6 * np.maximum(sh, rim)[..., None] * u)
            reg[:] = reg * (1 - mm[..., None] * u) + np.array([0.86, 0.66, 0.28], np.float32) * mm[..., None] * u
        return self.st.frame(t - 73.0, img=img if done else None)


class InkCaption18(E17.InkCaption):
    """the film's caption style; a phrase rises as one gesture (its characters 30 ms apart), fades out in 0.25 s"""
    def draw(self, base, t):
        if self.t_out is not None and t > self.t_out + 0.25: return
        if self.t_out is not None and t > self.t_out:
            k = 1 - ease(ramp(t, self.t_out, self.t_out + 0.25)); tmp = base.copy(); t_ = self.t_out; self.t_out = None
            super().draw(tmp, t); self.t_out = t_; base[:] = base * (1 - k) + tmp * k; return
        super().draw(base, t)


def subtitles18():
    """one caption per line of the narration text; each of its phrases appears whole 0.08 s before the voice starts it; a line stays
    until 0.5 s after its last phrase ends, and has faded before the next line rises"""
    caps = []
    for u in sorted(E17.LINES, key=lambda u: CHUNKS[u][0][1]):
        rows = list(CHUNKS[u]); j = 0
        for line in E17.LINES[u]:
            tt, shown_, ends, inside, k_in, t_b = [], '', [], False, 0, 0.0
            for c in line:
                if c in '，、':
                    shown_ += ' '; tt.append(tt[-1] if tt else None); inside = False; continue
                if not inside:
                    ph, a, b, _ = rows[j]; j += 1; ends.append(b); t_b = a - 0.08; k_in = 0; inside = True
                shown_ += c; tt.append(t_b + 0.03 * k_in); k_in += 1
            caps.append([shown_, tt, max(ends)])
    out = []
    for k, (shown_, tt, end) in enumerate(caps):
        nxt = caps[k + 1][1][0] if k + 1 < len(caps) else END
        t_out = max(min(end + 0.5, nxt - 0.3), max(tt) + 0.6)
        out.append(InkCaption18(shown_, tt, W / 2, H - 118, px=40, wght=500, color=E.E.INKC, t_out=t_out, shadow=False))
    return out


class NamesOverlay18(E17.NamesOverlay):
    """the three names in movable type on the vertical slip (no seals): each pressed as the voice says it"""
    def __init__(self):
        super().__init__()
        n = self.nm
        t1 = char_t('u04', '曾经，它叫天目林学院', 5); t2 = char_t('u05', '后来，它叫浙江林学院', 5); t3 = char_t('u06', '今天，它是浙江农林大学', 5)
        n.names = [('天目林学院', t1, t2 - 0.4), ('浙江林学院', t2, t3 - 0.4), ('浙江农林大学', t3, 99)]
        n.seals = []
    def draw(self, img, t):
        if t > 27.75: return
        # the slip fades with the gate, as the ink bloom carries us to the land of 2000 (27.0 - 27.7)
        k = 1 - ease(ramp(t, 27.0, 27.7))
        if k >= 1: return super().draw(img, t)
        base = img.copy(); super().draw(img, t); img[:] = base * (1 - k) + img * k


def film():
    t0, t1, S14, labels14, texts14, trans14 = E17._film16()
    pl, ti = Planet18(), Title19()
    rs = ResearchSamara18(ROOT / 'work/v16k/K_research_samara_plate_1080.png')
    beat = lambda k: 48.3 + k * (59.6 - 48.3) / 16
    # the era long take, part 1: 天目林学院 -> its new building -> 浙江林学院 (the 天目林学院 gate held long enough to read,
    # the run through the pines quick, the 浙江林学院 gate held)
    chain1 = Chain18([
        ('S1', 16.6, 18.8, 'K_S1_first', 'K_S1_last', (0.3, 6.0), dict(knots=[(0.0, 0.3), (0.95, 1.25), (1.25, 2.0), (1.5, 3.4), (1.95, 4.25), (2.2, 6.0)])),
        ('S2', 18.8, 19.8, 'K_S1_last', 'K_S2_last', (0.0, 6.0)),
        ('S3', 19.8, 22.2, 'K_S2_last', 'K_S3_last', (0.0, 6.0), dict(p=0.8, floor=1.0, smax=6.0, smin=0.7)),
        ('S4', 22.2, 24.0, 'K_S3_last', 'K_S4_last', (0.0, 1.3)),
    ], xfades={})
    for a_, b_, sh_ in chain1.segs:
        if sh_.path.stem == 'S2': sh_.zoom = 1.06
        if sh_.path.stem == 'S1': sh_.zoom_ramp = (18.1, 18.8, 1.0, 1.06)        # S2's crop, reached inside the run past the pines
    # part 2: the East Lake grows up (2000 -> 2001 -> 2002 -> the library 2004 -> today)
    # the building site of 2000-2001 is printed over, in one move, as the library
    chain2 = Chain18([
        ('E2', 27.4, 31.4, 'K_E_P2_2000', 'K_E_P14_2001', (0.3, 6.9)),
        ('S7', 31.4, 37.8, 'K_S6_last', 'K_S7_last', (0.0, 7.9)),
    ], xfades={31.4: ('reprint', 1.4)})
    t1s = AIShotMap('T1b', 37.8, 42.6, 'K_T1_first_AIG_last', 'K_T1_last_nobuild', src=(0.3, 9.6), p=0.0)   # linear: the three.js leaves fit it
    gate, bam, pet, fo = Gate(24.0, 27.4), Bamboo(42.6, 48.3), Petrified(59.6, 62.6), Forest(69.4, 73.0)
    carve = char_t('u14', '在这里，我们求真敬业', 6) - 0.12
    lin = dict(p=0.0)
    ai4 = AIShotMap('AI-4', 48.3, beat(6), 'K_ai4_library_high_last', src=(0.0, 6.0), p=0.0)
    shots = [
        (0.0, 5.6, pl.frame),
        # the opening: AI-1 follows the planet on a cut, already moving; a bank of mist leads into O1; the seed lands,
        # and sprouts on 萌芽, its leaves open on 生长
        (5.6, 9.9, AIShotMap('AI-1', 5.6, 9.9, 'T1_tianmu_ink', src=(0.0, 7.4),
                             knots=[(0.0, 0.0), (0.5, 0.78), (1.4, 2.0), (2.1, 3.35), (2.7, 5.0), (4.3, 7.4)]).frame),   # ~1.5x out of the planet
        (9.9, 12.9, AIShotMap('O1', 9.9, 12.9, 'K_O1_first_AI1_last', src=(2.55, 8.0), p=1.0, floor=1.0, smax=6.0, smin=0.45).frame),
        (12.9, 16.6, AIShotMap('AI-G', 12.9, 16.6, 'K_aiG_samara_soil_first', 'K_aiG_seedling_last', src=(0.0, 6.0),
                               knots=[(0.0, 0.0), (0.4, 2.0), (1.4, 4.0), (2.2, 5.0), (3.7, 6.0)]).frame),
        (16.6, 24.0, chain1.frame),
        (24.0, 27.4, gate.frame),
        (27.4, 37.8, chain2.frame),
        (37.8, 42.6, t1s.frame),
        (42.6, 48.3, bam.frame),
        (48.3, beat(6), ai4.frame),
        (beat(6), 55.95, Plants18(ai4.frame, beat(6), [beat(7), beat(8), beat(9), beat(10)]).frame),   # the campus becomes the first card
        (55.95, 56.85, (lambda P_: (lambda t: P_.frame(t - 55.95)))(Parallax18(kf('K_tianmu_colour'), move=(-40, 0, 0.05), dur=0.9))),
        (56.85, 59.6, (lambda P_: (lambda t: P_.frame(t - 56.85)))(Parallax18(kf('K_island_summer'), move=(0, 0, 0.06), dur=2.75))),
        (59.6, 62.6, pet.frame),
        (62.6, 66.8, AIShotMap('AI-5', 62.6, 66.8, 'K_ai5_library_mist_first', src=(2.6, 7.6), **lin).frame),
        (66.8, 69.4, rs.frame),
        (69.4, 73.0, fo.frame),
        (73.0, 80.0, Motto18(carve).frame),
        (80.0, 87.8, AIShotMap('AI-7', 80.0, 87.8, 'K_ai7_ginkgo_first', 'K_ai7_campus_mountains_last', src=(0.0, 7.8), **lin).frame),
        (87.8, 90.8, AIShotMap('AI-8', 87.8, 90.8, 'T1_swans', src=(0.0, 3.0), **lin).frame),
        (90.8, 93.8, AIShotMap('AI-9', 90.8, 93.8, 'K_ai9_luming_snow_first', 'K_ai9_luming_autumn_last', src=(0.0, 6.0), **lin).frame),
        (93.8, 101.2, AIShotMap('AI-10', 93.8, 101.2, 'K_ai10_campus_aerial_first', 'K_ai10_clouds_ink_last', src=(0.0, 7.4), **lin).frame),
        (101.2, END, ti.frame),
    ]
    trans_at = {
        9.9: ('mist', 1.4),                                # down through the mist: the ink valley -> the valley taking colour
        16.6: ('particles3d', 1.6),                        # the seedling lifts off as pigment grains, they gather, the first old gate bleeds out
        24.0: ('reprint', 1.4),                            # 今天: the 浙江林学院 gate washes off the sheet, today's gate is printed in its place
        27.4: ('ink', 1.1, dict(cx=0.52, cy=0.40)),        # ink blooms from the end of the avenue: the same mountains, the land in 2000
        37.8: ('particles3d', 1.6, dict(move=(5.0, 2.4))), # today's lakeside woods turn to particles, settle as one seedling
        42.6: ('light', 1.2),                              # the great maple's golden light -> the white mist the bamboo bursts from
        48.3: ('ink', 0.8),                                # the bamboo sky -> the library from above
        55.95: ('light', 1.1, dict(warm=(0.92, 0.95, 1.0), peak=0.6)),   # snow light: the snow glows, the cloud sea comes out of it
        56.85: ('ink', 0.9, dict(cx=0.70, cy=0.50)),       # 一山 -> 一水: water spreads from the island
        59.6: ('mist', 1.3),                               # the lake mist -> the petrified wood (古)
        62.6: ('lumakey', 1.2),                            # the ancient trunks -> the library in the morning mist (今), the mist first
        66.8: ('reprint', 1.5),                            # the library washes off the sheet, the study plate is printed
        69.4: ('ink', 1.1, dict(cx=0.12, cy=0.58)),        # ink spreads from the seed on the plate: a forest
        73.0: ('light', 1.2),                              # the forest's light swells, the motto stone comes out of it
        80.0: ('lumakey', 1.4),                            # the gold of the ginkgo avenue spreads in through the light places
        87.8: ('ink', 0.9, dict(cx=0.6, cy=0.6)),
        90.8: ('mist', 1.3), 93.8: ('luma', 1.4),        # the swans' lake mist -> the lake in snow; aerial to aerial: a long dissolve
        101.2: ('ink', 1.2),
    }
    ends = {round(sh[1], 3): j for j, sh in enumerate(shots)}
    trans = {ends[round(t, 3)]: v for t, v in trans_at.items()}
    names = NamesOverlay18()
    c1 = '本片部分画面由人工智能辅助生成'                       # the credit line at the end
    credit = [C.RiseText(c1, [107.4] * len(c1), W / 2, H - 70, px=22, wght=400, color=(80, 66, 52, 230), shadow=False)]
    keep = [T for T in texts14 if T.y != H - 118 and T.px < 100 and not (T.vertical and T.t_out is not None and T.t_out > 60)]
    keep += [C.RiseText('博览古今', phrase_times('u12', '博览古今'), 1660, 40, px=66, wght=600, color=(40, 34, 28, 235), vertical=True, t_out=65.9, shadow=False, gap=0.12),
             C.RiseText('学参天地', phrase_times('u12', '学参天地'), 1560, 40, px=66, wght=600, color=(40, 34, 28, 235), vertical=True, t_out=65.9, shadow=False, gap=0.12)]
    return 0.0, END, shots, [], keep + (subtitles18() if CHUNKS else E17.subtitles()) + credit, trans


def render(t_a=0.0, t_b=END, out=None):
    E.film = film
    E.render(t_a, t_b, out or ROOT / 'renders/v18/R13_preview.mov')


if __name__ == '__main__':
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; b = float(sys.argv[2]) if len(sys.argv) > 2 else END
    o = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    render(a, b, o)
