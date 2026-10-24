"""Shot list with the era long take S1-S7 from the real old gates to today's East Lake, the seedling growing into the great
maple (T1), every join designed. Same renderer as edit16 (scenes, captions, seals); the final cut is assembled in DaVinci from
the elements export17 writes. An AI shot whose clip is missing stands in as its keyframes (2.5D moves).
  .venv_ai/bin/python work/v16k/edit17.py [t0 t1 [out.mov]]   (a quick Python preview of the same cut)"""
import sys, math
from pathlib import Path
import numpy as np, cv2
import importlib.util as ilu
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
s = ilu.spec_from_file_location('edit16', ROOT / 'work/v16k/edit16.py'); E = ilu.module_from_spec(s); sys.modules['edit16'] = E; s.loader.exec_module(E)
C, P, S16, PL = E.C, E.P, E.S16, E.PL
W, H, FPS = 1920, 1080, 25
ease, ramp, out_cubic = C.ease, C.ramp, C.out_cubic
OUT, REFS = E.OUT, E.REFS
kf = E.kf
_film16 = E.film                                             # captions, seals and labels come from the v14/v16 timing


class Chain:
    """the era long take: AI segments back to back (first frame of each = last frame of the one before); each mapped onto its slot
    with its own speed; until a segment exists, its two keyframes cross over as a slow push (stand-in). xfades: {time: seconds}
    for joins whose two sides are not the same frame (they dissolve; in DaVinci a cross dissolve)"""
    def __init__(self, segs, xfades=None):
        self.segs = []; self.xf = xfades or {}
        for name, a, b, first, last, src in segs:
            self.segs.append((a, b, E.AIShot(name, a, b, first, last, src=src, curve='lin', move=(0, 0, 0.06))))
    def _at(self, t):
        for a, b, sh in self.segs:
            if t < b or sh is self.segs[-1][2]: return sh.frame(min(max(t, a), b))
    def frame(self, t):
        for at, spec in self.xf.items():
            kind, d, kw = (('luma', spec, {}) if not isinstance(spec, tuple) else (spec[0], spec[1], spec[2] if len(spec) > 2 else {}))
            if at - d / 2 <= t < at + d / 2:
                i = [k for k, (a, b, sh) in enumerate(self.segs) if abs(b - at) < 1e-6][0]
                A = self.segs[i][2].frame(t); B = self.segs[i + 1][2].frame(t)
                return C.TRANS[kind](A, B, (t - (at - d / 2)) / d, **kw)
        return self._at(t)


class RootsFrom:
    """30.6-37.8: the S7 picture (today's East Lake) sinks: the camera goes down through the ground into the roots on paper (v14 Roots),
    the years count to 2026, the gold line rises (the strength to grow upward)"""
    def __init__(self, top):
        self.top = top; self.R = C.Roots()
    def frame(self, t):
        # the whole picture is pushed up and out (a tilt down into the earth); the roots page arrives with its own ground line
        u = ease(ramp(t, 30.6, 31.9)); off = int(round(1080 * u))
        rf = self.R.frame(min(t, 37.7))
        if off <= 0: return self.top(t)
        top = self.top(30.6)
        out = np.empty((H, W, 3), np.float32)
        out[:H - off] = top[off:]
        out[H - off:] = rf[:off]
        return out


def T_pigment(A, B, u, seed=7, wind=(26, 34), wave='bottom', grain=2, gather=False):
    """pigment-grain transition: A breaks into grains (one per grain x grain px, its own colour) that drift with `wind` and settle into
    B, a wave travelling from `wave` ('bottom'|'left'|'right'|'centre'); grains are paired inside 32 px tiles by brightness so the frame
    keeps its form. gather=True: the grains of B come in out of A (B assembles from scattered pigment instead of A dissolving)"""
    h, w = A.shape[:2]
    key = (h, w, seed, grain)
    if key not in _PG:
        r = np.random.default_rng(seed)
        ys, xs = np.mgrid[0:h:grain, 0:w:grain]; ys = ys.ravel(); xs = xs.ravel()
        _PG[key] = (ys, xs, r.random(ys.size).astype(np.float32), r.normal(0, 1, (ys.size, 2)).astype(np.float32))
    ys, xs, jit, sw = _PG[key]
    pos = {'bottom': 1 - ys / h, 'left': xs / w, 'right': 1 - xs / w,
           'centre': np.hypot(xs / w - 0.5, (ys / h - 0.5) * h / w) * 1.6}[wave]
    wave_t = 0.05 + 0.55 * pos + 0.12 * jit
    tau = np.clip((u - wave_t) / 0.35, 0, 1); e = tau * tau * (3 - 2 * tau); bell = np.sin(np.pi * tau)
    cA = A[ys, xs]; cB = B[ys, xs]
    if gather:                                   # B's grains drift in out of the dark and gather into knots; B appears there, then bleeds outward
        import particles3d as P3
        if ('g', key) not in _PG:
            pts = P3.centres(B, 3, 300); cid = np.argmin(np.stack([np.hypot(xs - x_, ys - y_) for x_, y_ in pts]), 0)
            r2 = np.random.default_rng(seed + 1); rr = np.sqrt(r2.random(xs.size)) * 34; th = r2.random(xs.size) * 6.283
            _PG[('g', key)] = (pts[cid, 0] + rr * np.cos(th), pts[cid, 1] + rr * np.sin(th) * 0.8, P3.Bleed(pts))
        kx, ky, bl = _PG[('g', key)]
        ug = np.clip(u / 0.5, 0, 1); eg = ug * ug * (3 - 2 * ug)
        x = xs + (kx - xs) * eg + (1 - eg) * 60 * sw[:, 0]; y = ys + (ky - ys) * eg + (1 - eg) * 60 * sw[:, 1]
        col = cB; alive = np.full(xs.size, min(1.0, u / 0.15), np.float32)
    else:
        x = xs + bell * (wind[0] + 14 * sw[:, 0]); y = ys + bell * (wind[1] + 14 * sw[:, 1])
        col = cA * (1 - e[:, None]) + cB * e[:, None]; alive = np.ones_like(tau)
    acc = np.zeros((h * w, 3), np.float32); cov = np.zeros(h * w, np.float32)
    xi = np.clip(x.astype(int), 0, w - 1); yi = np.clip(y.astype(int), 0, h - 1)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (alive > 0)
    for dx in range(grain):
        for dy in range(grain):
            idx = (np.clip(yi + dy, 0, h - 1) * w + np.clip(xi + dx, 0, w - 1))[ok]
            cov += np.bincount(idx, minlength=h * w).astype(np.float32)
            for c in range(3): acc[:, c] += np.bincount(idx, col[ok, c], h * w)
    acc = acc / np.maximum(cov, 1e-6)[:, None]; cov = np.clip(cov / (1.1 * grain * grain), 0, 1)
    grains = acc.reshape(h, w, 3); o = cov.reshape(h, w, 1)
    lt = np.zeros(h * w, np.float32); lt[(ys * w + xs)] = np.clip((u - wave_t) / 0.35, 0, 1)
    left = cv2.resize(lt.reshape(h, w)[::grain, ::grain], (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    if gather:
        fr_ = np.clip(A * (1 - o * alive.max()) + grains * o * alive.max(), 0, 1)
        ub = np.clip((u - 0.5) / 0.5, 0, 1)
        if ub > 0:
            al, rim = bl.mask(ub)
            fr_ = fr_ * (1 - al[..., None]) + (B * (1 - 0.18 * rim[..., None])) * al[..., None]
        return fr_
    base = A * (1 - left) + B * left
    fly = (left > 0.001) & (left < 0.999)
    return np.clip(np.where(fly, base * (1 - 0.85 * o) + grains * 0.85 * o, base), 0, 1)


_PG = {}
C.TRANS['pigment'] = T_pigment
C.TRANS['particles3d'] = lambda A, B, u, move=None: T_pigment(A, B, u, wave='centre')


PUNCT = '，。、,.！？ '
LINES = {   # the narration text, one caption line per entry (u01 is the vertical question, u12 the vertical 博览古今 学参天地, u18 the title)
    'u02': ['1958年，它落在了钱王故里、天目山下'], 'u03': ['于是，它开始萌芽、生长'],
    'u04': ['曾经，它叫天目林学院'], 'u05': ['后来，它叫浙江林学院'], 'u06': ['今天，它是浙江农林大学'],
    'u07': ['在这片肥沃的大地上', '它扎根近70年', '只为积蓄向上生长的力量'],
    'u08': ['像一棵树，根深叶茂，硕果满枝'], 'u09': ['像一株竹，坚韧不拔，不断超越'], 'u10': ['整个校园就是一座植物园'],
    'u11': ['在这里，一草一木，皆有学问', '一山一水，皆是课堂'],
    'u13': ['在这里，我们研究一粒种子的生长', '我们也守护一座森林的未来'],
    'u14': ['在这里，我们求真敬业', '我们肯干、实干、能干'],
    'u15': ['这是一所大学', '坐落在国家森林公园里的大学'], 'u16': ['欢迎你来这里扎根', '欢迎你在这里生长'],
    'u17': ['他会成全你的梦想', '他会托举你的未来'],
}


class InkCaption(C.RiseText):
    """the one caption style of the film: ink characters on a soft glow of paper (readable on dark and light pictures)"""
    def draw(self, base, t):
        fade = 1.0 if self.t_out is None else 1 - ease(ramp(t, self.t_out, self.t_out + 0.3))
        if fade <= 0: return
        paper = np.array([0.95, 0.925, 0.86], np.float32)
        for g, tc, pos in self.items:
            u = out_cubic(ramp(t, tc, tc + 0.42))
            if u <= 0: continue
            h, w = g.shape[:2]; rise = (1 - u) * self.px * 0.55
            gx, gy = (self.x, self.y + pos) if self.vertical else (self.x + pos, self.y)
            vis = g.copy(); vis[max(0, h - int(rise) - 1):, :, 3] *= 0 if u < 1 else 1
            pd = int(self.px * 0.7)
            a = np.pad(vis[..., 3], pd)
            glow = np.clip(cv2.GaussianBlur(cv2.dilate(a, np.ones((7, 7), np.uint8)), (0, 0), self.px * 0.3) * 1.6, 0, 0.62)
            C.over(base, np.dstack([np.broadcast_to(paper, a.shape + (3,)), glow]).astype(np.float32), gx - pd, gy - rise - pd, fade * u)
            C.over(base, vis, gx, gy - rise, fade * u)


def subtitles():
    """every caption line timed from the voice: the characters of the official line mapped by position onto whisper's characters of
    the same utterance (whisper's homophones do not matter), each line leaves 0.35 s before the next one arrives"""
    WT = C.WT; out = []
    order = sorted(LINES, key=lambda u: WT[u]['film_start'])
    for u in order:
        seq = []
        for w, a, b in WT[u]['words']:
            cs = [c for c in w if c not in PUNCT]
            for k, c in enumerate(cs): seq.append(a + (b - a) * k / max(len(cs), 1))
        allc = ''.join(c for line in LINES[u] for c in line if c not in PUNCT)
        scale = (len(seq) - 1) / max(len(allc) - 1, 1)
        i = 0
        for line in LINES[u]:
            ts = []
            for c in line:
                if c in PUNCT: ts.append(None); continue
                ts.append(seq[min(len(seq) - 1, int(round(i * scale)))]); i += 1
            for k in range(len(ts)):
                if ts[k] is None: ts[k] = ts[k - 1] if k else next(x for x in ts if x is not None)
            shown = ''.join(' ' if c in '，。、' else c for c in line)
            out.append([shown, ts, WT[u]['words'][-1][2]])
    # one line at a time: a line stays until its last character has risen (+0.45 s) and fades out (0.3 s) before the next one rises;
    # when the voice runs straight on, the next line waits (it shifts as a whole, at most about half a second)
    caps = []; free = 0.0
    for k, (shown, ts, u_end) in enumerate(out):
        d = max(0.0, free - ts[0]); ts = [x + d for x in ts]
        hold = max(ts) + 0.45
        nxt = out[k + 1][1][0] if k + 1 < len(out) else 108.0
        t_out = max(hold, min(nxt - 0.35, max(ts) + 1.6, (u_end + 1.2) if (k + 1 < len(out) and out[k + 1][2] != u_end) else 1e9))
        caps.append(InkCaption(shown, ts, W / 2, H - 118, px=40, wght=500, color=E.E.INKC, t_out=t_out, shadow=False))
        free = t_out + 0.32
    return caps


def film():
    t0, t1, S14, labels14, texts14, trans14 = _film16()
    seals14 = [L for L in labels14 if isinstance(L, E.E.Seal)]
    beat = lambda k: 48.0 + k * 0.725
    pl, ti = PL.Planet(), E.E.Title()
    rs = E.ResearchSamara(ROOT / 'work/v16k/K_research_samara_plate_1080.png')
    # the era long take: (segment, slot start, slot end, first keyframe, last keyframe, source window in s)
    chain = Chain([
        ('S1', 16.6, 18.4, 'K_S1_first', 'K_S1_last', (0.3, 6.0)),       # 天目林学院 stone gate, through the pines -> 林化大楼 in scaffolding
        ('S2', 18.4, 20.2, 'K_S1_last', 'K_S2_last', (0.0, 6.0)),        # walls go up, the scaffolding comes down (zoomed 6 %: a car roof at the edge)
        ('S3', 20.2, 22.0, 'K_S2_last', 'K_S3_last', (0.0, 6.0)),        # the trees grow, ochre prints in: the early-80s 浙江林学院 gate
        ('S4', 22.0, 22.9, 'K_S3_last', 'K_S4_last', (0.0, 2.1)),        # into the dark of the gateway (the rest of S4 is not used)
        ('S5', 22.9, 23.8, 'K_S4_last', 'K_S4_last', (0.0, 1.1)),        # out of the dark: the iron gate (S5's first second only)
        ('S5', 23.8, 25.2, 'K_S5_last', 'K_S5_last', (5.1, 6.0)),        # today's 衣锦 gate (the lintel lettering S5 garbled in between is skipped)
        ('S6', 25.2, 27.4, 'K_S5_last', 'K_S6_last', (0.0, 3.2)),        # up past the plane trees, over the sea of cloud
        ('E2', 27.4, 30.6, 'K_E_P2_2000', 'K_E_P14_2001', (0.3, 6.6)),   # 2000 farmland -> 2001 site: lake dug, cranes
        ('E3a', 30.6, 31.4, 'K_E_P14_2001', 'K_E_P11_2002', (0.0, 0.7)), # the cranes turn ...
        ('E3a', 31.4, 32.4, 'K_E_P14_2001', 'K_E_P11_2002', (4.2, 6.0)), # ... the building has risen (E3a's sliding middle is skipped)
        ('S7', 32.4, 37.8, 'K_S6_last', 'K_S7_last', (0.0, 7.9)),        # another place: the library just finished (2004), its bare shore grows into today's woods
    ], xfades={18.4: 0.4, 22.9: ('pigment', 1.4, dict(gather=True, wind=(0, -20), wave='centre')),    # out of the gateway's dark the iron gate gathers from pigment
               23.8: ('pigment', 1.2, dict(wind=(160, -40), wave='left', grain=4)),        # the iron gate blows away as leaves, today's gate behind
               27.4: 0.8,
               31.4: ('pigment', 0.8, dict(gather=True, wind=(0, -10), wave='centre')),
               32.4: ('pigment', 0.9, dict(gather=True, wind=(0, -10), wave='centre'))})     # the 2002 site breaks up; the 2004 library gathers and bleeds out (a different building, an honest change of place)     # 2001 -> 2002: pigment knots, then the ink bleed
    for a_, b_, sh_ in chain.segs:
        if sh_.path.stem == 'S2': sh_.zoom = 1.06
    t1s = E.AIShot('T1b', 37.8, 42.6, 'K_T1_first_AIG_last', 'K_T1_last_nobuild', src=(0.3, 9.6))
    bamboo = S16.Parallax(kf('K_bamboo_up'), move=(0, -60, 0.10), dur=5.4)
    shots = [
        (0.0, 5.6, pl.frame),
        (5.6, 8.6, E.AIShot('AI-1', 5.6, 8.6, 'T1_tianmu_ink', src=(0.0, 8.0)).frame),                  # the ink ridge dive (fast)
        (8.6, 13.0, E.AIShot('O1', 8.6, 13.0, 'K_O1_first_AI1_last', 'K_O1_last_AIG_first', src=(0.0, 8.0)).frame),   # through the paper clouds, the samara lands
        (13.0, 16.6, E.AIShot('AI-G', 13.0, 16.6, 'K_aiG_samara_soil_first', 'K_aiG_seedling_last', src=(0.0, 8.2)).frame),   # leaves open by 15 s, then it turns to particles
        (16.6, 37.8, chain.frame),
        (37.8, 42.6, t1s.frame),
        (42.6, 48.0, lambda t: bamboo.frame(t - 42.6)),
        (48.0, beat(6), E.AIShot('AI-4', 48.0, beat(6), 'K_ai4_library_high_last', src=(0.0, 6.0)).frame),
        (beat(6), 56.0, E.Plants([beat(6), beat(7), beat(8), beat(9), beat(10)]).frame),
        (56.0, 57.4, (lambda P_: (lambda t: P_.frame(t - 56.0)))(S16.Parallax(kf('K_tianmu_colour'), move=(-40, 0, 0.05), dur=1.4))),
        (57.4, 59.6, (lambda P_: (lambda t: P_.frame(t - 57.4)))(S16.Parallax(kf('K_island_summer'), move=(0, 0, 0.06), dur=2.2))),
        (59.6, 66.8, E.AIShot('AI-5', 59.6, 66.8, 'K_ai5_library_mist_first', src=(0.0, 7.2)).frame),
        (66.8, 69.4, rs.frame),
        (69.4, 73.0, E.AIShot('AI-6', 69.4, 73.0, 'K_ai6_path_last', src=(2.4, 6.0)).frame),
        (73.0, 80.0, E.Motto().frame),
        (80.0, 87.8, E.AIShot('AI-7', 80.0, 87.8, 'K_ai7_ginkgo_first', 'K_ai7_campus_mountains_last', src=(0.0, 7.8)).frame),
        (87.8, 90.8, E.AIShot('AI-8', 87.8, 90.8, 'T1_swans', src=(0.0, 3.0)).frame),
        (90.8, 93.8, E.AIShot('AI-9', 90.8, 93.8, 'K_ai9_luming_snow_first', 'K_ai9_luming_autumn_last', src=(0.0, 6.0)).frame),
        (93.8, 101.2, E.AIShot('AI-10', 93.8, 101.2, 'K_ai10_campus_aerial_first', 'K_ai10_clouds_ink_last', src=(0.0, 7.4)).frame),
        (101.2, 108.0, ti.frame),
    ]
    # every join, keyed by its time, with its motive
    trans_at = {
        16.6: ('particles3d', 1.6, dict(align='end')),                    # the seedling breaks into pigment particles that fly into the first old gate
        37.8: ('particles3d', 1.6, dict(move=(5.0, 2.4))),   # today's lakeside woods turn to particles, settle as one seedling
        42.6: ('luma', 0.9),                           # up the great maple -> the bamboo looking up (a soft dissolve)
        48.0: ('ink', 0.7),                            # the bamboo sky -> the library from above
        beat(6): ('pigment', 1.2),                     # the red-leaf wave breaks into pigment grains -> the first plant print
        56.0: ('luma', 0.6),                           # snow white -> cloud white
        57.4: ('luma', 0.6),                           # 一山 -> 一水 (cloud white into water)
        59.6: ('luma', 0.8),                           # the lake -> the misty library
        66.8: ('luma', 0.9),                           # mist white -> the paper of the study plate
        69.4: ('luma', 0.5),                           # the plate -> leaves blowing down the forest road
        73.0: ('luma', 0.7), 80.0: ('luma', 0.7),      # road -> motto stone -> the ginkgo flight
        87.8: ('ink', 0.9, dict(cx=0.6, cy=0.6)),      # the ginkgo campus -> the lake: ink blooms into water
        90.8: ('luma', 0.8),                           # the swans' water -> the island seen from above
        93.8: ('luma', 1.0),                           # autumn island -> rising into the sky
        101.2: ('ink', 1.2),                           # clouds turn to ink -> paper, the title
    }
    ends = {round(sh[1], 3): j for j, sh in enumerate(shots)}
    trans = {ends[round(t, 3)]: v for t, v in trans_at.items()}
    L = C.Label
    labels = [L('天目山', 7.0, 11.0, 120, 880), L('东湖校区', 48.7, beat(6) - 0.05, 120, 880), L('天目山', 56.1, 57.35, 120, 880),
              L('浙江农林大学', 81.2, 87.6, 120, 880),
              L('东湖校区原址 · 二〇〇〇', 27.6, 29.9, 120, 880), L('建设中 · 二〇〇一', 30.0, 31.5, 120, 880),
              L('建设中 · 二〇〇二', 31.5, 32.3, 120, 880), L('东湖图书馆落成 · 二〇〇四', 32.9, 35.0, 120, 880), L('今天 · 二〇二六', 36.2, 37.7, 120, 880)]
    names = NamesOverlay()
    c1 = '本片部分画面由人工智能辅助生成'; c2 = '老照片来源：浙江农林大学校友总会“母校印记”“忆母校的门”、浙江农林大学官方公众号'
    credit = [C.RiseText(c1, [105.6] * len(c1), W / 2, H - 112, px=22, wght=400, color=(80, 66, 52, 230), shadow=False),
              C.RiseText(c2, [105.75] * len(c2), W / 2, H - 76, px=20, wght=400, color=(80, 66, 52, 230), shadow=False)]
    keep = [T for T in texts14 if T.y != H - 118]                       # the vertical question, '1958', 博览古今 学参天地
    return t0, t1, shots, labels + seals14 + [names], keep + subtitles() + credit, trans


class NamesOverlay:
    """the three names in movable type on the vertical slip, and the three year seals, drawn over the era long take"""
    def __init__(self):
        self.nm = S16.Names.__new__(S16.Names)
        n = self.nm
        n.slip_x, n.slip_y, n.slip_w, n.slip_h = 1640, 60, 150, 820
        n.names = [('天目林学院', 18.44, 21.3), ('浙江林学院', 21.67, 24.9), ('浙江农林大学', 25.6, 30.2)]
        n.px = 112; n.masks = {}
        for name, *_ in n.names:
            for ch in name:
                for r in (1.0, 0.55, 0.0):
                    if (ch, r) not in n.masks: n.masks[(ch, r)] = P.type_mask(ch, n.px, r, seed=ord(ch))
        n.seals = [(P.seal('一九五八', 72), 19.75), (P.seal('一九六六', 72), 22.75), (P.seal('二〇一〇', 72), 26.75)]
        n.paper = S16.paper()
    def draw(self, img, t):
        n = self.nm
        sa = ease(ramp(t, 17.4, 18.2)) * (1 - ease(ramp(t, 29.9, 30.4)))
        if sa <= 0: return
        x0, y0, sw, sh = n.slip_x, n.slip_y, n.slip_w, n.slip_h
        img[y0 + 6:y0 + sh + 6, x0 + 6:x0 + sw + 6] *= 1 - 0.18 * sa
        img[y0:y0 + sh, x0:x0 + sw] = img[y0:y0 + sh, x0:x0 + sw] * (1 - sa) + n.paper[y0:y0 + sh, x0:x0 + sw] * sa
        for k, (name, t_in, t_out) in enumerate(n.names):
            rough = (1.0, 0.55, 0.0)[k]; m_n = len(name); step = (sh - 60) / 6
            for i, ch in enumerate(name):
                ti = t_in + 0.16 * i
                a = ease(ramp(t, ti, ti + 0.12)) * (1 - ease(ramp(t, t_out, t_out + 0.25))) * sa
                if a <= 0: continue
                m = n.masks[(ch, rough)]; sc = 1.25 - 0.25 * out_cubic(ramp(t, ti, ti + 0.12))
                mm = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC); hh, ww = mm.shape
                cx = x0 + sw / 2; cy = y0 + 40 + step * (i + 0.5) + (step * (6 - m_n) / 2)
                X, Y = int(cx - ww / 2), int(cy - hh / 2)
                cov = mm[..., None] * a
                img[Y:Y + hh, X:X + ww] = img[Y:Y + hh, X:X + ww] * (1 - cov) + P.INK * cov
        for k, (s_, ts) in enumerate(n.seals):
            if t < ts: continue
            u = ramp(t, ts, ts + 0.09)
            P.over(img, s_, x0 - 96, y0 + 150 + 110 * k, alpha=u * sa, scale=1.3 - 0.3 * out_cubic(u))


def render(t_a=0.0, t_b=108.0, out=None):
    E.film = film                                            # the edit16 renderer, fed this shot list
    E.render(t_a, t_b, out or ROOT / 'renders/v16/R9_preview.mov')


if __name__ == '__main__':
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; b = float(sys.argv[2]) if len(sys.argv) > 2 else 108.0
    o = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    render(a, b, o)
