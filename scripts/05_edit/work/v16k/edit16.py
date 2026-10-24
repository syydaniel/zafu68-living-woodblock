"""Full film renderer (one living woodblock print): code scenes + generated keyframes (codex/out) + Seedance shots (libtv_jobs/out/AI-*.mp4,
picked up automatically; until a shot exists its keyframe stands in as a 2.5D move). Captions, labels and seals reuse the v14
timing (work/v14/edit14.film), so the picture lands on the same voice and score.
  .venv_ai/bin/python work/v16k/edit16.py [t0 t1 [out.mov]]   -> renders/v16/R8_picture.mov (ProRes 422 HQ 1080p, no audio)"""
import sys, math, os, subprocess
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import importlib.util as ilu

ROOT = Path(__file__).resolve().parents[2]
def _load(name, path):
    s = ilu.spec_from_file_location(name, path); m = ilu.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m
E = _load('edit14', ROOT / 'work/v14/edit14.py')                  # v14 scenes, captions, seals
C = E.C; P = E.P
S16 = _load('scenes16', ROOT / 'work/v16k/scenes.py')
PL = _load('planet16', ROOT / 'work/v16k/planet.py')
W, H, FPS = 1920, 1080, 25
ease, ramp, out_cubic = C.ease, C.ramp, C.out_cubic
OUT = ROOT / 'libtv_jobs/out'; REFS = ROOT / 'libtv_jobs/refs'; CX = ROOT / 'codex/out'


def kf(name):
    for p in (CX / f'{name}.png', REFS / f'{name}.png', REFS / f'{name}_1280x720.png', ROOT / f'work/v16t/T1/{name}_1080.png'):
        if p.exists(): return p
    print(f'  keyframe {name} missing, stand-in used', flush=True)
    return ROOT / 'work/v16t/T1/T1_tianmu_ink_1080.png'


class AIShot:
    """a Seedance shot mapped onto [a, b]: source time follows a speed curve; until the file exists, the keyframe(s) stand in"""
    def __init__(self, name, a, b, first, last=None, src=(0.0, None), curve='lin', move=(0, 0, 0.05)):
        self.a, self.b, self.curve = a, b, curve
        self.path = OUT / f'{name}.mp4'
        self.frames = None
        if self.path.exists():
            cap = cv2.VideoCapture(str(self.path)); fps = cap.get(cv2.CAP_PROP_FPS) or 24; fr = []
            while True:
                ok, f = cap.read()
                if not ok: break
                fr.append(f)
            self.fps = fps; self.raw = fr
            s0, s1 = src; self.s0 = s0; self.s1 = s1 if s1 is not None else (len(fr) - 1) / fps
            self.cache = {}
            print(f'  AI {name}: {len(fr)} frames, using {self.s0:.2f}-{self.s1:.2f} s', flush=True)
        else:
            print(f'  AI {name}: not yet generated, keyframe stand-in', flush=True)
            self.A = S16.Parallax(kf(first), move=move, dur=b - a)
            self.B = S16.Parallax(kf(last), move=move, dur=b - a) if last else None

    def frame(self, t):
        u = min(max((t - self.a) / (self.b - self.a), 0), 1)
        if self.path.exists():
            v = {'lin': u, 'in': u * u, 'ramp': u ** 1.7, 'out': 1 - (1 - u) ** 2}[self.curve]
            st = self.s0 + (self.s1 - self.s0) * v
            k = st * self.fps; i0 = int(math.floor(k)); f = k - i0
            i0 = min(max(i0, 0), len(self.raw) - 1); i1 = min(i0 + 1, len(self.raw) - 1)
            def get(i):
                if i not in self.cache:
                    fr_ = self.raw[i]; z_ = getattr(self, 'zoom', 1.0)
                    if z_ > 1.0:
                        hh_, ww_ = fr_.shape[:2]; ch_, cw_ = int(hh_ / z_), int(ww_ / z_); fr_ = fr_[(hh_ - ch_) // 2:(hh_ - ch_) // 2 + ch_, (ww_ - cw_) // 2:(ww_ - cw_) // 2 + cw_]
                    self.cache[i] = cv2.resize(cv2.cvtColor(fr_, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
                    if len(self.cache) > 6: self.cache.pop(next(iter(self.cache)))
                return self.cache[i]
            return get(i0) * (1 - f) + get(i1) * f if f > 0.01 else get(i0)
        im = self.A.frame(t - self.a)
        if self.B is not None:
            w = ease(ramp(u, 0.35, 0.85)); im = im * (1 - w) + self.B.frame(t - self.a) * w
        return im


class LawnRoots:
    """30.6-37.8: the lawn print (AI-2's last frame), the camera sinks through the ground into the roots on paper (v14 Roots: ochre
    woodblock roots, the years counting), then follows the gold line of 向上生长的力量 back up to the grass"""
    def __init__(self):
        self.lawn = S16.Parallax(kf('T1_lawn_golden'), move=(0, 0, 0.03), dur=7.2)
        self.R = C.Roots()
    def frame(self, t):
        off = 930 * ease(ramp(t, 30.75, 31.95)) * (1 - ease(ramp(t, 36.55, 37.75)))
        top = self.lawn.frame(t - 30.6)
        out = np.empty((H, W, 3), np.float32)
        y_seam = int(round(1080 - off))                                   # canvas: lawn 0..1080, roots frame from 930 (its ground at 150)
        if y_seam > 0: out[:y_seam] = top[int(off):int(off) + y_seam] if y_seam <= 1080 else top
        if y_seam < H:
            rf = self.R.frame(min(t, 37.7))
            r0 = 150 + (H - y_seam) - (H - y_seam)                       # roots frame row at the seam = its ground line
            out[max(y_seam, 0):] = rf[150:150 + H - max(y_seam, 0)] if 150 + H - max(y_seam, 0) <= H else np.concatenate(
                [rf[150:], np.repeat(rf[-1:], 150 + H - max(y_seam, 0) - H, 0)], 0)
        return out


class Plants:
    """52.3-56.0 一草一木, on the beat: five plant prints as cut-paper layers dropping onto one another, each a little smaller,
    with a paper edge and a soft shadow (a collage)"""
    NAMES = ['K_plant_lotus', 'K_plant_ginkgo', 'K_plant_canopy', 'K_plant_redleaf_snow', 'K_plant_blossom_snow']
    def __init__(self, beats):
        self.ims = [S16.load(kf(n)) for n in self.NAMES]; self.beats = beats; self.paper = S16.paper()
    def frame(self, t):
        img = self.paper.copy()
        for k, (im, b) in enumerate(zip(self.ims, self.beats)):
            if t < b - 0.3: continue
            u = ease(ramp(t, b - 0.3, b + 0.25))
            s = (1.0 - 0.06 * (len(self.ims) - 1 - k) * 0) * (1.0 + 0.04 * (1 - u))
            later = sum(ease(ramp(t, bb - 0.3, bb + 0.25)) for bb in self.beats[k + 1:])   # layers on top (smoothly)
            s *= 1.0 - 0.07 * later
            rot = (k % 2 * 2 - 1) * 1.6 * later
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, s); M[1, 2] += (1 - u) * -50
            card = cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
            m = (card[..., 0] >= 0).astype(np.float32)
            edge = cv2.dilate(m, np.ones((15, 15), np.uint8)) - m
            sh = cv2.GaussianBlur(np.clip(m + edge, 0, 1), (0, 0), 14); sh = np.roll(np.roll(sh, 10, 0), 8, 1)
            img = img * (1 - 0.35 * sh[..., None] * u)
            img = img * (1 - edge[..., None] * u) + self.paper * edge[..., None] * u
            img = img * (1 - m[..., None] * u) + np.where(m[..., None] > 0, card, 0) * u
        return img


class Motto:
    """73.0-80.0 求真敬业: the boulder print (its face left blank); the REAL calligraphy of the stone (業敬真求, read right to
    left, gold-filled carving), taken from the photo of the stone (work/v16k/motto_glyphs.npy) and set at its real places"""
    def __init__(self):
        self.st = S16.Parallax(kf('K_motto_stone'), move=(0, 0, 0.06), dur=7.0)
        sc = 1920 / 1383                                                   # motto_stone.png crop (268, 0, 1651, 778) -> 1080p
        self.g = []
        for gx, gy, g in np.load(ROOT / 'work/v16k/motto_glyphs.npy', allow_pickle=True):
            gg = cv2.resize(g, None, fx=sc / 2, fy=sc / 2, interpolation=cv2.INTER_AREA)
            self.g.append(((gx - 268) * sc, gy * sc, gg))
        self.g.sort(key=lambda e: -e[0])                                   # carved in reading order: right to left
    def frame(self, t):
        img = self.st.frame(t - 73.0)
        u0 = ease((t - 73.0) / 7.0)
        z = 1 + 0.06 * u0 * 1.0
        for k, (x, y, m) in enumerate(self.g):
            u = ease(ramp(t, 73.5 + 0.25 * k, 73.8 + 0.25 * k))
            if u <= 0: continue
            X = int(round((x - W / 2) * z + W / 2)); Y = int(round((y - H / 2) * z + H / 2))
            mm = cv2.resize(m, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
            mm = np.clip(mm * 2.4, 0, 1); mm = cv2.dilate(mm, np.ones((2, 2), np.uint8)); hh, ww = mm.shape
            rim = np.clip(cv2.dilate(mm, np.ones((5, 5), np.uint8)) - mm, 0, 1)            # the carved groove's dark edge
            sh = np.roll(np.roll(cv2.GaussianBlur(mm, (0, 0), 1.5), 3, 0), 3, 1)          # its shadow (light from upper left)
            reg = img[Y:Y + hh, X:X + ww]
            reg[:] = reg * (1 - 0.6 * np.maximum(sh, rim)[..., None] * u)
            gold = np.array([0.86, 0.66, 0.28], np.float32)
            reg[:] = reg * (1 - mm[..., None] * u) + gold * mm[..., None] * u
        return img

class ResearchSamara:
    """66.8-69.4 我们研究一粒种子的生长: the samara study plate (a generated print, no text) at night, labels as paper slips by code:
    翅 / 果皮 / 种子 / 子叶 / 胚根 (the structure of a samara)"""
    def __init__(self, png):
        self.pl = S16.Parallax(png, move=(0, 0, 0.04), dur=2.6)
        self.slips = {k: C.paper_slip(k, 24) for k in ('翅', '果皮', '种子', '子叶', '胚根')}
        self.anchor = [('种子', (262, 610), (300, 320), 67.25), ('胚根', (78, 650), (110, 860), 67.5), ('翅', (980, 430), (1060, 140), 67.75),
                       ('果皮', (1278, 530), (1230, 110), 68.0), ('子叶', (1520, 460), (1640, 110), 68.25)]   # (name, point, slip, time) at 1080p
    def frame(self, t):
        img = self.pl.frame(t - 66.8)
        if self.anchor:
            ink = P.Ink()
            for name, a, b, ts in self.anchor:
                ink.line([a, b], 1.1, taper=False, prog=ease(ramp(t, ts, ts + 0.25)))
            ink.onto(img, {'ink': P.INK})
            for name, a, b, ts in self.anchor:
                if t >= ts + 0.15: C.over(img, self.slips[name], b[0] - 40, b[1] - 30, ease(ramp(t, ts + 0.15, ts + 0.3)))
        return img


def film():
    t0, t1, S14, labels14, texts14, trans14 = E.film()
    seals14 = [L for L in labels14 if isinstance(L, E.Seal)]
    beat = lambda k: 48.0 + k * 0.725
    pl, pb, nm, lr = PL.Planet(), C.PaperB(), S16.Names(), LawnRoots()
    if (OUT / 'AI-B.mp4').exists():                                       # the campus grows out of the construction drawing
        nm.build = AIShot('AI-B', 23.8, 27.4, 'K_build_drawing', curve='lin').frame
    germ = AIShot('AI-G', 11.4, 16.6, 'K_aiG_samara_soil_first', 'K_aiG_seedling_last').frame if (OUT / 'AI-G.mp4').exists() else pb.frame
    # the question stands vertically at the right while the planet turns; the first clause leaves before the second arrives
    q = texts14[0]
    texts14[0] = C.RiseText('一粒种子会如何成长？', [tc for _, tc, _ in q.items] + [q.items[-1][1]], 1690, 170, px=58, wght=600, color=E.INKC,
                            vertical=True, t_out=5.1, shadow=False, gap=0.08)
    texts14[7].t_out = 29.9
    rs, ti = E.Research(), E.Title()
    bamboo = S16.Parallax(kf('K_bamboo_up'), move=(0, -60, 0.10), dur=5.4)
    shots = [
        (0.0, 5.6, pl.frame),
        (5.6, 11.4, AIShot('AI-1', 5.6, 11.4, 'T1_tianmu_ink', src=(0.0, 5.8)).frame),
        (11.4, 16.6, germ),
        (16.6, 27.4, nm.frame),
        (27.4, 30.6, AIShot('AI-2', 27.4, 30.6, 'K_ai2_library_first_clean_1280x720', 'T1_lawn_golden', src=(0.0, 5.0), curve='lin').frame),
        (30.6, 37.8, lr.frame),
        (37.8, 42.6, AIShot('AI-3', 37.8, 42.6, 'K_ai3_trunk_first', 'K_ai3_canopy_last').frame),
        (42.6, 48.0, lambda t: bamboo.frame(t - 42.6)),
        (48.0, beat(6), AIShot('AI-4', 48.0, beat(6), 'K_ai4_library_high_last', curve='out').frame),
        (beat(6), 56.0, Plants([beat(6), beat(7), beat(8), beat(9), beat(10)]).frame),
        (56.0, 57.4, (lambda P_: (lambda t: P_.frame(t - 56.0)))(S16.Parallax(kf('K_tianmu_colour'), move=(-40, 0, 0.05), dur=1.4))),
        (57.4, 59.6, (lambda P_: (lambda t: P_.frame(t - 57.4)))(S16.Parallax(kf('K_island_summer'), move=(0, 0, 0.06), dur=2.2))),
        (59.6, 66.8, AIShot('AI-5', 59.6, 66.8, 'K_ai5_library_mist_first').frame),
        (66.8, 69.4, ResearchSamara(ROOT / 'work/v16k/K_research_samara_plate_1080.png').frame),
        (69.4, 73.0, (lambda P_: (lambda t: P_.frame(t - 69.4)))(S16.Parallax(kf('CR5_one_maple_spring'), move=(0, 0, 0.05), dur=3.6))),   # a 2.5D move on the maple print (no AI shot here)
        (73.0, 80.0, Motto().frame),
        (80.0, 87.8, AIShot('AI-7', 80.0, 87.8, 'K_ai7_ginkgo_first', 'K_ai7_campus_mountains_last').frame),
        (87.8, 90.8, AIShot('AI-8', 87.8, 90.8, 'T1_swans', src=(0.0, 3.0)).frame),
        (90.8, 93.8, AIShot('AI-9', 90.8, 93.8, 'K_ai9_luming_snow_first', 'K_ai9_luming_autumn_last').frame),
        (93.8, 101.2, AIShot('AI-10', 93.8, 101.2, 'K_ai10_campus_aerial_first', 'K_ai10_clouds_ink_last').frame),
        (101.2, 108.0, ti.frame),
    ]
    trans = {1: ('ink', 1.0, dict(cx=0.5, cy=0.62)), 2: ('ink', 0.8), 4: ('luma', 0.4), 5: ('luma', 0.5), 6: ('zoom', 0.6),
             7: ('whip', 0.5, dict(dx=0, dy=-1100)), 8: ('ink', 0.7), 9: ('whip', 0.4, dict(dx=0, dy=-1000)), 10: ('whip', 0.4, dict(dx=0, dy=1000)),
             11: ('ink', 1.0), 12: ('ink', 1.0), 13: ('ink', 1.0, dict(cx=0.40, cy=0.46)), 14: ('luma', 0.6), 15: ('luma', 0.7),
             16: ('luma', 0.7), 18: ('whip', 0.5, dict(dx=0, dy=1000)), 19: ('ink', 1.2)}
    L = C.Label
    labels = [L('天目山', 7.0, 11.0, 120, 880), L('东湖校区', 48.7, beat(6) - 0.05, 120, 880), L('天目山', 56.1, 57.35, 120, 880),
              L('浙江农林大学', 81.2, 87.6, 120, 880)]
    return t0, t1, shots, labels + seals14, texts14, trans


def render(t_a=0.0, t_b=108.0, out=None):
    t0, t1, S, labels, texts, trans = film()
    out = out or ROOT / 'renders/v16/R8_picture.mov'; Path(out).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', str(out)], stdin=subprocess.PIPE)
    fa, fb = int(round((t_a - t0) * FPS)), int(round((t_b - t0) * FPS))
    for f in range(fa, fb):
        t = t0 + f / FPS
        i = max(k for k in range(len(S)) if S[k][0] <= t + 1e-6 or k == 0)
        img = None
        for j, tr in trans.items():
            bt = S[j][1]; d = tr[1]
            if bt - d / 2 <= t < bt + d / 2:
                A = S[j][2](t); B = S[j + 1][2](t)
                img = C.TRANS[tr[0]](A.astype(np.float32), B.astype(np.float32), (t - (bt - d / 2)) / d, **(tr[2] if len(tr) > 2 else {}))
                break
        if img is None: img = S[i][2](t)
        img = np.array(img, np.float32, copy=True)
        for L_ in labels: L_.draw(img, t)
        for T in texts: T.draw(img, t)
        enc.stdin.write((np.clip(img, 0, 1) * 65535 + 0.5).astype('<u2').tobytes())
        if f % 125 == 0: print(f'frame {f} t={t:.2f}', flush=True)
    enc.stdin.close(); enc.wait(); print('->', out)


if __name__ == '__main__':
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0; b = float(sys.argv[2]) if len(sys.argv) > 2 else 108.0
    o = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    render(a, b, o)

