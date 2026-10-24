"""The film: shots on the timeline, transitions, grade keys and text layers."""
import numpy as np
from timeline import T, bar, END, NAME_1958, NAME_1966, NAME_2010
from shots import FragShot
from textgen import make_text

class LT:
    """film times of the last sequence, following the score (climax to 122.7, hush, final swell ~127.3)"""
    start = T['climax']
    grow_end = T['climax'] + 15.7
    collapse = T['title']
    leaf = T['title'] + 2.6
    title = T['title'] + 4.6
    end = END


# (name, start, end, K subframes)
HIT1 = T['bamboo']
SHOTS = [
    ('roots', 0.0, 21.8, 8),
    ('bamboo', HIT1, HIT1 + 17.1, 6),
    ('land', 38.0, T['herb'] + 0.5, 6),
    ('herb', T['herb'] - 0.5, T['micro'] + 0.3, 8),
    ('micro', T['micro'] - 0.3, T['climax'] + 0.3, 6),
    ('light', T['climax'] - 0.1, END + 1.0, 6),
]
# transition into a shot: (type, duration)  type 0 dissolve, 1 white light, 2 ink bloom, 3 rising wipe, 4 luma
TRANS = {'bamboo': (1, 0.45), 'land': (0, 0.43), 'herb': (2, 1.0), 'micro': (0, 0.6), 'light': (1, 0.4)}

GRADE_DEFAULT = dict(uExposure=0.0, uBloomAmt=0.06, uHalation=0.25, uStreakAmt=0.0, uSat=1.0, uContrast=1.04, uTemp=0.0, uTint=0.0,
                     uVignette=0.35, uGrain=0.018, uAspect=2.39, uFade=0.0, uFlash=0.0, uCA=0.012, uZoom=1.0, uBloomNorm=7.0,
                     uShadow=(-0.04, 0.02, 0.03), uHigh=(0.08, 0.04, -0.06))
# (time, {param: value}) interpolated smoothly between keys
GRADE_KEYS = [
    (0.0, dict(uFade=1.0, uBloomAmt=0.35, uHalation=0.8)),
    (0.6, dict(uFade=0.0)),
    (20.8, dict(uBloomAmt=0.35, uHalation=0.8)),
    # bamboo: warm morning light
    (21.3, dict(uExposure=0.0, uTemp=0.25, uHigh=(0.08, 0.04, -0.06), uShadow=(-0.04, 0.02, 0.03), uVignette=0.35, uSat=1.12, uBloomAmt=0.14, uHalation=0.4)),
    # above the clouds: cooler, cleaner, a touch darker
    (38.0, dict(uExposure=0.0, uTemp=0.0)),
    (40.0, dict(uExposure=-0.35, uTemp=-0.35, uHigh=(0.0, 0.02, 0.04), uShadow=(-0.02, 0.0, 0.04), uVignette=0.22, uSat=1.05)),
    # ink painting: neutral paper
    (53.3, dict(uExposure=-0.35, uTemp=-0.35, uSat=1.05, uVignette=0.22)),
    (58.0, dict(uExposure=-0.1, uTemp=0.0, uHigh=(0.0, 0.0, 0.0), uShadow=(0.0, 0.0, 0.0), uSat=1.0, uVignette=0.15, uContrast=1.02)),
    # herbarium and micro: dark, rich
    (63.5, dict(uExposure=-0.1, uVignette=0.15, uBloomAmt=0.06, uHalation=0.25)),
    (64.5, dict(uExposure=0.0, uVignette=0.30, uBloomAmt=0.12, uHalation=0.3, uHigh=(0.04, 0.03, 0.0), uShadow=(-0.02, 0.01, 0.02))),
    (80.0, dict(uBloomAmt=0.12)),
    (80.6, dict(uBloomAmt=0.22, uHalation=0.2)),
    # light: glowing climax, then the letterbox opens on the name
    (101.2, dict(uBloomAmt=0.22)),
    (101.6, dict(uBloomAmt=0.40, uHalation=0.7, uVignette=0.35, uStreakAmt=0.0)),
    (122.6, dict(uAspect=2.39, uStreakAmt=0.0)),
    (LT.title - 0.4, dict(uAspect=2.39, uStreakAmt=0.15)),
    (LT.title + 2.2, dict(uAspect=1.7778, uStreakAmt=0.05)),
    (END - 1.6, dict(uFade=0.0)),
    (END - 0.1, dict(uFade=1.0)),
]

GOLD = (1.0, 0.80, 0.52)
WARM = (0.97, 0.94, 0.88)
SHADOW = (0.0, 0.0, 0.0)


def year_card(year, name, t, x=-0.56):
    return [
        dict(text=year, font='didot', px=360, tracking=0.06, h=0.125, center=(x, 0.06), t_in=t, t_out=t + 2.6, dur_in=1.0, dur_out=0.7,
             style=0, color=GOLD, glow=0.55, glow_color=SHADOW, blur=5.0),
        dict(text=name, font='song_light', px=260, tracking=0.42, h=0.052, center=(x, -0.045), t_in=t + 0.35, t_out=t + 2.6, dur_in=1.3,
             dur_out=0.7, style=2, color=WARM, glow=0.6, glow_color=SHADOW, rise=0.25, blur=3.0),
    ]


def plant(zh, la, t0, t1):
    return [dict(text=zh, font='song', px=240, tracking=0.5, h=0.046, center=(0.0, -0.262), t_in=t0, t_out=t1, dur_in=1.0, dur_out=0.6,
                 style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
            dict(text=la, font='didot_italic', px=220, tracking=0.04, h=0.040, center=(0.0, -0.312), t_in=t0 + 0.25, t_out=t1, dur_in=1.1,
                 dur_out=0.6, style=0, color=(0.80, 0.78, 0.70), glow=0.4, glow_color=SHADOW, blur=3.0)]


H0 = T['herb'] - 0.5
HERB_TEXTS = (plant('银杏', 'Ginkgo biloba', H0 + 1.4, H0 + 3.5) + plant('毛竹', 'Phyllostachys edulis', H0 + 4.3, H0 + 6.4)
              + plant('鹅掌楸', 'Liriodendron chinense', H0 + 7.5, H0 + 9.6) + plant('鸡爪槭', 'Acer palmatum', H0 + 10.7, H0 + 12.8)
              + plant('茶', 'Camellia sinensis', H0 + 13.8, H0 + 15.4) + [
    dict(text='校园\n本身就是一座植物园', font='song_light', px=240, vertical=True, tracking=0.30, line_gap=0.9, h=0.50, center=(-0.66, 0.02),
         t_in=H0 + 1.3, t_out=H0 + 6.4, dur_in=1.8, dur_out=0.9, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.08, blur=3.0, field='tb')])

M0 = T['micro']
MICRO_TEXTS = [
    dict(text='从一片叶子的呼吸，到一座森林的未来', font='song_light', px=260, tracking=0.30, h=0.050, center=(0.0, 0.292), t_in=M0 + 2.2,
         t_out=M0 + 7.0, dur_in=1.6, dur_out=0.9, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
    dict(text='9', font='didot', px=520, tracking=0.0, h=0.30, center=(-0.30, 0.0), t_in=T['rise'] + 0.4, t_out=T['rise'] + 2.9,
         dur_in=0.8, dur_out=0.5, style=0, color=GOLD, glow=0.6, glow_color=SHADOW, blur=6.0),
    dict(text='个学科领域', font='song', px=240, tracking=0.35, h=0.052, center=(0.08, 0.035), t_in=T['rise'] + 0.6, t_out=T['rise'] + 2.9,
         dur_in=1.0, dur_out=0.5, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
    dict(text='进入 ESI 全球前 1%', font='song_light', px=240, tracking=0.12, h=0.046, center=(0.10, -0.045), t_in=T['rise'] + 0.8,
         t_out=T['rise'] + 2.9, dur_in=1.0, dur_out=0.5, style=2, color=(0.85, 0.83, 0.78), glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
    dict(text='5', font='didot', px=520, tracking=0.0, h=0.30, center=(-0.30, 0.0), t_in=T['rise'] + 3.2, t_out=T['climax'] - 0.45,
         dur_in=0.8, dur_out=0.35, style=0, color=GOLD, glow=0.6, glow_color=SHADOW, blur=6.0),
    dict(text='个国家级创新平台', font='song', px=240, tracking=0.30, h=0.052, center=(0.14, 0.0), t_in=T['rise'] + 3.4,
         t_out=T['climax'] - 0.45, dur_in=1.0, dur_out=0.35, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
]

LIGHT_TEXTS = [
    dict(text='求真\n敬业', font='xingkai', px=420, vertical=True, tracking=0.10, line_gap=0.55, h=0.46, center=(0.60, 0.02),
         t_in=bar(40) - 0.2, t_out=bar(42) + 1.2, dur_in=2.6, dur_out=1.0, style=3, field='brush', color=(1.0, 0.86, 0.58), glow=0.8,
         glow_color=(0.0, 0.0, 0.0), blur=2.0),
    dict(text='坚韧不拔　不断超越', font='song_light', px=260, tracking=0.40, h=0.050, center=(0.0, -0.285), t_in=bar(43) - 0.3,
         t_out=T['title'] - 0.6, dur_in=1.8, dur_out=1.0, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
    dict(text='浙江农林大学', font='song', px=320, tracking=0.42, h=0.080, center=(0.0, -0.205), t_in=LT.title - 0.1, t_out=END - 2.2,
         dur_in=2.0, dur_out=1.4, style=2, color=(0.99, 0.96, 0.90), glow=0.5, glow_color=SHADOW, rise=0.12, blur=4.0),
    dict(text='ZHEJIANG A&F UNIVERSITY', font='optima', px=220, tracking=0.42, h=0.036, center=(0.0, -0.296), t_in=LT.title + 0.7,
         t_out=END - 2.2, dur_in=1.8, dur_out=1.4, style=0, color=(0.93, 0.89, 0.80), glow=0.4, glow_color=SHADOW, blur=4.0),
]

LAND_TEXTS = [
    dict(text='一山一水，皆是课堂', font='song', px=260, tracking=0.40, h=0.054, center=(0.0, 0.265), t_in=T['land'] + 2.0,
         t_out=T['land'] + 7.6, dur_in=1.8, dur_out=1.0, style=2, color=(0.10, 0.11, 0.12), glow=0.35, glow_color=(0.95, 0.95, 0.95), rise=0.2, blur=3.0),
]

TEXTS = LAND_TEXTS + LIGHT_TEXTS + MICRO_TEXTS + HERB_TEXTS + (year_card('1958', '天目林学院', NAME_1958) + year_card('1966', '浙江林学院', NAME_1966) + year_card('2010', '浙江农林大学', NAME_2010) + [
    dict(text='扎根，是为了向上', font='song_light', px=280, tracking=0.38, h=0.056, center=(0.0, 0.292), t_in=18.75, t_out=20.35, dur_in=1.4,
         dur_out=0.8, style=2, color=WARM, glow=0.5, glow_color=SHADOW, rise=0.2, blur=3.0),
])


from camera import Path, cam_uniforms
import math


def _S(a, b, t):
    x = min(1.0, max(0.0, (t - a) / (b - a)))
    return x * x * x * (x * (x * 6 - 15) + 10)


BAMBOO_CAM = Path([
    (0.0,  (1.05, 0.30, 0.85),  (0.0, 0.42, 0.0), 48, 0.006),
    (2.5,  (1.25, 0.95, 0.55),  (0.0, 1.75, 0.0), 50, 0.006),
    (6.0,  (1.45, 3.40, -0.35), (0.0, 5.00, 0.0), 52, 0.004),
    (10.5, (1.00, 7.60, -1.25), (0.1, 9.80, 0.0), 54, 0.003),
    (14.5, (0.30, 11.8, -1.55), (0.5, 14.8, 0.9), 56, 0.002),
    (17.1, (-0.2, 15.6, -1.20), (1.2, 18.5, 3.0), 58, 0.0),
])


def frame_left(u, amount):
    """shift the look-at point sideways so the subject on the unshifted view axis sits on the left third"""
    import numpy as np
    eye, tgt = np.array(u['uEye']), np.array(u['uTarget'])
    f = tgt - eye; d = np.linalg.norm(f); f /= d
    r = np.cross(f, (0, 1, 0)); r /= np.linalg.norm(r)
    u['uTarget'] = tuple(float(x) for x in tgt + r * d * amount)
    return u


def bamboo_uniforms(t):
    u = frame_left(cam_uniforms(BAMBOO_CAM, t), 0.34)
    burst = 1 - (1 - min(1.0, max(0.0, t / 0.5))) ** 3
    u['uHeroH'] = 0.22 + 0.5 * burst + max(0.0, t - 0.5) * 0.98 + 0.25 * _S(0.5, 3.0, t)
    u['uBurst'] = t
    u['uLeaves'] = _S(15.0, 16.3, t)
    u['uSeed'] = 0.0
    return u


LAND0 = 38.0
LAND_CAM = Path([   # local time = film time - 38.0: rise off the ridge, dive to the cloud tops on the hit, race over them, glide to rest
    (0.0,   (0.0, 229.0, -20.0),    (0.0, 208.0, 420.0), 58, 0.0),
    (2.2,   (0.0, 236.0, 80.0),     (10.0, 195.0, 900.0), 56, 0.0),
    (4.67,  (30.0, 215.0, 380.0),   (60.0, 150.0, 2600.0), 54, 0.0),
    (8.0,   (60.0, 168.0, 1100.0),  (80.0, 140.0, 3200.0), 56, 0.0),
    (11.5,  (70.0, 185.0, 2200.0),  (100.0, 170.0, 4800.0), 54, 0.0),
    (15.33, (60.0, 230.0, 3200.0),  (100.0, 220.0, 6500.0), 50, 0.0),
    (22.0,  (55.0, 250.0, 3700.0),  (100.0, 232.0, 7000.0), 48, 0.0),
    (26.4,  (55.0, 252.0, 3780.0),  (100.0, 233.0, 7050.0), 48, 0.0),
])


def land_uniforms(t):
    u = cam_uniforms(LAND_CAM, t)
    u['uRoll'] = (0.06 * math.sin(t * 0.45 + 0.6) * _S(4.0, 7.0, t) - 0.04 * _S(7.5, 10.5, t)) * (1 - _S(12.5, 16.5, t))
    u['uLeaves'] = 1.0 - _S(0.0, 0.7, t)
    ink = T['ink'] - LAND0
    u['uInk'] = _S(ink, ink + 6.2, t)
    lt0, lt1 = 22.4, 25.0
    u['uLeaf'] = (t - lt0) / (lt1 - lt0) if t >= lt0 else -1.0
    u['uLeaf'] = min(u['uLeaf'], 1.0)
    u['uRipple'] = t - lt1 if t >= lt1 else -1.0
    u['uLeafPos'] = (-0.30, -0.25)
    u['uBirds'] = t - 12.5 if 12.5 <= t <= 26.5 else -1.0
    return u


HERB0 = T['herb'] - 0.5
HERB_X = [0.0, 1.65, 3.30, 4.95, 6.60]
def _hold(x):
    return (x + 0.03, 0.10, -1.75), (x, 0.10, 0.1)


HERB_CAM = Path([   # local time = film time - HERB0: pause on each specimen while its label shows
    (0.0,  (0.0, 0.12, -0.80), (0.0, 0.14, 0.0), 36, 0.015),
    (1.6,  *_hold(0.0), 34, 0.030),
    (3.4,  *_hold(0.10), 34, 0.030),
    (4.5,  *_hold(1.65), 34, 0.030),
    (6.4,  *_hold(1.75), 34, 0.030),
    (7.6,  *_hold(3.30), 34, 0.030),
    (9.6,  *_hold(3.40), 34, 0.030),
    (10.8, *_hold(4.95), 34, 0.030),
    (12.8, *_hold(5.05), 34, 0.030),
    (13.9, *_hold(6.60), 34, 0.030),
    (15.3, *_hold(6.64), 33, 0.025),
    (16.8, (6.6, 0.08, -0.30), (6.6, 0.08, 0.3), 30, 0.010),
])


def herb_uniforms(t):
    import numpy as np
    u = cam_uniforms(HERB_CAM, t)
    eye, tgt = np.array(u['uEye']), np.array(u['uTarget'])
    k = int(np.argmin([abs(x - tgt[0]) for x in HERB_X]))
    u['uFocus'] = float(np.linalg.norm(np.array([HERB_X[k], 0.1, 0.1]) - eye))
    return u


MICRO0 = T['micro'] - 0.3
MICRO_CAM = Path([   # local time = film time - MICRO0 (cell units, 1 ~ 20 um)
    (0.0,  (0.2, 6.5, -1.5),    (0.5, 0.0, 0.8), 50, 0.0),
    (3.8,  (0.6, 1.2, 0.2),     (1.1, -1.5, 1.8), 50, 0.0),
    (6.3,  (0.9, -0.9, 1.0),    (1.6, -4.0, 2.6), 52, 0.0),
    (10.3, (1.6, -4.2, 2.3),    (2.4, -6.5, 3.3), 52, 0.0),
    (13.8, (2.25, -6.45, 3.15), (2.55, -6.95, 3.55), 50, 0.0),
    (15.9, (2.40, -6.72, 3.33), (2.55, -6.95, 3.55), 46, 0.0),
    (16.6, (2.36, -6.66, 3.28), (2.55, -6.95, 3.55), 46, 0.0),
    (21.9, (-2.0, 8.0, -6.0),   (2.2, -6.0, 3.0), 44, 0.0),
])


def micro_uniforms(t):
    u = cam_uniforms(MICRO_CAM, t)
    u['uCO2'] = _S(1.5, 3.5, t) * (1 - _S(17.0, 19.0, t))
    u['uFix'] = _S(13.0, 15.0, t) * (1 - _S(17.0, 18.5, t))
    u['uPull'] = _S(16.6, 18.5, t)
    u['uDepthFade'] = 0.22 - 0.17 * _S(16.6, 20.0, t)
    u['uSection'] = (1.3 * _S(4.5, 6.5, t) - 0.7 * _S(12.0, 14.5, t)) * (1 - _S(16.0, 17.5, t))
    return u


class Film:
    def __init__(self, eng):
        self.eng = eng
        self.shots = []
        for name, s, e, K in SHOTS:
            self.shots.append((self.make(eng, name, K), s, e, name))
        self.texts = []
        for i, d in enumerate(TEXTS):
            d = dict(d)
            arr = make_text(d['text'], d.get('font', 'song'), d.get('px', 300), d.get('vertical', False), d.get('tracking', 0.05),
                            d.get('line_gap', 0.4), d.get('field', 'lr'))
            d['arr'] = arr; d['key'] = f'text{i}'
            h = d['h']; asp = arr.shape[1] / arr.shape[0]
            d['size'] = (h * asp, h)
            self.texts.append(d)

    def make(self, eng, name, K):
        if name == 'roots':
            from shot_roots import RootsShot
            sh = RootsShot(eng); sh.K = K; self.roots = sh
            return sh
        if name == 'bamboo':
            return FragShot(eng, 'bamboo.glsl', K, uniforms=bamboo_uniforms)
        if name == 'land':
            return FragShot(eng, 'land.glsl', K, uniforms=land_uniforms)
        if name == 'herb':
            return FragShot(eng, 'herb.glsl', K, uniforms=herb_uniforms)
        if name == 'micro':
            return FragShot(eng, 'micro.glsl', K, uniforms=micro_uniforms)
        if name == 'light':
            from shot_light import LightShot
            sh = LightShot(eng, LT); sh.K = K
            return sh
        return FragShot(eng, f'{name}.glsl', K)

    def active(self, t):
        cur = [(sh, s, e, n) for sh, s, e, n in self.shots if s <= t < e]
        if not cur:
            sh, s, e, n = self.shots[-1] if t >= self.shots[-1][2] else self.shots[0]
            return [(sh, t - s)]
        if len(cur) == 1:
            sh, s, e, n = cur[0]
            return [(sh, t - s)]
        a, b = cur[0], cur[1]
        typ, dur = TRANS.get(b[3], (0, a[2] - b[1]))
        mix = np.clip((t - b[1]) / max(1e-6, dur), 0, 1)
        return [(a[0], t - a[1]), (b[0], t - b[1]), float(mix), int(typ), float(hash(b[3]) % 97)]

    def samples(self, t):
        return max(sh.K for sh, s, e, n in self.shots if s <= t < e) if any(s <= t < e for _, s, e, _ in self.shots) else 4

    def shutter(self, t):
        return 0.5

    def grade(self, t):
        g = dict(GRADE_DEFAULT)
        keys = {}
        for tk, d in GRADE_KEYS:
            for k, v in d.items():
                keys.setdefault(k, []).append((tk, v))
        for k, lst in keys.items():
            lst.sort(key=lambda x: x[0])
            if t <= lst[0][0]:
                g[k] = lst[0][1]; continue
            if t >= lst[-1][0]:
                g[k] = lst[-1][1]; continue
            for (t0, v0), (t1, v1) in zip(lst, lst[1:]):
                if t0 <= t <= t1:
                    x = (t - t0) / max(1e-6, t1 - t0); x = x * x * (3 - 2 * x)
                    if isinstance(v0, tuple):
                        g[k] = tuple(a + (b - a) * x for a, b in zip(v0, v1))
                    else:
                        g[k] = v0 + (v1 - v0) * x
                    break
        return g

    def texts_at(self, t):
        out = []
        for d in self.texts:
            t_in, t_out = d['t_in'], d['t_out']
            di, do = d.get('dur_in', 1.2), d.get('dur_out', 0.8)
            if t < t_in or t > t_out + do:
                continue
            e = dict(d)
            e['pin'] = float(np.clip((t - t_in) / di, 0, 1))
            e['pout'] = float(np.clip((t - t_out) / do, 0, 1))
            out.append(e)
        return out
