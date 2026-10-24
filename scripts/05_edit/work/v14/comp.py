"""Compositor for the REAL-FOOTAGE parts of the film: shots cut on the music, pushed / reframed in 16:9,
one grade for every source (ink blacks, paper whites), and the paper world laid into today's pictures: 标本签 labels (a slip of
宣纸, a Song-type name, a small 朱印), captions that rise out of a mask line, big vertical type.

  .venv/bin/python work/v14/comp.py segA            -> renders/v14/segA.mp4 (1080p, voice+music test mix)
"""
import sys, json, math, subprocess, glob, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1920, 1080, 25
CLIPS = ROOT / 'work/v14/clips'
SERIF = str(ROOT / 'data/fonts/NotoSerifSC[wght].ttf')
WT = json.load(open(ROOT / 'work/v14/word_times.json'))
PAPER = np.asarray(Image.open(ROOT / 'data/v14/paper.png').convert('RGB'), np.float32) / 255


def ease(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def out_cubic(x): x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3
def ramp(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))


def font(px, wght=500):
    f = ImageFont.truetype(SERIF, int(px))
    try: f.set_variation_by_axes([wght])
    except Exception: pass
    return f


# ------------------------------------------------------------------ sources
class Clip:
    """a ProRes clip (or a JPEG frame folder) read at film rate with a speed factor; slow motion is interpolated"""
    def __init__(self, src, t0, speed=1.0, fps_src=None, crop=None, scale=None, reverse=False):
        self.src, self.t0, self.speed, self.crop, self.scale, self.reverse = src, t0, speed, crop, scale, reverse
        self.cache = {}
        if os.path.isdir(src):
            self.files = sorted(glob.glob(f'{src}/*.jpg')) or sorted(glob.glob(f'{src}/*.png')); self.fps_src = fps_src or 30.0
        else:
            self.files = None

    def frames(self, n):
        """n consecutive film frames starting at source time t0"""
        if self.files:
            out = []
            for k in range(n):
                i = int(round(self.t0 * self.fps_src + k * self.speed * self.fps_src / FPS))
                out.append(cv2.cvtColor(cv2.imread(self.files[min(i, len(self.files) - 1)]), cv2.COLOR_BGR2RGB))
            return out
        vf = [f'crop={self.crop[2]}:{self.crop[3]}:{self.crop[0]}:{self.crop[1]}'] if self.crop else []
        if self.scale: vf.append(f'scale={self.scale}:-2:flags=lanczos')
        if self.speed < 0.85: vf.append(f'minterpolate=fps={FPS / self.speed:.3f}:mi_mode=mci:mc_mode=aobmc:vsbmc=1')
        vf += [f'setpts=(PTS-STARTPTS)/{self.speed}', f'fps={FPS}']
        t_start = max(0.0, self.t0 - n * self.speed / FPS) if self.reverse else self.t0
        p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t_start), '-i', str(self.src), '-vf', ','.join(vf), '-frames:v', str(n),
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
        a = np.frombuffer(p.stdout, np.uint8)
        h0, w0 = self.size()
        a = a.reshape(-1, h0, w0, 3)
        fr = list(a)
        while len(fr) < n: fr.append(fr[-1])
        fr = fr[:n]
        return fr[::-1] if self.reverse else fr

    def size(self):
        if self.files: im = cv2.imread(self.files[0]); return im.shape[:2]
        if self.crop: return self.crop[3], self.crop[2]
        if self.scale:
            o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(self.src)], capture_output=True, text=True).stdout.strip().split(',')
            w0, h0 = int(o[0]), int(o[1]); return int(round(h0 * self.scale / w0 / 2) * 2), self.scale
        o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(self.src)], capture_output=True, text=True).stdout.strip().split(',')
        return int(o[1]), int(o[0])


def reframe(img, cx, cy, zoom):
    """16:9 window centred (cx, cy) in source pixels; zoom 1 = the widest 16:9 window that fits the source"""
    h0, w0 = img.shape[:2]
    ww = min(w0, h0 * 16 / 9) / zoom; hh = ww * 9 / 16
    x = min(max(cx - ww / 2, 0), w0 - ww); y = min(max(cy - hh / 2, 0), h0 - hh)
    s = W / ww
    M = np.float32([[s, 0, -x * s], [0, s, -y * s]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ the grade (one look for every source)
INK = np.array([0.045, 0.042, 0.038], np.float32); WHITE = np.array([0.972, 0.945, 0.885], np.float32)
def grade(rgb8, exp=0.0, con=1.0, sat=1.0, temp=0.0, tint=0.0):
    x = rgb8.astype(np.float32) / 255
    x = x ** 2.2 * (2 ** exp)                                   # to linear, exposure
    x[..., 0] *= 1 + temp * 0.06; x[..., 2] *= 1 - temp * 0.06; x[..., 1] *= 1 + tint * 0.04
    x = np.clip(x, 0, None) ** (1 / 2.2)
    l = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    x = l[..., None] + (x - l[..., None]) * sat
    x = np.clip((x - 0.5) * con + 0.5, 0, 1)
    # soft shoulder and toe into ink / paper
    x = x * x * (3 - 2 * x) * 0.35 + x * 0.65
    return INK + (WHITE - INK) * x


# ------------------------------------------------------------------ overlays
def paper_slip(text, px=26):
    """a 标本签: a small slip of 宣纸 with a deckled edge, the name in Song type, a small 朱印 (阴文 农林)"""
    f = font(px, 600)
    tw = f.getbbox(text)[2]
    sw, sh = int(tw + px * 2.9), int(px * 2.25)
    sl = Image.new('RGBA', (sw + 24, sh + 24), (0, 0, 0, 0))
    y0, x0 = np.random.default_rng(len(text)).integers(0, 400, 2)
    tex = (PAPER[y0:y0 + sh, x0:x0 + sw] * 255 * 1.03).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(abs(hash(text)) % 2 ** 32)
    a = np.full((sh, sw), 255, np.uint8)
    for side in range(4):                                             # torn, slightly uneven edges
        n = sw if side < 2 else sh
        e = (np.convolve(rng.random(n), np.ones(9) / 9, 'same') * 3).astype(int)
        for i in range(n):
            if side == 0: a[:e[i], i] = 0
            elif side == 1: a[sh - e[i]:, i] = 0
            elif side == 2: a[i, :e[i]] = 0
            else: a[i, sw - e[i]:] = 0
    slip = Image.fromarray(np.dstack([tex, a]), 'RGBA')
    shadow = Image.new('RGBA', sl.size, (0, 0, 0, 0)); shadow.paste((20, 16, 12, 90), (14, 15, sw + 14, sh + 15), Image.fromarray(a).resize((sw, sh)))
    sl = Image.alpha_composite(sl, shadow.filter(ImageFilter.GaussianBlur(5)))
    sl.paste(slip, (10, 10), slip)
    d = ImageDraw.Draw(sl)
    d.text((10 + px * 0.6, 10 + (sh - px * 1.18) / 2), text, font=f, fill=(34, 30, 25, 255))
    # 朱印
    s = int(px * 1.15); sx, sy = 10 + sw - s - int(px * 0.55), 10 + (sh - s) // 2
    seal = Image.new('RGBA', (s, s), (0, 0, 0, 0)); sd = ImageDraw.Draw(seal)
    sd.rounded_rectangle([0, 0, s - 1, s - 1], radius=2, fill=(178, 42, 30, 235))
    sf = font(s * 0.36, 700)
    for k, ch in enumerate('农林'):
        sd.text((s * 0.5 - s * 0.19, s * 0.08 + k * s * 0.42), ch, font=sf, fill=(246, 236, 220, 255))
    ns = (np.random.default_rng(7).random((s, s)) > 0.12).astype(np.uint8) * 255      # stamp texture
    seal.putalpha(Image.fromarray(np.minimum(np.asarray(seal)[..., 3], ns)))
    sl.alpha_composite(seal, (sx, sy))
    return np.asarray(sl, np.float32) / 255


def over(base, rgba, x, y, alpha=1.0, angle=0.0, scale=1.0):
    """composite an RGBA float image onto base (float RGB) at top-left (x, y) with rotation / scale about its centre"""
    if angle or scale != 1.0:
        h, w = rgba.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        rgba = cv2.warpAffine(rgba, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    h, w = rgba.shape[:2]; x, y = int(round(x)), int(round(y))
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa >= xb or ya >= yb: return
    src = rgba[ya - y:yb - y, xa - x:xb - x]; a = src[..., 3:4] * alpha
    base[ya:yb, xa:xb] = base[ya:yb, xa:xb] * (1 - a) + src[..., :3] * a


class Label:
    """a slip pinned into the frame: drops in on a spring, the seal lands a beat later"""
    def __init__(self, text, t0, t1, x, y, rot=-1.5):
        self.img = paper_slip(text); self.t0, self.t1, self.x, self.y, self.rot = t0, t1, x, y, rot
    def draw(self, base, t):
        if not (self.t0 <= t < self.t1): return
        u = ramp(t, self.t0, self.t0 + 0.45)
        spring = 1 - math.exp(-6 * u) * math.cos(9 * u)
        a = ease(ramp(t, self.t0, self.t0 + 0.12)) * (1 - ease(ramp(t, self.t1 - 0.25, self.t1)))
        over(base, self.img, self.x, self.y + (1 - spring) * -14, a, self.rot + (1 - spring) * -5)


def glyph(ch, px, wght, color):
    f = font(px, wght); b = f.getbbox(ch)
    im = Image.new('RGBA', (int(px * 1.2), int(px * 1.35)), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((px * 0.1, px * 0.05), ch, font=f, fill=color)
    return np.asarray(im, np.float32) / 255


class RiseText:
    """characters rise out of a mask line (each clipped below its own baseline), staggered; horizontal or vertical"""
    def __init__(self, text, times, x, y, px=36, wght=500, color=(244, 233, 213, 255), vertical=False, t_out=None, shadow=True, gap=0.0):
        self.items = []; self.px = px; self.vertical = vertical; self.t_out = t_out; self.shadow = shadow
        pos = 0.0
        for ch, tc in zip(text, times):
            if ch == ' ':
                pos += px * 0.55; continue
            g = glyph(ch, px, wght, color)
            self.items.append((g, tc, pos)); pos += px * (1.0 + gap)
        self.total = pos
        self.x = x - (0 if vertical else self.total / 2); self.y = y
    def draw(self, base, t):
        fade = 1.0 if self.t_out is None else 1 - ease(ramp(t, self.t_out, self.t_out + 0.3))
        if fade <= 0: return
        for g, tc, pos in self.items:
            u = out_cubic(ramp(t, tc, tc + 0.42))
            if u <= 0: continue
            h, w = g.shape[:2]
            rise = (1 - u) * self.px * 0.55
            gx, gy = (self.x, self.y + pos) if self.vertical else (self.x + pos, self.y)
            cut = int(round(h - (h * u)))                                   # the mask line: only the risen part shows
            vis = g.copy(); vis[max(0, h - int(rise) - 1):, :, 3] *= 0 if u < 1 else 1
            if self.shadow:
                sh = cv2.GaussianBlur(vis[..., 3], (0, 0), self.px * 0.16)
                over(base, np.dstack([np.zeros_like(vis[..., :3]), np.clip(sh[..., None] * 1.6, 0, 0.85)]), gx, gy - rise + 2, fade * u)
            over(base, vis, gx, gy - rise, fade * u)


def char_times(u, text, t_from=None, t_to=None):
    """one time per character of text, from the whisper word times of utterance u (characters spread inside each word)"""
    ws = WT[u]['words']; seq = []
    for w, a, b in ws:
        cs = [c for c in w if c not in '，。,.、！？!? ']
        for k, c in enumerate(cs): seq.append((c, a + (b - a) * k / max(len(cs), 1)))
    out, j = [], 0
    for ch in text:
        if ch in '，。 ': out.append(None); continue
        k = next((q for q in range(j, min(j + 3, len(seq))) if seq[q][0] == ch), None)   # a near match, else by position
        if k is None: k = min(j, len(seq) - 1)
        out.append(seq[k][1]); j = k + 1
    # fill punctuation slots
    for i in range(len(out)):
        if out[i] is None: out[i] = out[i - 1] if i else out[i + 1]
    return out


def caption(u, text, t_out):
    """a bottom caption of one clause; punctuation becomes space"""
    ts = char_times(u, text)
    shown = ''.join(' ' if c in '，。' else c for c in text)
    return RiseText(shown, ts, W / 2, H - 118, px=40, wght=500, t_out=t_out)


# ------------------------------------------------------------------ segment A: 47.92 - 66.8
SJ = f'{os.environ["HOME"]}/Desktop/Ori_Pre/浙江农林大学校园十景（删油橄榄.mp4'; SJC = (0, 151, 1920, 778)
YX = f'{os.environ["HOME"]}/Desktop/Ori_Pre/银杏终稿.mp4'


def segA():
    beat = lambda k: 48.0 + k * 0.725
    S = []          # (film_a, film_b, Clip, (cx0, cy0, z0), (cx1, cy1, z1), grade kw)
    S.append((47.92, beat(6), Clip(ROOT / 'footage/dive_library', 2100 / 30, speed=2.6), (960, 540, 1.55), (960, 540, 1.0), dict(exp=0.15, con=1.12, sat=1.08, temp=0.4)))
    # 一草一木: five plants through the year, from several sources (校园十景, own DJI_0208, 雪韵·农林); names only where verified
    XY = f'{os.environ["HOME"]}/Desktop/Ori_Pre/雪韵·农林.mp4'
    S.append((beat(6), beat(7), Clip(SJ, 147.6, crop=SJC), (960, 389, 1.0), (960, 389, 1.08), dict(sat=0.95, temp=0.1)))           # 荷, summer
    S.append((beat(7), beat(8), Clip(SJ, 250.7, crop=SJC), (960, 389, 1.02), (960, 389, 1.1), dict(temp=0.05)))                     # 银杏, autumn
    S.append((beat(8), beat(9), Clip(ROOT / 'footage/topdown_lake', 1200 / 30), (960, 540, 1.15), (960, 540, 1.25), dict(con=1.08, sat=1.05, temp=0.15)))   # canopy, own DJI_0208
    S.append((beat(9), beat(10), Clip(XY, 0.4), (960, 540, 1.0), (980, 540, 1.08), dict(exp=0.05, temp=0.0)))                       # snow on red leaves, winter
    S.append((beat(10), beat(11), Clip(XY, 4.3), (960, 540, 1.0), (960, 540, 1.08), dict(exp=0.0, temp=0.05)))                      # blossoms in snow
    S.append((beat(11), beat(13), Clip(SJ, 4.0, crop=SJC), (960, 389, 1.0), (960, 389, 1.08), dict(exp=0.05, con=1.05, temp=0.2)))
    S.append((beat(13), 59.6, Clip(YX, 17.5), (960, 540, 1.0), (960, 540, 1.1), dict(con=1.05, sat=1.0, temp=0.1)))
    S.append((59.6, 66.8, Clip(SJ, 10.5, speed=0.5, crop=SJC), (960, 389, 1.0), (1000, 380, 1.1), dict(exp=0.05, con=1.04, temp=0.05)))
    labels = [Label('东湖校区', 48.7, beat(6) - 0.05, 120, 880),
              Label('银杏', beat(7) + 0.05, beat(8) - 0.02, 120, 880),
              Label('天目山', beat(11) + 0.15, beat(13) - 0.05, 120, 880),
              Label('鹿鸣岛', beat(13) + 0.15, 59.55, 120, 880)]
    texts = [caption('u10', '整个校园就是一座植物园', 51.4),
             caption('u11', '在这里，一草一木，皆有学问', 56.1), caption('u11', '一山一水，皆是课堂', 59.45)]
    ts1 = char_times('u12', '博览古今'); ts2 = char_times('u12', '学参天地')
    ink = (40, 34, 28, 235)
    texts += [RiseText('博览古今', ts1, 1660, 40, px=66, wght=600, color=ink, vertical=True, t_out=66.4, shadow=False, gap=0.12),
              RiseText('学参天地', ts2, 1560, 40, px=66, wght=600, color=ink, vertical=True, t_out=66.4, shadow=False, gap=0.12)]
    return 47.92, 66.8, S, labels, texts


BAND = (1 - 0.22 * np.clip((np.arange(H)[:, None, None] - H * 0.78) / (H * 0.22), 0, 1) ** 1.5).astype(np.float32)   # a soft darker floor under captions


def render(name):
    t0, t1, S, labels, texts = {'segA': segA, 'segB': segB}[name]()
    out = ROOT / f'renders/v14/{name}.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
    n_all = int(round((t1 - t0) * FPS))
    aud = ROOT / 'audio/v3/out/v14vm/stereo/master.wav'
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-ss', str(t0), '-t', str(t1 - t0), '-i', str(aud), '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '256k', '-shortest', str(out)], stdin=subprocess.PIPE)
    fidx = 0
    for sp in S:
        a, b, clip, k0, k1, g = sp[:6]; clean = sp[6] if len(sp) > 6 else []
        fa, fb = int(round((a - t0) * FPS)), int(round((b - t0) * FPS)); n = fb - fa
        if callable(clip):
            for k in range(n):
                t = t0 + (fa + k) / FPS
                img = clip(t)
                for L in labels: L.draw(img, t)
                for T in texts: T.draw(img, t)
                enc.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()); fidx += 1
            print(f'{a:6.2f}-{b:6.2f} {n} frames (drawn)', flush=True)
            continue
        frames = clip.frames(n)
        for k, fr in enumerate(frames):
            t = t0 + (fa + k) / FPS
            if clean:
                fr = np.ascontiguousarray(fr); m = np.zeros(fr.shape[:2], np.uint8); ul = k / max(n - 1, 1)
                for b0, b1 in clean:
                    bx = [int(round(b0[i] + (b1[i] - b0[i]) * ul)) for i in range(4)]; m[bx[1]:bx[3], bx[0]:bx[2]] = 255
                fr = cv2.inpaint(fr, m, 7, cv2.INPAINT_TELEA)
            u = ease(k / max(n - 1, 1))
            cx, cy, z = [k0[i] + (k1[i] - k0[i]) * u for i in range(3)]
            img = grade(reframe(fr, cx, cy, z), **g)
            img *= BAND
            for L in labels: L.draw(img, t)
            for T in texts: T.draw(img, t)
            enc.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()); fidx += 1
        print(f'{a:6.2f}-{b:6.2f} {n} frames', flush=True)
    enc.stdin.close(); enc.wait()
    print('->', out, fidx, 'frames of', n_all)



# ================================================================== segment B: 11.4 - 37.8 (paper world, names, first real picture, roots)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('paper', ROOT / 'work/v14/paper.py'); P = _ilu.module_from_spec(_spec); _spec.loader.exec_module(P)
INKC = (40, 33, 26, 255)


def ink_caption(text, times, t_out, y=H - 118):
    shown = ''.join(' ' if c in '，。、' else c for c in text)
    return RiseText(shown, times, W / 2, y, px=40, wght=500, color=INKC, t_out=t_out, shadow=False)


class PaperB:
    """萌芽 -> the shoot becomes the 68-year timeline -> three names in movable type (shared characters stay)"""
    def __init__(self):
        r = np.random.default_rng(3)
        self.hatch = []                                     # woodblock soil hatching (world coords)
        for k in range(420):
            x, y = r.uniform(140, 1780), 612 + r.random() ** 1.6 * 700
            L = r.uniform(9, 22); a = math.radians(-35 + r.normal(0, 8))
            self.hatch.append(((x, y), (x + L * math.cos(a), y + L * math.sin(a)), r.uniform(0.9, 1.5)))
        self.ground = [(x, 600 + 2.5 * math.sin(x / 57.0) + 1.5 * math.sin(x / 13.0)) for x in range(-300, 2300, 12)]
        self.lat = [(0.28 + 0.1 * k, (-1) ** k, 45 + 9 * k, 13.3 + 0.15 * k) for k in range(6)]
        # names
        L5 = lambda i: 1110 + (i - 2) * 168; L6 = lambda i: 1110 + (i - 2.5) * 168
        self.chars = [  # ch, [(t, x)], t_in, t_out, era roughness
            ('天', [(0, L5(0))], 18.44, 21.25, 1.0), ('目', [(0, L5(1))], 18.76, 21.35, 1.0),
            ('林', [(0, L5(2)), (24.55, L6(3))], 18.98, 99, 1.0), ('学', [(0, L5(3)), (24.65, L6(5))], 19.18, 99, 1.0),
            ('院', [(0, L5(4))], 19.40, 24.40, 1.0),
            ('浙', [(0, L5(0)), (24.5, L6(0))], 21.67, 99, 0.55), ('江', [(0, L5(1)), (24.5, L6(1))], 21.95, 99, 0.55),
            ('农', [(0, L6(2))], 25.91, 99, 0.0), ('大', [(0, L6(4))], 26.33, 99, 0.0)]
        self.masks = {}
        for ch, *_ in self.chars:
            self.masks[ch] = {r_: P.type_mask(ch, 150, r_, seed=ord(ch)) for r_ in (1.0, 0.55, 0.0)}
        self.seals = [(P.seal('一九五八', 64), 19.75, 1958), (P.seal('一九六六', 64), 22.75, 1966), (P.seal('二〇一〇', 64), 26.75, 2010)]
        self.digits = {}
        sky = json.load(open(ROOT / 'data/v14/skyline.json'))
        yy = np.arange(H, dtype=np.float32)[:, None]
        self.wash = np.zeros((H, W), np.float32)
        for band, a0 in (('far', 0.10), ('mid', 0.15), ('near', 0.22)):
            prof = np.interp(np.linspace(0, len(sky[band]) - 1, W), np.arange(len(sky[band])), sky[band])
            edge = (1010 - np.maximum(prof, 0) * 3000).astype(np.float32)[None, :]
            inside = np.clip((yy - edge) / 2.5, 0, 1)
            mist = np.clip(1 - (yy - edge) / (110 if band == 'far' else 160), 0.15, 1)
            self.wash = np.maximum(self.wash, inside * mist * a0)

    def ty(self, Y): return 930 - (Y - 1958) / 68 * 780            # timeline screen y of a year

    def frame(self, t):
        dx = -660 * ease(ramp(t, 14.9, 16.4)); dy = 700 * ease(ramp(t, 14.6, 16.4))
        age = 1.0 - 0.45 * ease(ramp(t, 20.2, 21.0)) - 0.55 * ease(ramp(t, 23.8, 24.4))
        img = P.paper(age); ink = P.Ink()
        S = lambda p: (p[0] + dx, p[1] + dy)
        if dy < 1000:
            ink.line([S(p) for p in self.ground], 1.8)
            for a, b, w in self.hatch:
                if a[1] + dy < H + 20: ink.line([S(a), S(b)], w, alpha=0.55)
        # the seed: two halves of a coat, opening
        op = ease(ramp(t, 12.0, 12.6))
        for half in (1, -1):
            th = np.linspace(0, math.pi, 24) if half > 0 else np.linspace(math.pi, 2 * math.pi, 24)
            rot = math.radians(15 + half * 9 * op)
            pts = [(960 + 56 * math.cos(q) * math.cos(rot) - 38 * math.sin(q) * math.sin(rot), 700 - half * 6 * op - (56 * math.cos(q) * math.sin(rot) + 38 * math.sin(q) * math.cos(rot))) for q in th]
            ink.line([S(p) for p in pts], 3.0, taper=False)
        # radicle and its laterals
        rp = ease(ramp(t, 12.5, 13.9))
        R = [(972 + 34 * math.sin(s * 2.2), 734 + s * 330) for s in np.linspace(0, 1, 40)]
        ink.line([S(p) for p in R], 3.4, prog=rp)
        for s, side, L, t0 in self.lat:
            lp = ease(ramp(t, t0, t0 + 0.6))
            if lp > 0:
                o = R[int(s * 39)]
                pts = [(o[0] + side * 1.8 * L * u, o[1] + 0.6 * L * u * u) for u in np.linspace(0, 1, 10)]
                ink.line([S(p) for p in pts], 2.0, prog=lp)
        # the shoot: a hook that straightens once it is out of the soil, then the stem rises to become the timeline
        h = ease(ramp(t, 13.1, 16.4)) * (664 + 550)
        if h > 1:
            tip = 664 - h; k = 1 - ramp(600 - tip, 0, 120)
            n = max(4, int(h / 6)); pts = [(960, 664 - h * u) for u in np.linspace(0, 1, n)]
            if k > 0:
                hook = [(960 + 40 * k * (1 - math.cos(a)), tip - 40 * k * math.sin(a) * 0.6) for a in np.linspace(0, math.radians(150) * k, 12)][1:]
                pts += hook
            ink.line([S(p) for p in pts], 3.2, taper=False)
        # the 68 years along the stem
        for Y in range(1958, 2027):
            tp = ease(ramp(t, 15.8 + (Y - 1958) / 68 * 0.7, 15.95 + (Y - 1958) / 68 * 0.7))
            if tp <= 0: continue
            big = Y % 10 == 0 or Y in (1958, 1966, 2010, 2026)
            y = self.ty(Y); L = 18 if big else 9
            ink.line([(300 - 4, y), (300 - 4 - L, y)], 1.4 if big else 1.0, prog=tp, taper=False)
        # names in movable type
        sweep_t = lambda x: 23.9 + (x - 600) / 1000 * 0.5
        for ch, path, t_in, t_out, era in self.chars:
            if t < t_in or t > t_out + 0.4: continue
            x = path[0][1]
            for (tk, xk), (tn, xn) in zip(path, path[1:]):
                if t >= tn: x = xn
                else:
                    u = ramp(t, tn - 0.0, tn + 0.6); x = xk + (xn - xk) * (1 - math.exp(-6 * u) * math.cos(4 * u)) if u > 0 else xk
            rough = era * (1 - ease(ramp(t, sweep_t(x), sweep_t(x) + 0.35)))
            ms = self.masks[ch]
            m = ms[1.0] * (rough - 0.55) / 0.45 + ms[0.55] * (1 - (rough - 0.55) / 0.45) if rough > 0.55 else ms[0.55] * rough / 0.55 + ms[0.0] * (1 - rough / 0.55)
            press = ramp(t, t_in, t_in + 0.12); sc = 1.08 - 0.08 * out_cubic(press)
            lift = ramp(t, t_out, t_out + 0.35); a = press * (1 - lift)
            if lift > 0:                                             # dry-brush dissolve as it lifts away
                m = m * (P._grain[:m.shape[0], :m.shape[1]] > 0.8 + 0.2 * lift)
            hh, ww = m.shape
            ink.blit(m, x - ww * sc / 2, 520 - hh * sc / 2 - 14 * lift, alpha=a, scale=sc)
        # the year marker on the stem
        Y = 1958 + 8 * ease(ramp(t, 20.9, 21.6)) + 44 * ease(ramp(t, 24.1, 24.9)) + 16 * ease(ramp(t, 26.4, 27.0))
        if t > 16.4:
            y = self.ty(Y); c = [(300 + 7 * math.cos(q), y + 7 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 26)]
            ink.line(c, 1.8, taper=False)
        wa = ease(ramp(t, 16.0, 17.2))
        if wa > 0:
            cov = (self.wash * wa * (0.75 + 0.25 * P._grain[::2, ::2][:H, :W]))[..., None]
            img[:] = img * (1 - cov) + P.INK * cov
        ink.onto(img, {'ink': P.INK})
        for sl, ts, Yr in self.seals:
            if t < ts: continue
            u = ramp(t, ts, ts + 0.1); P.over(img, sl, 300 + 58, self.ty(Yr), alpha=u, scale=1.25 - 0.25 * u)
        if t > 16.3:                                                 # the top of the timeline: 2026
            if '2026' not in self.digits: self.digits['2026'] = glyph_row('2026', 22, INKC)
            over(img, self.digits['2026'], 300 - 70, self.ty(2026) - 16, ease(ramp(t, 16.3, 16.8)))
        return img


def glyph_row(text, px, color):
    f = font(px, 500); w = int(f.getbbox(text)[2] + px * 0.4)
    im = Image.new('RGBA', (w, int(px * 1.5)), (0, 0, 0, 0)); ImageDraw.Draw(im).text((2, 2), text, font=f, fill=color)
    return np.asarray(im, np.float32) / 255


class Roots:
    """underground on paper: the root system drawn in ochre woodblock line, the years counting with it, then a gold line of
    sap rising to the trunk (37.8: the camera follows it up to the tree)"""
    def __init__(self):
        b = np.fromfile(ROOT / 'data/v14/tree.bin', np.float32).reshape(-1, 12)
        b = b[b[:, 10] < 1.5]                                       # roots only
        self.segs = b
        r = np.random.default_rng(9)
        self.hatch = []
        for k in range(360):
            x, y = r.uniform(0, 1920), 160 + r.random() ** 1.3 * 940
            L = r.uniform(8, 18); a = math.radians(-35 + r.normal(0, 8))
            self.hatch.append(((x, y), (x + L * math.cos(a), y + L * math.sin(a)), r.uniform(0.8, 1.3)))
        self.gold = []
        for k in range(5):                                         # a few strands converging up to the trunk base
            x0 = 960 + r.uniform(-230, 230); y0 = 330 + r.uniform(0, 120)
            self.gold.append([(x0 + (960 - x0) * u ** 1.6 + 14 * math.sin(u * 5 + k), y0 + (150 - y0) * u) for u in np.linspace(0, 1, 40)])
        self.cache = {}

    def P2(self, x, y, z): return 960 + x * 72 - z * 18, 150 - y * 72

    def frame(self, t):
        img = P.paper(0.0); ink = P.Ink()
        ink.line([(x, 150 + 2 * math.sin(x / 50)) for x in range(-20, 1960, 12)], 2.0)
        for a, b, w in self.hatch: ink.line([a, b], w, alpha=0.4)
        T = (t - 30.65) / 0.85                                       # draw time of the roots (0..6)
        lay = ink.layer('ochre')
        for s in self.segs:
            if s[3] > T: continue
            a = self.P2(*s[0:3]); b = self.P2(*s[4:7])
            if s[7] > T: f = (T - s[3]) / max(s[7] - s[3], 1e-6); b = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            w = max(1, int(round(s[8] * 1.7 * P.SS)))
            cv2.line(lay, (int(a[0] * P.SS * 16), int(a[1] * P.SS * 16)), (int(b[0] * P.SS * 16), int(b[1] * P.SS * 16)), float(min(1.0, 0.35 + 0.65 * s[9])), w, cv2.LINE_AA, shift=4)
        for k, g in enumerate(self.gold):
            ink.line(g, 2.6, key='gold', prog=ease(ramp(t, 35.7 + 0.12 * k, 37.4)))
        ink.onto(img, {'ink': P.INK, 'ochre': P.OCHRE, 'gold': P.GOLD})
        Y = int(round(1958 + 68 * ease(ramp(t, 30.9, 35.6))))
        if t > 30.8:
            if Y not in self.cache: self.cache[Y] = glyph_row(str(Y), 58, INKC)
            over(img, self.cache[Y], 1560, 930, ease(ramp(t, 30.8, 31.2)))
        return img


def segB():
    pb, ro = PaperB(), Roots()
    S = [(11.4, 27.4, pb.frame, None, None, None),
         (27.4, 30.6, Clip(ROOT / 'footage/orbit_library', 1 / 30), (960, 540, 1.0), (960, 660, 1.2), dict(exp=0.05, con=1.06, temp=0.2)),
         (30.6, 37.8, ro.frame, None, None, None)]
    labels = [Label('图书馆', 27.9, 30.5, 120, 880)]
    texts = [ink_caption('于是，它开始萌芽、生长', char_times('u03', '于是，它开始萌芽、生长'), 15.4),
             ink_caption('曾经，它叫', char_times('u04', '曾经，它叫'), 20.0),
             ink_caption('后来，它叫', char_times('u05', '后来，它叫'), 23.6),
             ink_caption('今天，它是', char_times('u06', '今天，它是'), 27.2),
             caption('u07', '在这片肥沃的大地上，', 30.45),
             ink_caption('它扎根近70年，', [30.12, 30.6, 30.84, 31.02, 31.2, 31.4, 31.68], 31.95),
             ink_caption('只为积蓄向上生长的力量', char_times('u07', '只为继续向上生长的力量'), 34.9)]
    return 11.4, 37.8, S, labels, texts



# ================================================================== transitions (render2): shots are rendered with handles and
# blended at their boundaries; every join has its own motivated move
_rngT = np.random.default_rng(21)
_N = cv2.resize(cv2.GaussianBlur(_rngT.random((135, 240)).astype(np.float32), (0, 0), 6), (W, H), interpolation=cv2.INTER_CUBIC)
_N += 0.35 * cv2.resize(cv2.GaussianBlur(_rngT.random((270, 480)).astype(np.float32), (0, 0), 3), (W, H), interpolation=cv2.INTER_CUBIC)
_N = (_N - _N.min()) / (_N.max() - _N.min())
_R = np.sqrt(((np.arange(W)[None, :] - W / 2) / (W / 2)) ** 2 + ((np.arange(H)[:, None] - H / 2) / (H / 2)) ** 2).astype(np.float32) / 1.42


def _lum(x): return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
def _ss(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)


def mblur(img, dx, dy):
    """directional motion blur along (dx, dy) pixels"""
    L = int(round(math.hypot(dx, dy)))
    if L < 2: return img
    k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
    cv2.line(k, (L - int(dx / 2), L - int(dy / 2)), (L + int(dx / 2), L + int(dy / 2)), 1.0, 1)
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def shift(img, dx, dy, scale=1.0):
    M = np.float32([[scale, 0, (1 - scale) * W / 2 + dx], [0, scale, (1 - scale) * H / 2 + dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def T_whip(A, B, u, dx=0, dy=-900):
    """A accelerates away along (dx, dy) with blur; on the cut B arrives from the opposite side and settles"""
    if u < 0.5:
        e = (u / 0.5) ** 2.2; v = 2.2 * (u / 0.5) ** 1.2
        return mblur(shift(A, dx * e * 0.35, dy * e * 0.35), dx * v * 0.06, dy * v * 0.06)
    e = 1 - ((u - 0.5) / 0.5); e2 = e ** 2.2; v = 2.2 * e ** 1.2
    return mblur(shift(B, -dx * e2 * 0.35, -dy * e2 * 0.35), dx * v * 0.06, dy * v * 0.06)


def T_zoom(A, B, u):
    """push through: A dives in (scale up, radial blur feel), B arrives from close and settles"""
    if u < 0.5:
        e = (u / 0.5) ** 2.4; s = 1 + 0.6 * e
        x = shift(A, 0, 0, s); return x * (1 - 0.0) if e < 0.6 else cv2.addWeighted(x, 0.65, shift(A, 0, 0, s * 1.06), 0.35, 0)
    e = (1 - (u - 0.5) / 0.5) ** 2.4; s = 1 + 0.35 * e
    x = shift(B, 0, 0, s); return x if e < 0.5 else cv2.addWeighted(x, 0.65, shift(B, 0, 0, s * 1.05), 0.35, 0)


def T_luma(A, B, u):
    """bright first: the whites of B (clouds) grow out of the whites of A (snow)"""
    l = _lum(B) * 0.7 + _N * 0.3
    m = _ss(1 - u * 1.35, 1.15 - u * 1.35, l)[..., None]
    return A * (1 - m) + B * m


def T_ink(A, B, u, fringe=0.10, cx=0.5, cy=0.5):
    """B bleeds in like ink / water on paper from (cx, cy), with a slightly darker wet edge"""
    Rc = _R if (cx, cy) == (0.5, 0.5) else (np.sqrt(((np.arange(W)[None, :] - W * cx) / (W / 2)) ** 2 + ((np.arange(H)[:, None] - H * cy) / (H / 2)) ** 2).astype(np.float32) / 1.42)
    v = 0.40 * cv2.GaussianBlur(_N, (0, 0), 9) / max(cv2.GaussianBlur(_N, (0, 0), 9).max(), 1e-6) + 0.60 * Rc
    th = u * 1.25 - 0.05
    m = _ss(th, th - 0.08, v)
    edge = _ss(0.0, 0.25, m) * _ss(0.6, 0.25, m)
    out = A * (1 - m[..., None]) + B * m[..., None]
    return out * (1 - fringe * edge[..., None])


_PAPER1 = None
def T_print(A, B, u):
    """past -> today: B (a real picture) first PRINTS onto the paper A as ink (its darks only), then its colour floods in from
    the centre and the paper grain fades away"""
    p = np.clip(ramp(u, 0.0, 0.45), 0, 1); c = ramp(u, 0.3, 1.0)
    l = _lum(B)
    printed = A * (1 - (0.85 * ease(p)) * np.clip(1.1 - l * 1.25, 0, 1)[..., None] * (0.85 + 0.15 * _N[..., None]))
    if c <= 0: return printed
    m = _ss(c * 1.3 - 0.15, c * 1.3 - 0.35, 0.5 * _N + 0.5 * _R)[..., None]
    paper_k = 1 - 0.12 * (1 - c)
    return printed * (1 - m) + B * paper_k * m


TRANS = {'whip': T_whip, 'zoom': T_zoom, 'luma': T_luma, 'ink': T_ink, 'print': T_print}


def render2(name, spec):
    t0, t1, S, labels, texts, trans = spec
    out = ROOT / f'renders/v14/{name}.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
    aud = ROOT / 'audio/v3/out/v14vm/stereo/master.wav'
    # handles: each boundary transition (kind, dur, kw) is centred on the boundary
    pre = [0.0] * len(S); post = [0.0] * len(S)
    for i, tr in trans.items():
        d = tr[1] / 2 + 0.04; post[i] = max(post[i], d); pre[i + 1] = max(pre[i + 1], d)
    cache = []
    for i, sp in enumerate(S):
        a, b, clip, k0, k1, g = sp[:6]
        fa = int(round((a - pre[i] - t0) * FPS)); fb = int(round((b + post[i] - t0) * FPS)); n = fb - fa
        fr = {}
        if callable(clip):
            for k in range(n): fr[fa + k] = clip(t0 + (fa + k) / FPS).astype(np.float32)
        else:
            src_shift = pre[i] * clip.speed
            c2 = Clip(clip.src, max(0.0, clip.t0 - src_shift), clip.speed, getattr(clip, 'fps_src', None), clip.crop) if not clip.files else clip
            if clip.files:
                c2 = Clip(clip.src, max(0.0, clip.t0 - src_shift), clip.speed, clip.fps_src)
            raw = c2.frames(n)
            na = int(round((b - a) * FPS))
            for k, f in enumerate(raw):
                u = ease(min(max((k - int(round(pre[i] * FPS))) / max(na - 1, 1), 0), 1))
                cx, cy, z = [k0[j] + (k1[j] - k0[j]) * u for j in range(3)]
                fr[fa + k] = grade(reframe(f, cx, cy, z), **g) * BAND
        cache.append(fr); print(f'shot {i} {a:.2f}-{b:.2f} +handles {len(fr)} frames', flush=True)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-ss', str(t0), '-t', str(t1 - t0), '-i', str(aud), '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '256k', '-shortest', str(out)], stdin=subprocess.PIPE)
    N = int(round((t1 - t0) * FPS))
    for f in range(N):
        t = t0 + f / FPS
        i = max(k for k in range(len(S)) if int(round((S[k][0] - t0) * FPS)) <= f or k == 0)
        img = cache[i].get(f)
        for j, tr in trans.items():                        # inside a transition window around boundary j|j+1 ?
            bt = S[j][1]; d = tr[1]
            if bt - d / 2 <= t < bt + d / 2:
                A = cache[j].get(f); B = cache[j + 1].get(f)
                if A is not None and B is not None:
                    img = TRANS[tr[0]](A, B, (t - (bt - d / 2)) / d, **(tr[2] if len(tr) > 2 else {}))
        if img is None:
            near = [k for k in cache[i] if abs(k - f) <= 3]
            print('missing frame', f, 'shot', i, 'range', min(cache[i]), max(cache[i]), flush=True)
            img = cache[i][min(cache[i], key=lambda k: abs(k - f))]
        img = np.array(img, np.float32, copy=True)
        for L in labels: L.draw(img, t)
        for T in texts: T.draw(img, t)
        enc.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait(); print('->', out, N, 'frames')


def segA2():
    t0, t1, S, labels, texts = segA()
    trans = {0: ('zoom', 0.5), 5: ('luma', 0.8), 6: ('whip', 0.5, dict(dx=0, dy=-1000)), 7: ('ink', 1.1)}
    return t0, t1, S, labels, texts, trans


def segB2():
    t0, t1, S, labels, texts = segB()
    trans = {0: ('print', 1.3), 1: ('whip', 0.5, dict(dx=0, dy=-1100))}
    return t0, t1, S, labels, texts, trans


if __name__ == '__main__':
    n = sys.argv[1]
    if n.endswith('2'): render2(n, {'segA2': segA2, 'segB2': segB2}[n]())
    else: render(n)
