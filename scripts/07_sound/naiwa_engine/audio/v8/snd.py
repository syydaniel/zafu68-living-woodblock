"""Sample handling for the sound design (audio/v8): library files by NAME straight from the Final Cut Pro Sound
Effects / Fairlight Sound Library (decoded on demand by ffmpeg, cached in audio/v8/cache as 48 kHz float wav, with
their own channel layout kept: stereo beds stay stereo), active segments of multi-take files, onsets, level helpers,
tape speed, tones, a compressor, a look-ahead limiter. Run with ~/venvs/audio/bin/python (numpy, scipy, soundfile)."""
import os, json, subprocess, hashlib
from fractions import Fraction
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from eng import SR, V8, db, lpf, hpf, bpf, shelf_hi, fade, act_db, peak_db, fold6

CACHE = f"{V8}/cache"; os.makedirs(CACHE, exist_ok=True)
_meta = None; _mem = {}
rng = np.random.default_rng(20261012)


def meta():
    global _meta
    if _meta is None:
        _meta = {}
        for m in json.load(open(f"{V8}/lib_index.json")):
            _meta.setdefault(m["name"], m)
    return _meta


def path_of(name):
    m = meta().get(name)
    assert m, "no library file named " + name
    return m["path"]


def raw(name):
    """the whole library file, [n, ch] float32 at 48 kHz (ch as in the file, at most 2: surround files give L R)"""
    if name not in _mem:
        p = path_of(name); c = f"{CACHE}/{hashlib.md5(p.encode()).hexdigest()[:10]}_{name.replace('/', '_')[:60]}.wav"
        if not os.path.exists(c):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-ar", str(SR), "-c:a", "pcm_f32le", c], check=True)   # channels as in the file (6 = L R C LFE Ls Rs)
        x, sr = sf.read(c, dtype="float32", always_2d=True); assert sr == SR
        _mem[name] = x
    return _mem[name]


def S(name, a=0.0, b=None, mono=True, surround=False):
    """seconds a..b of a library file: mono [n] (default), stereo [n, 2] (mono=False) or, for a 5.1 recording with
    surround=True, the recording itself [n, 6]. The "LFE ..." files only have an LFE channel: they come back as mono,
    or as the same signal on both sides."""
    x = raw(name); i = int(a * SR); j = len(x) if b is None else int(b * SR); x = x[i:j]
    if x.shape[1] == 6:
        fronts = np.abs(x[: SR * 5, :2]).max() > 1e-5
        if mono or not fronts:
            m = fold6(x); return m if mono else (np.stack([m, m], 1) * 0.707).astype(np.float32)
        return x.astype(np.float32).copy() if surround else x[:, :2].astype(np.float32).copy()
    if mono:
        return x.mean(1).astype(np.float32)
    return (np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]).astype(np.float32).copy()


def is_stereo(name):
    x = raw(name)
    if x.shape[1] < 2: return False
    n = min(len(x), SR * 5); return not np.allclose(x[:n, 0], x[:n, 1], atol=1e-5)


def segs(name, thr=-38.0, gap=0.12, minlen=0.04, maxlen=None, pad=0.012):
    """the separate takes inside a library file: [(a, b), ...] seconds (level above thr dB relative to the file's peak)"""
    m = np.abs(raw(name)).max(1); h = SR // 200; e = np.array([m[i:i + h].max() for i in range(0, len(m) - h, h)]); on = e > (m.max() * db(thr))
    out = []; i = 0; g = int(gap * 200)
    while i < len(on):
        if on[i]:
            j = i; last = i
            while j < len(on) and (on[j] or j - last <= g):
                if on[j]: last = j
                j += 1
            a = max(0.0, i / 200 - pad); b = min(len(m) / SR, (last + 1) / 200 + pad + 0.03)
            if b - a >= minlen:
                out.append((a, b if maxlen is None else min(b, a + maxlen)))
            i = j
        else:
            i += 1
    return out


def onset(x, thr=0.25):
    """seconds from the clip start to its attack (first sample above thr x its peak)"""
    m = np.abs(x if x.ndim == 1 else x.mean(1)); return float(np.argmax(m > thr * m.max())) / SR


def speed(x, r):
    """r > 1: faster and higher (tape speed)"""
    if abs(r - 1) < 1e-3: return x
    fr = Fraction(1.0 / r).limit_denominator(96); return resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)


def to_act(x, level): return (x * db(level - act_db(x))).astype(np.float32)
def to_peak(x, level): return (x * db(level) / (np.abs(x).max() + 1e-9)).astype(np.float32)


def tame(x, ceil_db):
    """soft ceiling for beds: clicks above 0.7 x the ceiling are bent under it"""
    C = db(ceil_db); T = 0.7 * C; a = np.abs(x); over = a > T
    if not over.any(): return x
    y = x.copy(); y[over] = np.sign(x[over]) * (T + (C - T) * np.tanh((a[over] - T) / (C - T))); return y


def loop_to(x, n, xf=1.5):
    """a bed of n samples from a shorter recording: repeated with long equal-power crossfades, starting somewhere inside"""
    if len(x) >= n: return x[:n].copy()
    k = min(int(xf * SR), len(x) // 3); out = x.copy()
    w = np.sin(np.linspace(0, np.pi / 2, k)).astype(np.float32); w = w if x.ndim == 1 else w[:, None]
    while len(out) < n:
        out = np.concatenate([out[:-k], out[-k:] * w[::-1] + x[:k] * w, x[k:]])
    return out[:n]


def env(t0, n, pts):
    """breakpoint gain curve (dB) over a clip that starts at film time t0: pts = [(t, dB), ...] -> linear gain [n]"""
    t = t0 + np.arange(n) / SR; return db(np.interp(t, [p[0] for p in pts], [p[1] for p in pts])).astype(np.float32)


def tone(f, dur, decay=25.0, harm=((1, 1.0), (2, 0.25), (3, 0.08))):
    """a small clean interface tone (decaying partials)"""
    t = np.arange(int(dur * SR)) / SR; x = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in harm) * np.exp(-decay * t)
    return fade(x.astype(np.float32), 0.002, 0.01)


def comp(x, thr=-22.0, ratio=2.5, att=0.008, rel=0.12, makeup=0.0):
    """feed-forward compressor on the mono sum (RMS detector, soft knee 6 dB)"""
    m = x if x.ndim == 1 else x.mean(1); e = np.sqrt(np.maximum(1e-12, np.convolve(m ** 2, np.ones(480) / 480, "same"))); L = 20 * np.log10(e + 1e-9)
    over = L - thr; k = 6.0; gr = np.where(over <= -k / 2, 0.0, np.where(over >= k / 2, over * (1 - 1 / ratio), (over + k / 2) ** 2 / (2 * k) * (1 - 1 / ratio)))
    g = np.zeros_like(gr); a = np.exp(-1 / (att * SR)); r = np.exp(-1 / (rel * SR)); s = 0.0; h = 16
    for i in range(0, len(gr), h):                                       # block-wise smoothing (attack / release)
        v = gr[i:i + h].max(); c = a ** h if v > s else r ** h; s = c * s + (1 - c) * v; g[i:i + h] = s
    y = x * db(-g + makeup)[(slice(None),) + (None,) * (x.ndim - 1)]; return y.astype(np.float32)


def limiter(x, ceil_db=-1.0, look=0.004, rel=0.15, ret_gain=False):
    """look-ahead peak limiter, all channels together (4x oversampled peak estimate). The gain is computed per 1 ms
    block (instant attack inside the look-ahead window, exponential release), interpolated, and never above what the
    peaks need. Returns (y, deepest gain reduction in dB)."""
    from scipy.ndimage import minimum_filter1d
    up = resample_poly(x, 4, 1, axis=0); pk = np.abs(up).max(1) if up.ndim == 2 else np.abs(up); pk = pk[: len(x) * 4].reshape(-1, 4).max(1)
    C = db(ceil_db); need = np.minimum(1.0, C / np.maximum(pk, 1e-9))
    if need.min() >= 1.0:
        return (x.astype(np.float32), 0.0, np.ones(len(x), np.float32)) if ret_gain else (x.astype(np.float32), 0.0)
    B = 48; nb = len(need) // B + 1; pad = np.ones(nb * B, np.float64); pad[: len(need)] = need; blk = pad.reshape(nb, B).min(1)
    k = int(np.ceil(look * SR / B)); blk = minimum_filter1d(blk, size=2 * k + 1, mode="nearest"); r = np.exp(-B / (rel * SR)); g = np.empty(nb); s = 1.0
    for i in range(nb):
        s = blk[i] if blk[i] < s else 1.0 - (1.0 - s) * r; g[i] = s
    gs = np.interp(np.arange(len(need)), (np.arange(nb) + 0.5) * B, g); n = int(look * SR)
    gs = np.minimum(gs, minimum_filter1d(need, size=2 * n + 1, mode="nearest"))
    w = np.hanning(2 * (n // 2) + 1); w /= w.sum(); h_ = len(w) // 2; gs = np.minimum(np.convolve(np.pad(gs, (h_, h_), mode="edge"), w, "valid"), minimum_filter1d(need, size=n + 1, mode="nearest"))      # (edge padding: with mode "same" the smoothing would halve the gain in the first and last 2 ms and report a false reduction)
    y = (x * gs[(slice(None),) + (None,) * (x.ndim - 1)]).astype(np.float32)
    return (y, float(20 * np.log10(gs.min())), gs.astype(np.float32)) if ret_gain else (y, float(20 * np.log10(gs.min())))
