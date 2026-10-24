"""NAIWA BEYOND sound engine (audio/v8): object based. Every sound is an OBJECT (a mono clip, a start time, a level
at the listener, a direction that may move, a distance, a reverb send into the scene's ROOM); beds are stereo objects
with a width. One object list is rendered to several LAYOUTS with the same code:
    "stereo"  L R                       loudspeaker safe: constant-power amplitude panning, no inter-aural delays
    "5.1"     L R C LFE Ls Rs           pairwise constant-power panning round the five speakers, LFE from flagged objects
    "bin"     L R                       headphones: every object at its own place in 3D through Apple's HRTF (_render_bin)
and the 5.1 result can be turned into a headphone (binaural) mix by Apple's HRTF renderer (spat/spat: six virtual speakers).
Rooms are REAL impulse responses (/Library/Audio/Impulse Responses/Apple): the B-format ones (W X Y Z) are decoded to the
layout, so the reverb surrounds the listener in 5.1; stereo ones feed the fronts (and, delayed and darker, the rears).
Stems: DX (dialogue), MX (music), FX (effects + foley), AMB (beds). Everything 48 kHz float32.
Run with ~/venvs/audio/bin/python (numpy, scipy, soundfile, pyloudnorm)."""
import os, json, subprocess, math, zlib
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, resample_poly

R = "~/Movies/NAIWA_BEYOND"; V8 = f"{R}/audio/v8"; SR = 48000
IRDIR = "/Library/Audio/Impulse Responses/Apple"
LAYOUTS = {"stereo": ["L", "R"], "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"], "bin": ["L", "R"]}       # bin: every object through the HRTF, in 3D
SPK = {"L": -30.0, "R": 30.0, "C": 0.0, "Ls": -110.0, "Rs": 110.0}                 # azimuth, + = right
STEMS = ("DX", "MX", "FX", "AMB")


def db(v): return 10.0 ** (v / 20.0)
def lpf(x, f, o=2): return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x, axis=0).astype(np.float32)
def hpf(x, f, o=2): return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x, axis=0).astype(np.float32)
def bpf(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], 'band', fs=SR, output='sos'), x, axis=0).astype(np.float32)
def shelf_hi(x, f, g):
    """first-order high shelf (g dB above f)"""
    lo = lpf(x, f, 1); return (lo + (x - lo) * db(g)).astype(np.float32)


def fade(x, fi=0.005, fo=0.05):
    x = np.array(x, np.float32, copy=True); n = len(x); a = min(int(fi * SR), n // 2); b = min(int(fo * SR), n // 2)
    sh = (slice(None),) + (None,) * (x.ndim - 1)
    if a > 0: x[:a] *= (np.sin(np.linspace(0, np.pi / 2, a)) ** 2).astype(np.float32)[sh]
    if b > 0: x[-b:] *= (np.cos(np.linspace(0, np.pi / 2, b)) ** 2).astype(np.float32)[sh]
    return x


def act_db(x):
    """level of the sound itself: RMS of the loud half of its 50 ms windows (pauses ignored)"""
    m = x if x.ndim == 1 else x.mean(1); h = SR // 20
    if len(m) < 2 * h:
        return 20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-9)
    e = np.array([(m[i:i + h] ** 2).mean() for i in range(0, len(m) - h, h)]); e = np.sort(e)[len(e) // 2:]
    return 10 * np.log10(e.mean() + 1e-12)


def peak_db(x): return 20 * np.log10(np.abs(x).max() + 1e-9)


def fold6(x):
    """a 5.1 recording (L R C LFE Ls Rs) as one mono signal (for measuring it, or for using it as a point source)"""
    return (0.5 * (x[:, 0] + x[:, 1]) + 0.35 * x[:, 2] + 0.5 * x[:, 3] + 0.35 * (x[:, 4] + x[:, 5])).astype(np.float32)


def lev(x):
    """active level of any clip (mono, stereo or 5.1)"""
    return act_db(fold6(x)) if (x.ndim == 2 and x.shape[1] == 6) else act_db(x)


# ---------------------------------------------------------------------------------------------- impulse responses
_ir = {}
def ir(name):
    """an Apple impulse response by (part of) its file name -> (array [n, ch], kind 'bf' (W X Y Z) / 'st' / 'mono')"""
    if name not in _ir:
        hit = None
        for dp, dn, fn in os.walk(IRDIR):
            for f in fn:
                if f.endswith(".SDIR") and name.lower() in f.lower():
                    hit = os.path.join(dp, f); break
            if hit: break
        assert hit, "no impulse response named " + name
        ch = int(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=channels", "-of", "csv=p=0", hit], capture_output=True, text=True).stdout.strip())
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", hit, "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
        x = np.frombuffer(raw, np.float32).reshape(-1, ch).copy()
        i0 = max(0, int(np.argmax(np.abs(x[:, 0]) > 0.05 * np.abs(x[:, 0]).max())) - 24); x = x[i0:]          # trim the lead-in
        x /= np.sqrt((x[:, 0] ** 2).sum()) + 1e-9                                                             # unit energy (W / left)
        _ir[name] = (x, "bf" if ch == 4 else ("st" if ch == 2 else "mono"))
    return _ir[name]


def room_decode(send, name, layout, pre=0.0, lp=None, hp=120.0, width=1.0):
    """a mono reverb send through a room -> the layout's channels (wet only). B-format: virtual cardioid microphones
    toward each speaker (stereo: +-55 deg; 5.1: the four corners, nothing in the centre). Stereo IR: fronts, and for
    5.1 the rears get the other channel 18 ms later and darker (a different, later part of the same room)."""
    h, kind = ir(name); chs = LAYOUTS[layout]; n = len(send); out = np.zeros((n + len(h) - 1, len(chs)), np.float32)
    s = hpf(send, hp) if hp else send
    if kind == "bf":
        W, X, Y = h[:, 0], h[:, 1], h[:, 2]                                # FuMa: X front, Y left
        angs = {"L": -55.0, "R": 55.0} if layout == "stereo" else {"L": -45.0, "R": 45.0, "Ls": -125.0, "Rs": 125.0}
        for c, a in angs.items():
            th = math.radians(a); v = 0.5 * (math.sqrt(2) * W + width * (X * math.cos(th) - Y * math.sin(th)))
            out[:, chs.index(c)] = fftconvolve(s, v)[: len(out)]
    else:
        hl = h[:, 0]; hr = h[:, 1] if h.shape[1] > 1 else h[:, 0]
        out[:, chs.index("L")] = fftconvolve(s, hl)[: len(out)]; out[:, chs.index("R")] = fftconvolve(s, hr)[: len(out)]
        if layout == "5.1":
            d = int(0.018 * SR)
            for c, hh in (("Ls", hr), ("Rs", hl)):
                v = lpf(fftconvolve(s, hh)[: len(out)], 5000) * db(-2.0); out[d:, chs.index(c)] = v[:-d]
    if lp: out = lpf(out, lp)
    if pre > 0:
        d = int(pre * SR); out = np.concatenate([np.zeros((d, out.shape[1]), np.float32), out])
    return out


# ---------------------------------------------------------------------------------------------- panning
def _gains(az, layout):
    """constant-power gains for azimuth arrays (deg, + right; 180 = behind) -> [n, channels] (no C for point sources
    unless centre=True is asked by the caller; LFE never)"""
    az = (np.asarray(az, np.float64) + 180.0) % 360.0 - 180.0; chs = LAYOUTS[layout]; g = np.zeros((len(az), len(chs)), np.float32)
    if layout == "stereo":
        fold = np.where(np.abs(az) > 90.0, np.sign(az) * (180.0 - np.abs(az)), az)       # behind folds to the same side
        p = np.clip(fold / 60.0, -1.0, 1.0)                                               # +-60 deg and beyond = hard left / right
        th = (p + 1.0) * np.pi / 4.0; g[:, 0] = np.cos(th); g[:, 1] = np.sin(th)
        back = np.clip((np.abs(az) - 90.0) / 90.0, 0.0, 1.0); g *= (1.0 - 0.16 * back)[:, None].astype(np.float32)   # behind: 1.5 dB down
        return g
    ring = [("Ls", -110.0), ("L", -30.0), ("C", 0.0), ("R", 30.0), ("Rs", 110.0)]
    for k in range(len(ring)):
        (c0, a0), (c1, a1) = ring[k], ring[(k + 1) % len(ring)]
        if k == len(ring) - 1:                                                           # Rs -> Ls across the back (110 .. 250)
            a = np.where(az < 0, az + 360.0, az); a1 = 250.0
        else:
            a = az
        m = (a >= a0) & (a <= a1)
        if not m.any(): continue
        u = (a[m] - a0) / (a1 - a0) * np.pi / 2
        g[m, chs.index(c0)] = np.cos(u); g[m, chs.index(c1)] = np.sin(u)
    return g


def _interp_path(path, t0, n):
    """path = [(t (s, relative to the clip start), az, dist), ...] -> per-sample az, dist"""
    if not isinstance(path, (list, tuple)) or not isinstance(path[0], (list, tuple)):
        az, dist = path if isinstance(path, (list, tuple)) else (path, 2.0); return np.full(n, float(az)), np.full(n, float(dist))
    tt = np.array([p[0] for p in path], np.float64); t = np.arange(n) / SR
    azs = np.unwrap(np.radians([p[1] for p in path])); az = np.degrees(np.interp(t, tt, azs))
    return az, np.interp(t, tt, [p[2] for p in path])


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR); self.objs = []; self.rooms = []        # rooms: (t0, t1, key, ir name, wet dB, kwargs)

    # ---- scene acoustics
    def room(self, key, t0, t1, irname, wet=-12.0, **kw):
        self.rooms.append(dict(key=key, t0=t0, t1=t1, ir=irname, wet=wet, kw=kw))

    def room_at(self, t):
        for r in self.rooms:
            if r["t0"] <= t < r["t1"]: return r["key"]
        return None

    # ---- objects
    def add(self, clip, t, stem="FX", az=0.0, dist=2.0, rv=-18.0, room="auto", name="", lfe=None, centre=0.0, width=None, note="", doppler=False, only=None, el=0.0):
        """clip: mono [n] (a point source) or stereo [n, 2] (a bed: 'width' = half angle of its image, default 60;
        az turns it). Level is the level AT THE LISTENER (prepare the clip at that level). path: az / dist constants or
        az = [(t, az, dist), ...]. rv = reverb send in dB relative to the direct sound (None = dry).
        centre (5.1 only): share of a front source that goes to C instead of the phantom centre (dialogue 1.0).
        lfe: dB of the object's low band (< 90 Hz) also sent to LFE in 5.1.
        el ("bin" only): elevation in degrees (+ = up); a path may carry it as a 4th value: (t, az, dist, el). Beds may have an az path too
        (the pair turns together: an orbit)."""
        clip = np.asarray(clip, np.float32)
        self.objs.append(dict(clip=clip, t=float(t), stem=stem, az=az, dist=dist, rv=rv, room=(self.room_at(t + min(0.05, len(clip) / SR / 2)) if room == "auto" else room),
                              name=name, lfe=lfe, centre=centre, width=width, note=note, doppler=doppler, only=only, el=el))
        return self.objs[-1]

    # ---- rendering
    def render(self, layout="stereo", stems=STEMS, verbose=False):
        if layout == "bin":
            return self._render_bin(stems, verbose)
        chs = LAYOUTS[layout]; nc = len(chs); out = {s: np.zeros((self.n + SR * 12, nc), np.float32) for s in stems}
        sends = {}                                                        # (room key, stem) -> mono send bus
        for o in self.objs:
            if o["stem"] not in out or (o["only"] and o["only"] != layout): continue
            x = o["clip"]; i = int(round(o["t"] * SR)); n = len(x)
            if i < 0: x = x[-i:]; n = len(x); i = 0
            if i + n > self.n + SR * 10: n = self.n + SR * 10 - i; x = x[:n]
            if n <= 0: continue
            if x.ndim == 2 and x.shape[1] == 6:                           # a 5.1 recording (L R C LFE Ls Rs): as it is / folded down
                if layout == "5.1":                                       # (the LFE channel plays 10 dB up on a real system: hold it back)
                    out[o["stem"]][i:i + n] += x * np.array([1, 1, 0.6, 0.4, 1, 1], np.float32)[None, :]
                    out[o["stem"]][i:i + n, 0] += 0.35 * x[:, 3]; out[o["stem"]][i:i + n, 1] += 0.35 * x[:, 3]   # and keep its body in the mains
                else:
                    out[o["stem"]][i:i + n, 0] += x[:, 0] + 0.5 * x[:, 2] + 0.707 * x[:, 4] + 0.5 * x[:, 3]
                    out[o["stem"]][i:i + n, 1] += x[:, 1] + 0.5 * x[:, 2] + 0.707 * x[:, 5] + 0.5 * x[:, 3]
                m = x[:, [0, 1, 4, 5]].mean(1)
            elif x.ndim == 2:                                             # a stereo bed: two sources at az -+ width
                w = 60.0 if o["width"] is None else float(o["width"]); az0 = float(o["az"]) if not isinstance(o["az"], (list, tuple)) else float(o["az"][0][1])
                for k, a in ((0, az0 - w), (1, az0 + w)):
                    g = _gains(np.array([a]), layout)[0]
                    if layout == "5.1" and abs(a) < 45:                   # keep beds out of the centre speaker
                        gl = _gains(np.array([-30.0 if a < 0 else 30.0]), layout)[0]; u = abs(a) / 30.0 if abs(a) < 30 else 1.0
                        g = gl * math.sqrt(0.5 + 0.5 * min(1.0, u)) + _gains(np.array([30.0 if a < 0 else -30.0]), layout)[0] * math.sqrt(max(0.0, 0.5 - 0.5 * min(1.0, u)))
                    out[o["stem"]][i:i + n] += x[:, k:k + 1] * g[None, :]
                m = x.mean(1)
                if layout == "5.1" and o["lfe"] is not None:
                    out[o["stem"]][i:i + n, chs.index("LFE")] += lpf(m, 90, 4) * db(o["lfe"])
            else:
                az, dist = _interp_path(o["az"] if isinstance(o["az"], (list, tuple)) and isinstance(o["az"][0], (list, tuple)) else (o["az"], o["dist"]), o["t"], n)
                if o["doppler"] and isinstance(o["az"], (list, tuple)):  # delay line by distance: pitch glides as it passes
                    dly = dist / 343.0 * SR; pos = np.arange(n) - (dly - dly[0]); pos = np.clip(pos, 0, n - 2); j = pos.astype(np.int64); f = (pos - j).astype(np.float32)
                    x = (x[j] * (1 - f) + x[j + 1] * f).astype(np.float32)
                g = _gains(az, layout)
                if layout == "5.1" and o["centre"] > 0:                   # dialogue: into the real centre
                    c = float(o["centre"]); gc = np.zeros_like(g); front = (np.abs(az) < 30.0)
                    gc[:, chs.index("C")] = 1.0; g = np.where(front[:, None], g * (1 - c) + gc * c, g)
                elif layout == "5.1":                                     # effects: phantom centre shared 50 / 50 with C
                    pass
                out[o["stem"]][i:i + n] += x[:, None] * g
                if layout == "5.1" and o["lfe"] is not None:
                    out[o["stem"]][i:i + n, chs.index("LFE")] += lpf(x, 90, 4) * db(o["lfe"])
                m = x
            if o["rv"] is not None and o["room"]:
                k = (o["room"], o["stem"])
                if k not in sends: sends[k] = np.zeros(self.n + SR * 12, np.float32)
                sends[k][i:i + n] += m * db(o["rv"])
        for (rk, stem), s in sends.items():
            r = next(r for r in self.rooms if r["key"] == rk)
            if not np.any(s): continue
            wet = room_decode(s, r["ir"], layout, **r["kw"]) * db(r["wet"]); m = min(len(wet), len(out[stem])); out[stem][:m] += wet[:m]
            if verbose: print("  room", rk, stem, r["ir"], "wet peak %.1f dB" % peak_db(wet), flush=True)
        return {s: v[: self.n] for s, v in out.items()}



# ---------------------------------------------------------------------------------------------- binaural objects
SPAT = f"{V8}/spat/spat"
BIN_DIRS = [(float(a), 0.0) for a in range(0, 360, 45)] + [(a, 45.0) for a in (45.0, 135.0, 225.0, 315.0)]   # the room all round and above
BIN_WET = float(os.environ.get("BIN_WET", "0.0"))                        # dB on the decoded rooms (calibrated against the 5.1 decode)


def _xyz(az, el, d):
    a, e = math.radians(az), math.radians(el); return [d * math.cos(e) * math.sin(a), d * math.sin(e), -d * math.cos(e) * math.cos(a)]


def _keys(az, dist, el, n, step=0.04):
    """an object's place as keyframes [(t from its start, az, dist, el)]: one key when it stands still, every `step` s along a path
    (spat moves in straight lines between keys: dense keys keep a circle a circle)"""
    if isinstance(az, (list, tuple)) and len(az) and isinstance(az[0], (list, tuple)):
        tt = np.array([p[0] for p in az], np.float64); azs = np.degrees(np.unwrap(np.radians([p[1] for p in az])))
        ds = np.array([p[2] for p in az], np.float64); es = np.array([(p[3] if len(p) > 3 else el) for p in az], np.float64)
        T = np.arange(0.0, n / SR + step, step)
        return [(float(t), float(np.interp(t, tt, azs)), max(0.3, float(np.interp(t, tt, ds))), float(np.interp(t, tt, es))) for t in T]
    return [(0.0, float(az), max(0.3, float(dist)), float(el))]


def _per_sample(keys, n):
    if len(keys) == 1: return np.full(n, keys[0][1]), np.full(n, keys[0][2])
    t = np.arange(n) / SR; tt = [k[0] for k in keys]
    return np.interp(t, tt, [k[1] for k in keys]), np.interp(t, tt, [k[2] for k in keys])


def room_bin(send, name, pre=0.0, lp=None, hp=120.0, width=1.0):
    """a mono reverb send through a room -> [(az, el, wet)] for virtual sources all round and above (BIN_DIRS). B-format: a
    virtual supercardioid toward each direction; stereo IR: fronts at +-45, behind at +-135 18 ms later and darker (as 5.1)"""
    h, kind = ir(name); s = hpf(send, hp) if hp else send; out = []
    if kind == "bf":
        W, X, Y, Z = h[:, 0], h[:, 1], h[:, 2], (h[:, 3] if h.shape[1] > 3 else 0.0 * h[:, 0]); g = 0.46                  # FuMa: X front, Y left, Z up
        for a, e in BIN_DIRS:
            th, ph = math.radians(a), math.radians(e)
            v = 0.37 * math.sqrt(2) * W + 0.63 * width * (X * math.cos(th) * math.cos(ph) - Y * math.sin(th) * math.cos(ph) + Z * math.sin(ph))
            out.append((a, e, fftconvolve(s, v).astype(np.float32) * g))
    else:
        hl = h[:, 0]; hr = h[:, 1] if h.shape[1] > 1 else h[:, 0]; L = fftconvolve(s, hl).astype(np.float32); Rr = fftconvolve(s, hr).astype(np.float32); d = int(0.018 * SR)
        def late(x): y = np.zeros_like(x); y[d:] = (lpf(x, 5000) * db(-2.0))[:-d]; return y
        out = [(-45.0, 0.0, L), (45.0, 0.0, Rr), (-135.0, 0.0, late(Rr)), (135.0, 0.0, late(L))]
    res = []
    for a, e, y in out:
        if lp: y = lpf(y, lp)
        if pre > 0: y = np.concatenate([np.zeros(int(pre * SR), np.float32), y])
        res.append((a, e, (y * db(BIN_WET)).astype(np.float32)))
    return res


# ---- for a clear front and back and a strong sense of being inside the scene (behind the head too): three aids to
# the HRTF, as listening research describes them: (1) Blauert's directional bands made a little stronger than one generic head gives
# them (in front: presence at 3-4 kHz; behind: the band near 1 kHz up, the top down; above: the band near 8 kHz up); (2) early
# reflections of every placed sound, from the floor, the walls, BEHIND (the room the scene is in: a cabin, a forest, a gorge...): they
# take the sound out of the head and give the space behind; (3) more of the room and of the beds behind.
BIN_EQ = float(os.environ.get("BIN_EQ", "1.6"))                          # 0 = the plain HRTF; 1.6 = the directional bands clearly stronger
BIN_REFL = float(os.environ.get("BIN_REFL", "0.0"))                      # dB on all early reflections (-99 = none)
BIN_REAR = float(os.environ.get("BIN_REAR", "3.0"))                      # dB on the beds' second stretch behind (5.1 "rear" copies)
ROOM_CLASS = {"Car Interior": "small", "Tight Garage": "room", "House Room": "room", "Old Forest Long": "forest", "Valley Forest": "valley",
              "Deep Canon": "canyon", "On the Field Natural": "open", "Seashore": "open", "Big Cave One": "cave", "Hangar One": None, "Michaelis Chapel": None,
              "Opulent Ballroom": "hall"}
# reflection: (how its direction follows the source, ms after the direct sound, dB, low-pass Hz)
#   ("floor", el) same az below; ("ceil", el) same az above; "side" = the source mirrored left <-> right; "back" = mirrored front <-> back;
#   ("rel", daz, el) = turned by daz
REFL = {"small": [(("floor", -40.0), 2.5, -8.0, 8000), (("ceil", 55.0), 3.5, -10.0, 8000), ("side", 4.0, -9.0, 8000), ("back", 6.0, -8.0, 8000)],
        "room": [(("floor", -40.0), 3.5, -8.0, 7000), (("ceil", 55.0), 5.0, -10.0, 7000), ("side", 7.0, -10.0, 7000), ("back", 11.0, -9.0, 7000)],
        "forest": [(("floor", -35.0), 4.0, -7.0, 5000), (("rel", 70.0, 0.0), 17.0, -15.0, 4000), (("rel", -80.0, 5.0), 23.0, -15.0, 4000), ("back", 31.0, -13.0, 4000),
                   (("rel", 160.0, 10.0), 44.0, -16.0, 3500)],
        "valley": [(("floor", -35.0), 4.0, -7.0, 5000), ("back", 45.0, -15.0, 3500), (("rel", 100.0, 3.0), 70.0, -17.0, 3000), (("rel", -110.0, 3.0), 95.0, -18.0, 3000)],
        "canyon": [(("floor", -35.0), 3.0, -8.0, 5000), (("abs", -90.0, 5.0), 18.0, -11.0, 5000), (("abs", 90.0, 5.0), 26.0, -11.0, 5000), ("back", 52.0, -13.0, 4500),
                   (("abs", -80.0, 20.0), 75.0, -15.0, 4000)],
        "open": [(("floor", -35.0), 4.0, -6.0, 5000), ("back", 60.0, -20.0, 3000)],
        "cave": [(("floor", -40.0), 5.0, -8.0, 4000), ("side", 12.0, -9.0, 4000), ("back", 20.0, -8.0, 4000), (("ceil", 60.0), 9.0, -9.0, 4000)],
        "hall": [(("floor", -35.0), 6.0, -9.0, 6000), ("side", 18.0, -11.0, 6000), ("back", 30.0, -10.0, 5500), (("ceil", 50.0), 22.0, -13.0, 5500)]}

BIN_UP = float(os.environ.get("BIN_UP", "-5.0"))                          # full envelopment: every bed also above (dB), decorrelated
BIN_LOW = float(os.environ.get("BIN_LOW", "-6.0"))                        # and the ground's beds also below
GROUND = ("water", "lake", "river", "pond", "marsh", "grass", "stream", "creek", "rain", "lagoon", "shore", "waves", "fall", "drip")
UPLESS = ("water", "lake", "river", "pond", "marsh", "stream", "creek", "lagoon", "shore", "waves", "grass")   # (no water overhead; rain is both)


def _decor(x, seed, n=2048):
    """a random-phase all-pass (43 ms): the same sound, its fine structure made different, so a copy above does not comb with the original"""
    rng = np.random.default_rng(seed); ph = rng.uniform(-np.pi, np.pi, n // 2 + 1); ph[0] = 0.0; ph[-1] = 0.0
    h = np.fft.irfft(np.exp(1j * ph), n); h /= np.sqrt((h ** 2).sum()); return fftconvolve(x, h)[: len(x)].astype(np.float32)


def _refl_keys(rule, keys):
    out = []
    for kt, a, d, e in keys:
        if isinstance(rule, tuple) and rule[0] == "floor": out.append((kt, a, d * 1.2, rule[1]))
        elif isinstance(rule, tuple) and rule[0] == "ceil": out.append((kt, a, d * 1.2, rule[1]))
        elif isinstance(rule, tuple) and rule[0] == "rel": out.append((kt, a + rule[1], d * 2.0, rule[2]))
        elif isinstance(rule, tuple) and rule[0] == "abs": out.append((kt, rule[1], d * 2.0, rule[2]))
        elif rule == "side": out.append((kt, -a, d * 1.6, e * 0.5))
        elif rule == "back": out.append((kt, 180.0 - a, d * 1.8, e * 0.5))
    return out


def _band(f, f0, oct_sd):
    return np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / f0) / oct_sd) ** 2)


def _shelf(f, fc):
    return 1.0 / (1.0 + np.exp(-np.log2(np.maximum(f, 1.0) / fc) / 0.3))


def _dir_eq(x, keys, k=None):
    """Blauert's directional bands (narrow-band noise is heard in front at ~450 Hz and ~4 kHz, behind at ~1.1 kHz and ~11 kHz, above at
    ~8 kHz), made stronger than one generic head gives them; below: the pinna notch near 6.5 kHz. Each weighted by how far the source is
    toward that side, following a moving source (STFT, 10 ms hops). BIN_EQ scales all of it (default 1.6: front / back and
    up / down clearly stronger)."""
    k = BIN_EQ if k is None else k
    if k <= 0 or len(x) < 4096: return x
    from scipy.signal import stft, istft
    nper = 2048; f, tt, Z = stft(x, SR, nperseg=nper, noverlap=nper - 512, boundary="even", padded=True)
    if len(keys) == 1: az = np.full(len(tt), keys[0][1]); el = np.full(len(tt), keys[0][3])
    else:
        kt = [q[0] for q in keys]; az = np.interp(tt, kt, [q[1] for q in keys]); el = np.interp(tt, kt, [q[3] for q in keys])
    ca = np.cos(np.radians(az)); ce = np.cos(np.radians(el)); se = np.sin(np.radians(el))
    wf = (np.clip(ca, 0, 1) ** 1.5) * ce; wr = (np.clip(-ca, 0, 1) ** 1.5) * ce; wu = np.clip(se, 0, 1) ** 1.2; wd = np.clip(-se / 0.7, 0, 1)
    b450, b1k, b4k, b8k, b11k, n65 = _band(f, 450.0, 0.6), _band(f, 1100.0, 0.45), _band(f, 4000.0, 0.5), _band(f, 8000.0, 0.4), _band(f, 11000.0, 0.4), _band(f, 6500.0, 0.3)
    front = 3.0 * b4k + 1.5 * b450 - 2.0 * b1k - 2.0 * b11k
    rear = 3.0 * b1k + 3.0 * b11k - 3.0 * b4k - 1.5 * b450
    up = 4.0 * b8k - 2.0 * b4k
    down = -5.0 * n65 - 2.0 * _shelf(f, 10000.0)
    gdb = k * (np.outer(front, wf) + np.outer(rear, wr) + np.outer(up, wu) + np.outer(down, wd))
    _, y = istft(Z * (10.0 ** (gdb / 20.0)), SR, nperseg=nper, noverlap=nper - 512, boundary=True)
    y = y[: len(x)]
    if len(y) < len(x): y = np.concatenate([y, np.zeros(len(x) - len(y))])
    return y.astype(np.float32)

def _with_refl(x, t, keys, room_ir, srcs_list):
    """the direct sound (direction EQ) and its early reflections for the scene's room class"""
    srcs_list.append((_dir_eq(x, keys), t, keys))
    cls = ROOM_CLASS.get(room_ir) if room_ir else None
    if cls is None or BIN_REFL <= -90: return
    for rule, ms, lvl, lp in REFL[cls]:
        rk = _refl_keys(rule, keys); y = lpf(x, lp) * db(lvl + BIN_REFL)
        srcs_list.append((_dir_eq(y, rk), t + ms / 1000.0, rk))


def _spat_render(srcs, n, wd, verbose=False):
    """run spat (Apple HRTFHQ, no distance roll-off, no built-in reverb) on [(mono, t, keys)] -> [n, 2]"""
    import shutil
    os.makedirs(wd, exist_ok=True); sc = []
    for k, (x, t, keys) in enumerate(srcs):
        if len(x) == 0 or not np.any(x): continue
        f = f"{wd}/s{k:05d}.wav"; sf.write(f, x.astype(np.float32), SR, subtype="FLOAT")
        sc.append(dict(file=f, t=float(t), gain=0.0, rv=0.0, pos=[[float(t) + kt, *_xyz(a, e, d)] for kt, a, d, e in keys]))
    if not sc: shutil.rmtree(wd, ignore_errors=True); return np.zeros((n, 2), np.float32)
    json.dump(dict(out=f"{wd}/_out.wav", dur=n / SR, sr=SR, reverb=None, ref=1.0, rolloff=0.0, maxdist=100000.0, src=sc), open(f"{wd}/_scene.json", "w"))
    r = subprocess.run([SPAT, f"{wd}/_scene.json"], capture_output=True, text=True); assert "SPAT DONE" in r.stdout, r.stdout + r.stderr
    y, _ = sf.read(f"{wd}/_out.wav", dtype="float32", always_2d=True); shutil.rmtree(wd, ignore_errors=True)
    z = np.zeros((n, 2), np.float32); m = min(n, len(y)); z[:m] = y[:m, :2]
    if verbose: print("  spat: %d sources" % len(sc), flush=True)
    return z


def _render_bin(self, stems=STEMS, verbose=False):
    """layout "bin": every object at its own place in 3D (az, el, distance; moving along its path) through Apple's HRTF, the
    rooms decoded to twelve directions round and above, low end (LFE sends) straight to both ears. -> {stem: [n, 2]}"""
    N = self.n + SR * 12; srcs = {s: [] for s in stems}; lf = {s: np.zeros(N, np.float32) for s in stems}; sends = {}
    for o in self.objs:
        if o["stem"] not in srcs or (o["only"] and o["only"] not in ("5.1", "bin")): continue
        x = o["clip"]; i = int(round(o["t"] * SR)); n = len(x)
        if i < 0: x = x[-i:]; n = len(x); i = 0
        if i + n > self.n + SR * 10: n = self.n + SR * 10 - i; x = x[:n]
        if n <= 0: continue
        t = i / SR; st = o["stem"]; el = float(o.get("el") or 0.0); azp = o.get("bin_az", o["az"])      # bin_az / bin_width: 3D-only overrides (design_v10)
        if x.ndim == 2 and x.shape[1] == 6:                               # a 5.1 recording: five virtual speakers
            for ci, a in ((0, -30.0), (1, 30.0), (2, 0.0), (4, -110.0), (5, 110.0)):
                kk = [(0.0, a, 2.0, 0.0)]; srcs[st].append((_dir_eq(x[:, ci] * (0.6 if ci == 2 else 1.0), kk), t, kk))
            if st == "AMB":                                               # full envelopment: a 5.1 bed also above
                sd = zlib.crc32(str(o["name"]).lower().encode()) % 10000
                for ci, a in ((0, -70.0), (1, 70.0)):
                    kk = [(0.0, a, 2.0, 50.0)]; srcs[st].append((_dir_eq(hpf(_decor(0.5 * (x[:, ci] + x[:, ci + 4]), sd + ci), 250) * db(BIN_UP), kk), t, kk))
            lf[st][i:i + n] += 0.5 * x[:, 3]                                  # its LFE channel as the stereo fold-down has it
            m = x[:, [0, 1, 4, 5]].mean(1)
        elif x.ndim == 2:                                                 # a stereo bed: its two sides at az -+ width (turning with an az path)
            wb = o.get("bin_width", o["width"]); w = 60.0 if wb is None else float(wb); keys = _keys(azp, 2.5, el, n)
            rg = db(BIN_REAR) if (o["only"] == "5.1" and "rear" in str(o["name"])) else 1.0
            for k, sg in ((0, -1.0), (1, 1.0)):
                kk = [(kt, a + sg * w, d, e) for kt, a, d, e in keys]; srcs[st].append((_dir_eq(x[:, k] * rg, kk), t, kk))
            if st == "AMB" and o["only"] is None:                         # full envelopment: the bed above too, and the ground's beds below
                nm = str(o["name"]).lower(); sd = zlib.crc32(nm.encode()) % 10000
                for k, sg in ((0, -1.0), (1, 1.0)):
                    if not any(g_ in nm for g_ in UPLESS):
                        kk = [(kt, a + sg * 70.0, d, 50.0) for kt, a, d, e in keys]; srcs[st].append((_dir_eq(hpf(_decor(x[:, k], sd + k), 250) * db(BIN_UP), kk), t, kk))
                    if any(g_ in nm for g_ in GROUND):
                        kk = [(kt, a + sg * 50.0, d, -35.0) for kt, a, d, e in keys]; srcs[st].append((_dir_eq(lpf(_decor(x[:, k], sd + 7 + k), 9000) * db(BIN_LOW), kk), t, kk))
            m = x.mean(1)
            if o["lfe"] is not None: lf[st][i:i + n] += lpf(m, 90, 4) * db(o["lfe"] - 3.0)          # a little sub for headphones (5.1 has its LFE)
        else:                                                             # a point source
            keys = _keys(azp, o["dist"], el, n)
            if o["doppler"] and len(keys) > 1:
                _, dist = _per_sample(keys, n); dly = dist / 343.0 * SR; pos = np.arange(n) - (dly - dly[0]); pos = np.clip(pos, 0, n - 2); j = pos.astype(np.int64); f = (pos - j).astype(np.float32)
                x = (x[j] * (1 - f) + x[j + 1] * f).astype(np.float32)
            rm = next((r["ir"] for r in self.rooms if r["key"] == o["room"]), None) if (o["rv"] is not None and o["room"]) else None
            _with_refl(x, t, keys, rm, srcs[st])
            if o["lfe"] is not None: lf[st][i:i + n] += lpf(x, 90, 4) * db(o["lfe"] - 3.0)
            m = x
        if o["rv"] is not None and o["room"]:
            k = (o["room"], st)
            if k not in sends: sends[k] = np.zeros(N, np.float32)
            sends[k][i:i + n] += m * db(o["rv"])
    for (rk, st), s in sends.items():                                     # the rooms: only where each send is alive
        r = next(r for r in self.rooms if r["key"] == rk); nz = np.nonzero(s)[0]
        if not len(nz): continue
        a0, a1 = int(nz[0]), int(nz[-1]) + 1
        for a, e, y in room_bin(s[a0:a1], r["ir"], **r["kw"]):
            kk = [(0.0, a, 3.0, e)]; srear = db(2.0) if abs(((a + 180.0) % 360.0) - 180.0) >= 135.0 else 1.0     # the room behind a little more
            srcs[st].append((_dir_eq(y * db(r["wet"]) * srear, kk), a0 / SR, kk))
        if verbose: print("  room", rk, st, r["ir"], flush=True)
    out = {}
    for st in stems:
        y = _spat_render(srcs[st], N, f"{V8}/spat/tmp/bin_{st}_{os.getpid()}", verbose)
        y += lf[st][:, None]; out[st] = y[: self.n]
    return out


def orbit(t0, t1, az0=0.0, period=6.0, sense=1.0, el=0.0, dist=2.5, ramp=1.5, step=0.1, el1=None):
    """an "8D" turn for a bin path, seconds relative to the object's start: from t0 the source speeds up (over `ramp` s) to one turn
    per `period` s (sense +1 = to the right: front -> right -> behind -> left), slows down to rest by t1; az0 before, az0 + the turns
    after. el may glide to el1. -> [(t, az, dist, el)]"""
    T = np.arange(0.0, t1 + step, step); w = np.zeros_like(T)
    on = (T >= t0) & (T <= t1); up = np.clip((T - t0) / ramp, 0, 1); dn = np.clip((t1 - T) / ramp, 0, 1)
    w[on] = (360.0 / period) * (np.sin(np.pi / 2 * np.minimum(up, dn)[on]) ** 2)
    az = az0 + sense * np.concatenate([[0.0], np.cumsum((w[1:] + w[:-1]) / 2 * step)])
    e = np.full_like(T, el) if el1 is None else el + (el1 - el) * np.clip((T - t0) / max(1e-6, t1 - t0), 0, 1)
    keys = [(0.0, az0, dist, el)] + [(float(t), float(a), dist, float(x)) for t, a, x in zip(T, az, e) if t0 - step <= t <= t1 + step]
    return keys + [(float(t1 + 1.0), float(az[-1]), dist, float(e[-1]))]


Mix._render_bin = _render_bin

# ---------------------------------------------------------------------------------------------- measurement helpers
def lufs(x):
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(x.astype(np.float64)))


def width_stats(x):
    """stereo image numbers: correlation of L and R, side / mid energy in dB"""
    L, Rr = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64); c = float((L * Rr).sum() / (np.sqrt((L ** 2).sum() * (Rr ** 2).sum()) + 1e-12))
    mid = ((L + Rr) / 2) ** 2; side = ((L - Rr) / 2) ** 2; return c, float(10 * np.log10(side.sum() / (mid.sum() + 1e-12) + 1e-12))


def binaural(x51, out_wav, room=None, level=-18.0, dist=2.2, workdir=None):
    """a 5.1 mix [n, 6] -> headphone stereo through Apple's HRTF: six virtual speakers round the listener (LFE added to
    L and R below 120 Hz, un-spatialised). room = an AVAudioEnvironment reverb preset name for a little 'cinema' air."""
    wd = workdir or f"{V8}/spat/tmp"; os.makedirs(wd, exist_ok=True); src = []
    for c, a in (("L", -30.0), ("R", 30.0), ("C", 0.0), ("Ls", -110.0), ("Rs", 110.0)):
        p = f"{wd}/_{c}.wav"; sf.write(p, x51[:, LAYOUTS["5.1"].index(c)], SR, subtype="FLOAT"); th = math.radians(a)
        src.append(dict(file=p, t=0.0, gain=0.0, rv=(0.25 if room else 0.0), pos=[[0.0, dist * math.sin(th), 0.0, -dist * math.cos(th)]]))
    sc = dict(out=f"{wd}/_bin.wav", dur=len(x51) / SR, sr=SR, reverb=(dict(preset=room, level=level) if room else None), ref=dist, rolloff=1.0, maxdist=100, src=src)
    json.dump(sc, open(f"{wd}/_scene.json", "w")); r = subprocess.run([f"{V8}/spat/spat", f"{wd}/_scene.json"], capture_output=True, text=True)
    assert "SPAT DONE" in r.stdout, r.stdout + r.stderr
    y, _ = sf.read(f"{wd}/_bin.wav", dtype="float32"); n = min(len(y), len(x51)); y = y[:n]
    lf = lpf(x51[:n, LAYOUTS["5.1"].index("LFE")], 120, 4) * db(4.0); y[:, 0] += lf; y[:, 1] += lf
    sf.write(out_wav, y, SR, subtype="PCM_24"); return y


if __name__ == "__main__":                                               # self test: a click train circling, both layouts
    rng = np.random.default_rng(3); n = SR * 8; x = rng.standard_normal(n).astype(np.float32) * 0.05
    M = Mix(8.0); M.room("r", 0, 8, "Old Forest Long", wet=-10.0)
    M.add(x, 0.0, az=[(t, 360.0 * t / 8.0, 3.0) for t in np.arange(0, 8.01, 0.1)], rv=-12.0)
    for lay in ("stereo", "5.1"):
        o = M.render(lay, verbose=True)["FX"]; print(lay, o.shape)
        for nm, t in (("front", 0.2), ("right 90", 2.0), ("back", 4.0), ("left 270", 6.0)):
            a = o[int((t - 0.15) * SR): int((t + 0.15) * SR)]; print("   %-9s" % nm, " ".join("%s %6.1f" % (c, 20 * np.log10(np.sqrt((a[:, k] ** 2).mean()) + 1e-9)) for k, c in enumerate(LAYOUTS[lay])))
    o = M.render("5.1")["FX"]; y = binaural(o, f"{V8}/spat/tmp/_selftest_bin.wav")
    for nm, t in (("front", 0.2), ("right 90", 2.0), ("back", 4.0), ("left 270", 6.0)):
        a = y[int((t - 0.15) * SR): int((t + 0.15) * SR)]; print("   bin %-9s L %6.1f R %6.1f" % (nm, 20 * np.log10(np.sqrt((a[:, 0] ** 2).mean()) + 1e-9), 20 * np.log10(np.sqrt((a[:, 1] ** 2).mean()) + 1e-9)))
