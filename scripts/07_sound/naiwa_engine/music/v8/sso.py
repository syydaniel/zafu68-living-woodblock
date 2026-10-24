"""A small sample player for the Sonatina Symphonic Orchestra: real recordings of an orchestra's sections
(16 first violins, celli, basses, four horns, trombones, timpani, harp, celesta, a chorus ...) played from the score's own
note lists, so no part depends on a General MIDI soundfont (the piano is rendered separately, see render_score.py).
Library: music/sf/sso-master (github.com/peastman/sso, Creative Commons Sampling Plus 1.0), SFZ text files + WAV / FLAC.

Supported from SFZ (what this library uses): #include, #define, <control> <global> <master> <group> <region>; sample,
key / lokey / hikey / pitch_keycenter, lovel / hivel, tune, transpose, volume, group_volume, pan, offset, loop_mode,
loop_start / loop_end (and the loop stored inside a WAV's smpl chunk), ampeg_attack / hold / decay / sustain / release,
amp_veltrack, gain_cc1 (dynamics), xfin / xfout over cc1 and over velocity (layer crossfades), a one-pole low-pass
(cutoff, cutoff_cc1, fil_keytrack), seq_length / seq_position (round robin). Ignored: vibrato LFOs, key switches.

    inst = Inst("Strings - Performance/1st Violins Sustain.sfz")
    y = inst.render([(t, dur, midi pitch, dynamics 1..127), ...], total_seconds)      -> stereo float32 at 48 kHz
The dynamics value is both the note's velocity and the library's "Dynamics" controller (cc1)."""
import os, re, struct, math
import numpy as np, soundfile as sf
from scipy.signal import lfilter

SR = 48000; ROOT = "~/Movies/NAIWA_BEYOND/music/sf/sso-master/Sonatina Symphonic Orchestra"
_NOTE = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}; _samples = {}


def key(v):
    v = str(v).strip().lower()
    if re.fullmatch(r"-?\d+", v): return int(v)
    m = re.fullmatch(r"([a-g])([#b]?)(-?\d+)", v); assert m, v
    return 12 * (int(m.group(3)) + 1) + _NOTE[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]


def _read(path, defines):
    txt = open(path, encoding="utf-8", errors="ignore").read(); out = []
    for line in txt.splitlines():
        line = re.sub(r"//.*", "", line)
        m = re.match(r"\s*#define\s+(\$\w+)\s+(.+?)\s*$", line)
        if m: defines[m.group(1)] = m.group(2); continue
        m = re.match(r'\s*#include\s+"(.+?)"', line)
        if m:
            p = os.path.join(os.path.dirname(path), m.group(1).replace("\\", "/"))
            if "vibrato" in p: continue
            out.append(_read(p, defines)); continue
        for k in sorted(defines, key=len, reverse=True): line = line.replace(k, defines[k])
        out.append(line)
    return "\n".join(out)


def parse(path):
    """-> list of regions (dicts of opcodes, 'sample' as an absolute path)"""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path); txt = _read(full, {}); base = os.path.dirname(full)
    toks = list(re.finditer(r"<(\w+)>|(\w+)=", txt)); lvl = {"control": {}, "global": {}, "master": {}, "group": {}}; cur = None; regs = []
    for i, m in enumerate(toks):
        if m.group(1):
            h = m.group(1)
            if h == "region": cur = {}; regs.append((dict(lvl["global"]), dict(lvl["master"]), dict(lvl["group"]), cur))
            else:
                cur = lvl.setdefault(h, {}); cur.clear()
                if h == "global": lvl["master"] = {}; lvl["group"] = {}
                if h == "master": lvl["group"] = {}
        elif cur is not None:
            end = toks[i + 1].start() if i + 1 < len(toks) else len(txt); val = txt[m.end(): end].strip()
            if m.group(2) != "sample": val = val.split()[0] if val.split() else ""
            cur[m.group(2)] = val
    out = []
    for g, ms, gr, r in regs:
        o = {}; o.update(g); o.update(ms); o.update(gr); o.update(r)
        if "sample" not in o: continue
        o["sample"] = os.path.normpath(os.path.join(base, o["sample"].replace("\\", "/")))
        if "key" in o: o.setdefault("lokey", o["key"]); o.setdefault("hikey", o["key"]); o.setdefault("pitch_keycenter", o["key"])
        o["_lo"] = key(o.get("lokey", 0)); o["_hi"] = key(o.get("hikey", 127)); o["_kc"] = key(o.get("pitch_keycenter", 60))
        out.append(o)
    return out


def sample(p):
    """-> (float32 [n, 2], rate, loop (start, end) or None from the file itself)"""
    if p not in _samples:
        x, sr = sf.read(p, dtype="float32", always_2d=True); x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]; loop = None
        if p.lower().endswith(".wav"):
            d = open(p, "rb").read(); i = 12
            while i + 8 <= len(d):
                cid = d[i:i + 4]; sz = struct.unpack("<I", d[i + 4:i + 8])[0]
                if cid == b"smpl" and sz >= 60 and struct.unpack("<I", d[i + 8 + 28:i + 8 + 32])[0] > 0:
                    ls, le = struct.unpack("<II", d[i + 8 + 44:i + 8 + 52]); loop = (int(ls), int(le))
                i += 8 + sz + (sz & 1)
        _samples[p] = (x, sr, loop)
    return _samples[p]


def _xf(v, lo, hi, rising, curve):
    if hi <= lo: return 1.0
    u = min(1.0, max(0.0, (v - lo) / (hi - lo))); u = u if rising else 1.0 - u
    return u if curve == "gain" else math.sqrt(u)


def _cubic(x, pos):
    i = np.floor(pos).astype(np.int64); f = (pos - i).astype(np.float32)[:, None]; n = len(x)
    a = x[np.clip(i - 1, 0, n - 1)]; b = x[np.clip(i, 0, n - 1)]; c = x[np.clip(i + 1, 0, n - 1)]; d = x[np.clip(i + 2, 0, n - 1)]
    return b + 0.5 * f * (c - a + f * (2 * a - 5 * b + 4 * c - d + f * (3 * (b - c) + d - a)))


class Inst:
    def __init__(self, sfz, gain_db=0.0, release=None, attack=None, transpose=0):
        self.regions = parse(sfz); self.gain = 10 ** (gain_db / 20); self.rel = release; self.att = attack; self.tr = transpose; self.seq = {}; self.name = os.path.basename(sfz)
        assert self.regions, sfz

    def note(self, pitch, dur, vel):
        """-> stereo float32 of one note (with its release tail), or None"""
        p = pitch + self.tr; out = None; cc1 = float(vel)
        cand = [r for r in self.regions if r["_lo"] <= p <= r["_hi"] and int(r.get("lovel", 0)) <= vel <= int(r.get("hivel", 127))]
        if not cand: return None
        sl = max(int(r.get("seq_length", 1)) for r in cand)
        if sl > 1:
            k = self.seq.get(p, 0) % sl + 1; self.seq[p] = self.seq.get(p, 0) + 1; cand = [r for r in cand if int(r.get("seq_position", 1)) == k] or cand
        for r in cand:
            curve = r.get("xf_cccurve", "power"); w = 1.0
            if "xfin_locc1" in r or "xfin_hicc1" in r: w *= _xf(cc1, float(r.get("xfin_locc1", 0)), float(r.get("xfin_hicc1", 0)), True, curve)
            if "xfout_locc1" in r or "xfout_hicc1" in r: w *= _xf(cc1, float(r.get("xfout_locc1", 127)), float(r.get("xfout_hicc1", 127)), False, curve)
            vc = r.get("xf_velcurve", "power")
            if "xfin_lovel" in r or "xfin_hivel" in r: w *= _xf(vel, float(r.get("xfin_lovel", 0)), float(r.get("xfin_hivel", 0)), True, vc)
            if "xfout_lovel" in r or "xfout_hivel" in r: w *= _xf(vel, float(r.get("xfout_lovel", 127)), float(r.get("xfout_hivel", 127)), False, vc)
            if w <= 1e-4 or not os.path.exists(r["sample"]): continue
            x, sr, floop = sample(r["sample"]); off = int(r.get("offset", 0))
            ratio = 2 ** (((p - r["_kc"]) * float(r.get("pitch_keytrack", 100)) + float(r.get("tune", 0)) + 100 * float(r.get("transpose", 0))) / 1200.0) * sr / SR
            mode = r.get("loop_mode"); loop = (int(r["loop_start"]), int(r["loop_end"])) if "loop_start" in r and "loop_end" in r else floop
            if mode is None: mode = "loop_continuous" if loop else "no_loop"
            rel = self.rel if self.rel is not None else float(r.get("ampeg_release", 0.05)); att = self.att if self.att is not None else max(0.0, float(r.get("ampeg_attack", 0.0)) + float(r.get("ampeg_vel2attack", 0.0)) * vel / 127.0)
            if mode == "one_shot":
                n = int((len(x) - off) / ratio); held = n
            else:
                held = int(dur * SR); n = held + int(rel * SR)
                if not (mode.startswith("loop") and loop): n = min(n, int((len(x) - off) / ratio))
            if n <= 8: continue
            pos = off + np.arange(n, dtype=np.float64) * ratio
            if mode.startswith("loop") and loop and loop[1] > loop[0] + 16:
                ls, le = loop; over = pos > le; pos[over] = ls + np.mod(pos[over] - ls, le - ls)
            y = _cubic(x, pos)
            # envelope: attack, hold, decay to sustain, release from the note's end
            e = np.ones(n, np.float32); na = int(att * SR)
            if na > 1: e[:min(na, n)] = np.linspace(0, 1, na, dtype=np.float32)[:min(na, n)]
            sus = float(r.get("ampeg_sustain", 100)) / 100.0
            if sus < 0.999:
                h0 = na + int(float(r.get("ampeg_hold", 0)) * SR); nd = max(1, int(float(r.get("ampeg_decay", 0)) * SR))
                if h0 < n: k = np.arange(n - h0) / nd; e[h0:] *= (sus + (1 - sus) * np.exp(-4.0 * k)).astype(np.float32)
            if mode != "one_shot" and held < n:
                nr = n - held; e[held:] *= np.exp(-5.0 * np.arange(nr) / max(1, nr)).astype(np.float32)
            elif mode == "one_shot":
                k = min(n, int(0.01 * SR)); e[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
            vt = float(r.get("amp_veltrack", 100)) / 100.0
            gdb = float(r.get("volume", 0)) + float(r.get("group_volume", 0)) + float(r.get("gain_cc1", 0)) * cc1 / 127.0 + vt * 40.0 * math.log10(max(vel, 1) / 127.0)
            y = y * (e * (10 ** (gdb / 20.0)) * w)[:, None]
            if r.get("fil_type", "").startswith("lpf") and "cutoff" in r:
                fc = float(r["cutoff"]) * 2 ** ((float(r.get("cutoff_cc1", 0)) * cc1 / 127.0 + float(r.get("fil_keytrack", 0)) * (p - key(r.get("fil_keycenter", 60)))) / 1200.0)
                if fc < 16000:
                    a = math.exp(-2 * math.pi * fc / SR); y = lfilter([1 - a], [1, -a], y, axis=0).astype(np.float32)
            pan = float(r.get("pan", 0)) / 100.0; th = (pan + 1) * math.pi / 4; y = y * np.array([math.cos(th), math.sin(th)], np.float32) * 1.4142
            if out is None or len(y) > len(out):
                z = np.zeros((len(y), 2), np.float32)
                if out is not None: z[:len(out)] = out
                out = z
            out[:len(y)] += y
        return out

    def render(self, notes, total):
        buf = np.zeros((int(total * SR) + SR * 12, 2), np.float32); miss = 0
        for t, d, p, v in notes:
            y = self.note(int(p), float(d), int(v))
            if y is None: miss += 1; continue
            i = int(max(0.0, t) * SR); m = min(len(y), len(buf) - i); buf[i:i + m] += y[:m]
        if miss: print("   ", self.name, "notes outside the instrument's range:", miss)
        return buf[: int(total * SR)] * self.gain


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["Strings - Performance/1st Violins Sustain.sfz", "Strings - Performance/Cello Solo Sustain.sfz", "Brass - Performance/Horns Sustain.sfz", "Percussion/Timpani.sfz", "Concert Harp.sfz"]:
        I = Inst(f); lo = min(r["_lo"] for r in I.regions); hi = max(r["_hi"] for r in I.regions)
        y = I.render([(0.0, 2.0, (lo + hi) // 2, 60), (2.5, 2.0, (lo + hi) // 2, 110)], 6.0)
        r = lambda a: 20 * np.log10(np.sqrt((a ** 2).mean()) + 1e-9)
        print(f, len(I.regions), "regions, keys", lo, "-", hi, "| soft %.1f dB, loud %.1f dB, peak %.1f" % (r(y[: 2 * SR]), r(y[int(2.5 * SR): int(4.5 * SR)]), 20 * np.log10(np.abs(y).max() + 1e-9)))
