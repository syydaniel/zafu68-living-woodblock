"""NAIWA BEYOND - the final mix (audio/v8).  python mix_v8.py <score dir, score.wav or "none"> <out name> [a-b seconds]

Stems of the sound design (<PRE_DIR>/<layout>/{DX,FX,AMB}.wav, default audio/v8/pre; e.g. PRE_DIR=audio/v8/pre_v9t from
design_v9.py) + the score (a score dir from music/v8/mix_score.py, or one stereo file) -> masters (layouts from LAYS):
    out/<name>/stereo/master.wav     loudspeaker stereo
    out/<name>/51/master.wav         5.1 (L R C LFE Ls Rs): dialogue in the centre, effects and beds all round, the score
                                     in the fronts with its air behind, real sub-bass in the LFE
    out/<name>/binaural/master.wav   the 5.1 master through Apple's HRTF (six virtual speakers): spatial audio on headphones
    out/<name>/bin/master.wav        with LAYS=...,bin: the 3D headphone mix (objects from PRE_BIN, the score's mx_bin.wav)
and for each the four stems DX / MX / FX / AMB (they add up to the master: the limiter's gain is applied to each).

Dialogue is the anchor. Under a line the score steps back (up to 5 dB) and so do effects and beds (2 dB). No bus
compression: the quiet stays quiet so that the big moments are big. Master: -1 dBTP look-ahead limiter; the whole film
is set to TARGET LUFS (integrated) by one gain.
Run with ~/venvs/audio/bin/python."""
import sys, os, json
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eng import SR, V8, LAYOUTS, db, lpf, hpf, fade, binaural, width_stats
from snd import limiter
from cut import END, FILM
import pyloudnorm as pyln

TARGET = float(os.environ.get("TARGET", "-16.0")); CEIL = -1.0
G = dict(DX=float(os.environ.get("G_DX", "0.0")), MX=float(os.environ.get("G_MX", "0.0")), FX=float(os.environ.get("G_FX", "0.0")), AMB=float(os.environ.get("G_AMB", "0.0")))
DUCK_MX, DUCK_FX = float(os.environ.get("DUCK_MX", "5.0")), float(os.environ.get("DUCK_FX", "2.0"))
LAYS = os.environ.get("LAYS", "stereo,5.1").split(",")                     # "bin" = the 3D headphone mix (design_v10 / design_v11 + mix_score_bin), read from PRE_BIN
CRED_DB = float(os.environ.get("CRED_DB", "-3.0"))                         # the ending song sits a little under the film (as rendered, it is 4 LU louder than the film's body)


def duck_curve(dx_mono, depth, att=0.22, rel=0.9, hold=0.25, thr=-42.0):
    """gain (linear) that steps back by `depth` dB while there is speech"""
    h = SR // 100; e = np.sqrt(np.maximum(1e-12, np.array([(dx_mono[i:i + h] ** 2).mean() for i in range(0, len(dx_mono) - h, h)]))); on = (20 * np.log10(e) > thr).astype(np.float64)
    k = int(hold * 100); on = np.convolve(on, np.ones(2 * k + 1), "same") > 0                # hold across the gaps between words
    g = np.zeros(len(on)); s = 0.0; a = np.exp(-1 / (att * 100)); r = np.exp(-1 / (rel * 100))
    for i, v in enumerate(on):
        s = (a if v > s else r) * s + (1 - (a if v > s else r)) * v; g[i] = s
    g = np.interp(np.arange(len(dx_mono)), np.arange(len(g)) * h + h / 2, g); return db(-depth * g).astype(np.float32)


def leveler(x, ceil_db, most=4.0, hold=0.5, smooth=0.6):
    """a slow hand on the master fader for the few places where the whole mix would run several dB into the limiter (the end of the passage, the
    mothership): the gain the peaks need (10 ms blocks) is held half a second either side and smoothed over 0.6 s, at most `most` dB. The limiter
    behind it is left with the transients only. -> gain (linear) per sample"""
    from scipy.ndimage import minimum_filter1d
    h = SR // 100; pk = np.abs(x).max(1); nb = len(pk) // h; b = pk[: nb * h].reshape(nb, h).max(1)
    need = np.clip(ceil_db + 1.5 - 20 * np.log10(np.maximum(b, 1e-9)), -most, 0.0)                 # (the limiter may still take 1.5 dB off the tops)
    k = int(hold * 100); need = minimum_filter1d(need, size=2 * k + 1, mode="nearest"); w = np.hanning(int(smooth * 100) | 1); w /= w.sum()
    g = np.convolve(np.pad(need, (len(w) // 2, len(w) // 2), mode="edge"), w, "valid")
    return db(np.interp(np.arange(len(x)), np.arange(nb) * h + h / 2, g)).astype(np.float32)


def score_to(layout, mx):
    """the stereo score in a layout. 5.1: fronts as they are; behind, its own air (other side, 22 ms later, dull, -7 dB);
    below 80 Hz also in the LFE (-10 dB)."""
    if layout == "stereo": return mx
    chs = LAYOUTS[layout]; out = np.zeros((len(mx), len(chs)), np.float32); out[:, 0] = mx[:, 0]; out[:, 1] = mx[:, 1]
    d = int(0.022 * SR); back = lpf(mx[:, ::-1], 6500) * db(-7.0); out[d:, chs.index("Ls")] = back[:-d, 0]; out[d:, chs.index("Rs")] = back[:-d, 1]
    out[:, chs.index("LFE")] = lpf(mx.mean(1), 80, 4) * db(-10.0); return out


def main():
    score = sys.argv[1]; name = sys.argv[2]; rng = None
    if len(sys.argv) > 3:
        a, b = (float(v) for v in sys.argv[3].split("-")); rng = (int(a * SR), int(b * SR))
    n = int(END * SR); rep = {}
    mx = np.zeros((n, 2), np.float32); mx_lay = {}
    if os.path.isdir(score):                                              # a score already seated in its hall for each layout (music/v8/mix_score.py)
        for lay, f in (("stereo", "mx_stereo.wav"), ("5.1", "mx_51.wav"), ("bin", "mx_bin.wav")):
            if not os.path.exists(f"{score}/{f}"): continue
            y, sr = sf.read(f"{score}/{f}", dtype="float32", always_2d=True); assert sr == SR, sr; z = np.zeros((n, y.shape[1]), np.float32); m = min(n, len(y)); z[:m] = y[:m]; mx_lay[lay] = z
    elif score != "none":
        y, sr = sf.read(score, dtype="float32", always_2d=True); assert sr == SR, sr; m = min(n, len(y)); mx[:m] = y[:m, :2]
    masters = {}
    for lay in LAYS:
        d = f"{os.environ.get('PRE_BIN' if lay == 'bin' else 'PRE_DIR', os.environ.get('PRE_DIR', V8 + '/pre'))}/{lay.replace('.', '')}"; chs = LAYOUTS[lay]         # e.g. PRE_DIR=audio/v8/pre_v9t (design_v9.py)
        st = {s: sf.read(f"{d}/{s}.wav", dtype="float32", always_2d=True)[0][:n] for s in ("DX", "FX", "AMB")}
        st["MX"] = mx_lay[lay] if lay in mx_lay else score_to(lay, mx)
        if lay == "bin":                                                  # the 3D stems keep the stereo balance: each as loud as its stereo stem
            ref = f"{os.environ.get('PRE_DIR', V8 + '/pre')}/stereo"; mt = pyln.Meter(SR)
            for k_ in ("DX", "FX", "AMB"):
                if not os.path.exists(f"{ref}/{k_}.wav") or np.abs(st[k_]).max() < 1e-6: continue
                a_ = sf.read(f"{ref}/{k_}.wav", dtype="float32", always_2d=True)[0][:n]; dg = float(np.clip(mt.integrated_loudness(a_.astype(np.float64)) - mt.integrated_loudness(st[k_].astype(np.float64)), -3.0, 3.0))
                st[k_] = st[k_] * db(dg); print("bin", k_, "matched to stereo: %+.2f dB" % dg, flush=True)
        dxm = st["DX"].sum(1) if lay in ("stereo", "bin") else st["DX"][:, [0, 1, 2]].sum(1)
        gm = duck_curve(dxm, DUCK_MX); gf = duck_curve(dxm, DUCK_FX)
        cr = np.ones(n, np.float32); i0 = int(FILM * SR); k2 = int(2.0 * SR); cr[i0:] = db(CRED_DB); cr[i0:i0 + k2] = db(CRED_DB * np.linspace(0, 1, k2)).astype(np.float32); gm = gm[:n] * cr[: len(gm)]
        st["MX"] = st["MX"] * gm[:, None] * db(G["MX"]); st["FX"] = st["FX"] * gf[:, None] * db(G["FX"]); st["AMB"] = st["AMB"] * gf[:, None] * db(G["AMB"]); st["DX"] = st["DX"] * db(G["DX"])
        ef = np.ones(n, np.float32); k_ = int(1.2 * SR); ef[-k_:] = np.cos(np.linspace(0, np.pi / 2, k_)) ** 2          # the film ends in silence: the last 1.2 s of the hall's tail fade
        st = {k: v * ef[: len(v), None] for k, v in st.items()}
        if rng: st = {k: v[rng[0]: rng[1]] for k, v in st.items()}
        mix = sum(st.values())
        # loudness: one gain for the whole film, then the limiter, then a second pass for what the limiter took
        meter = pyln.Meter(SR); order = [0, 1, 2, 4, 5] if lay == "5.1" else [0, 1]         # BS.1770: no LFE
        g0 = None
        L0 = meter.integrated_loudness(mix[:, order].astype(np.float64)); g = TARGET - L0                  # each layout is set to the target by its own gain
        for _ in range(3):
            lv = leveler(mix * db(g), CEIL); y, gr = limiter(mix * db(g) * lv[:, None], CEIL); L1 = meter.integrated_loudness(y[:, order].astype(np.float64)); g += TARGET - L1
        lv = leveler(mix * db(g), CEIL); y, gr, gl = limiter(mix * db(g) * lv[:, None], CEIL, ret_gain=True); gl = gl * lv        # stems get both gains: they still add up to the master
        print(lay, "leveler: at most %.1f dB, below -1 dB for %.1f s" % (20 * np.log10(lv.min()), (lv < db(-1.0)).sum() / SR), flush=True)
        L1 = meter.integrated_loudness(y[:, order].astype(np.float64)); o = f"{V8}/out/{name}/{lay.replace('.', '')}"; os.makedirs(o, exist_ok=True)
        gd = 20 * np.log10(np.maximum((gl / lv)[::480], 1e-6)); hot = np.where(gd < -1.5)[0]; spans = []                 # where the limiter works by more than 1.5 dB (10 ms steps)
        for i in hot:
            if spans and i - spans[-1][1] <= 50: spans[-1][1] = i; spans[-1][2] = min(spans[-1][2], gd[i])
            else: spans.append([i, i, gd[i]])
        print(lay, "limiter over 1.5 dB: %.1f s in all;" % (len(hot) / 100.0), ", ".join("%.1f-%.1f (%.1f)" % (a_ / 100, b_ / 100, c_) for a_, b_, c_ in spans[:40]), flush=True)
        sf.write(f"{o}/master.wav", y, SR, subtype="PCM_24")
        for k, v in st.items():
            sf.write(f"{o}/{k}.wav", v * db(g) * gl[:, None], SR, subtype="PCM_24")
        if lay == "5.1":                                                  # tag the layout (L R C LFE Ls Rs): untagged 6-channel files arrive in Resolve as six mono tracks
            import subprocess
            for k in ["master"] + list(st):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{o}/{k}.wav", "-c:a", "pcm_s24le", "-af", "channelmap=channel_layout=5.1", f"{o}/_{k}.wav"], check=True); os.replace(f"{o}/_{k}.wav", f"{o}/{k}.wav")
        gdir = f"{score}/groups" if lay == "stereo" else f"{score}/groups_bin"
        if lay in ("stereo", "bin") and os.path.isdir(gdir) and not rng:          # the score's instrument groups with the same gains: separate tracks in Resolve that add up to MX
            for f in sorted(os.listdir(gdir)):
                x, _ = sf.read(f"{gdir}/{f}", dtype="float32", always_2d=True); z = np.zeros((n, 2), np.float32); m = min(n, len(x)); z[:m] = x[:m]
                sf.write(f"{o}/MX_{f}", z * gm[:, None] * db(G["MX"] + g) * gl[:, None], SR, subtype="FLOAT")        # float: one group alone may pass 0 dBFS where the sum does not
        st_l = {k: float(meter.integrated_loudness((v * db(g))[:, order].astype(np.float64))) if np.abs(v).max() > 1e-6 else None for k, v in st.items()}
        rep[lay] = dict(lufs=round(L1, 2), master_gain_db=round(g, 2), limiter_max_reduction_db=round(gr, 2), peak_dbfs=round(float(20 * np.log10(np.abs(y).max() + 1e-9)), 2), stems_lufs={k: (round(v, 1) if v else None) for k, v in st_l.items()})
        if lay in ("stereo", "bin"):
            c, sm = width_stats(y); rep[lay]["lr_correlation"] = round(c, 3); rep[lay]["side_over_mid_db"] = round(sm, 1)
        masters[lay] = y; print(lay, rep[lay], flush=True)
    if "5.1" not in masters:
        json.dump(rep, open(f"{V8}/out/{name}/report.json", "w"), indent=1); return
    o = f"{V8}/out/{name}/binaural"; os.makedirs(o, exist_ok=True)
    yb = binaural(masters["5.1"], f"{o}/_raw.wav", room="mediumHall", level=-24.0); meter = pyln.Meter(SR)
    Lb = meter.integrated_loudness(yb.astype(np.float64)); yb = yb * db(TARGET - Lb); yb = yb * leveler(yb, CEIL)[:, None]; yb2, gr = limiter(yb, CEIL); sf.write(f"{o}/master.wav", yb2, SR, subtype="PCM_24"); os.remove(f"{o}/_raw.wav")
    c, sm = width_stats(yb2); rep["binaural"] = dict(lufs=round(float(meter.integrated_loudness(yb2.astype(np.float64))), 2), limiter_max_reduction_db=round(gr, 2), lr_correlation=round(c, 3), side_over_mid_db=round(sm, 1))
    print("binaural", rep["binaural"], flush=True)
    json.dump(rep, open(f"{V8}/out/{name}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
