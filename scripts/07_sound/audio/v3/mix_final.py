"""ZAFU final mix (after the 奶蛙 mix_v8): narration (DX) + score (MX) + effects (FX) + beds (AMB) -> two masters
  out/<tag>/stereo/master.wav    loudspeakers / phones without head tracking
  out/<tag>/binaural/master.wav  "8D" headphone mix: the score seated round you with orbits (music/mix_zafu_bin.py), effects and beds
                                 placed and turning round the head (audio/v3/design_zafu.py bin), the voice in front with a little room
Under speech the score steps back (DUCK_MX dB) and effects / beds (DUCK_FX dB). One gain to TARGET LUFS, -1 dBTP look-ahead limiter.
Stems are written beside each master (they add up to it).
  ~/venvs/audio/bin/python audio/v3/mix_final.py <tag> <cues.json> [--voice vox_design] [--end 108]
cues.json = [["u01", 1.2], ...] narration utterance -> film second (from work/v3/build.py CUES)."""
import sys, os, json
import numpy as np, soundfile as sf, pyloudnorm as pyln
V8 = "~/Movies/NAIWA_BEYOND/audio/v8"; sys.path.insert(0, V8)
from eng import SR, Mix, db, lpf, hpf, width_stats
from snd import limiter
ROOT = "~/Movies/ZAFU2"
SCORE = os.environ.get("SCORE_DIR", f"{ROOT}/music/score")
PRE = os.environ.get("PRE_DIR", f"{ROOT}/audio/v3/pre")
tag = sys.argv[1]; cues = json.load(open(sys.argv[2]))
voice = sys.argv[sys.argv.index('--voice') + 1] if '--voice' in sys.argv else 'vox_design'
END = float(sys.argv[sys.argv.index('--end') + 1]) if '--end' in sys.argv else 108.0
TARGET = float(os.environ.get('TARGET', '-14.0')); DUCK_MX = float(os.environ.get('DUCK_MX', '5.0')); DUCK_FX = float(os.environ.get('DUCK_FX', '2.0'))
N = int(END * SR)


def fit(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 1: x = np.stack([x, x], 1)
    y = np.zeros((N, 2), np.float32); n = min(N, len(x)); y[:n] = x[:n, :2]; return y


def duck(dx_mono, depth, att=0.22, rel=0.9, hold=0.25, thr=-42.0):
    h = SR // 100; e = np.sqrt(np.maximum(1e-12, np.array([(dx_mono[i:i + h] ** 2).mean() for i in range(0, len(dx_mono) - h, h)]))); on = (20 * np.log10(e) > thr).astype(np.float64)
    k = int(hold * 100); on = np.convolve(on, np.ones(2 * k + 1), 'same') > 0
    g = np.zeros(len(on)); s = 0.0; a = np.exp(-1 / (att * 100)); r = np.exp(-1 / (rel * 100))
    for i, v in enumerate(on):
        s = (a if v > s else r) * s + (1 - (a if v > s else r)) * v; g[i] = s
    g = np.interp(np.arange(len(dx_mono)), np.arange(len(g)) * h + h / 2, g); return db(-depth * g).astype(np.float32)


def polish(x):
    """broadcast-narration polish of one line (the voice keeps its character; nothing is pitched or time-stretched):
    a gentle de-esser (5.5-10 kHz only, when it stands out), a slow leveler so every word of the line sits at about the same
    level (2:1 from 3 dB under the line average, 20 ms attack, 250 ms release), and a light tone: -1.5 dB at 280 Hz (boom),
    +1.5 dB at 3.4 kHz (presence), +1 dB shelf above 10 kHz (air)"""
    import pedalboard as pb
    from scipy.signal import butter, sosfiltfilt
    x = x.astype(np.float32)
    # de-esser: split at 5.5 kHz, compress the top band against the whole line's level
    sos = butter(4, 5500, 'hp', fs=SR, output='sos'); hi = sosfiltfilt(sos, x).astype(np.float32); lo = x - hi
    w = int(0.004 * SR); env_h = np.sqrt(np.convolve(hi ** 2, np.ones(w) / w, 'same') + 1e-12); env_a = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, 'same') + 1e-12)
    rel = 20 * np.log10(env_h / env_a); over = np.clip(rel + 9.0, 0, None)            # the top band louder than -9 dB of the whole: sibilant
    g = 10 ** (-over * 0.6 / 20); k = int(0.002 * SR); g = np.convolve(g, np.ones(k) / k, 'same')
    x = lo + hi * g.astype(np.float32)
    L = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
    board = pb.Pedalboard([pb.Compressor(threshold_db=L - 3.0, ratio=2.0, attack_ms=20.0, release_ms=250.0),
                           pb.PeakFilter(cutoff_frequency_hz=280.0, gain_db=-1.5, q=1.0),
                           pb.PeakFilter(cutoff_frequency_hz=3400.0, gain_db=1.5, q=0.8),
                           pb.HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.0, q=0.7)])
    return board(x[None, :], SR)[0].astype(np.float32)


def polish2(x):
    """fuller narration polish, set from measurements of the whole narration (LTAS 160-400 Hz +11 dB over 1 kHz (thick), 4 kHz
    -16 dB (dull), 24 breaths up to -25 dB under the peaks, a little 50 / 150 Hz hum, a roomy tail of about -130 dB/s). In order:
    narrow notches on the hum; breaths between phrases (unvoiced, quieter than peak - 20 dB, longer than 0.15 s) lowered 9 dB with
    40 ms ramps (kept, never cut out); the de-esser of polish(); a soft downward expander (1:1.8 from 30 dB under the line) that
    shortens the room's tail between words; a leveler (2:1); tone: -4.5 dB at 220 Hz (wide), +3 dB at 4 kHz, +2 dB shelf above 10 kHz"""
    import pedalboard as pb
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
    x = x.astype(np.float32)
    for f0 in (50.0, 150.0):
        b, a = iirnotch(f0, 30.0, SR); x = filtfilt(b, a, x).astype(np.float32)
    # breaths
    hop = int(0.01 * SR); n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    zc = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(1)                       # zero crossings: high for breath / fricatives
    pk = np.percentile(e, 98)
    win = np.hanning(2 * hop).astype(np.float32); spec = np.zeros((n, hop + 1), np.float32)
    for q in range(n):                                                            # 20 ms frames: a sibilant (s sh x c ch q) has its
        seg_ = x[q * hop: q * hop + 2 * hop]                                       # energy above 4 kHz, a breath is flatter
        if len(seg_) == 2 * hop: spec[q] = np.abs(np.fft.rfft(seg_ * win)) ** 2
    ff = np.fft.rfftfreq(2 * hop, 1 / SR)
    sib = 10 * np.log10(spec[:, (ff > 4000) & (ff < 10000)].sum(1) + 1e-12) - 10 * np.log10(spec[:, (ff > 500) & (ff < 3000)].sum(1) + 1e-12) > 2.0
    sib = np.convolve(sib.astype(np.float32), np.ones(5), 'same') > 0               # and a little round each sibilant frame
    cand = (e < pk - 20) & (e > pk - 55) & (zc > 0.12) & ~sib
    g = np.ones(n, np.float32); i = 0; nb = 0
    while i < n:
        if cand[i]:
            j = i
            while j < n and cand[j]: j += 1
            if j - i >= 15: g[i:j] = 10 ** (-9 / 20); nb += 1
            i = j
        else: i += 1
    k = 4; g = np.convolve(np.pad(g, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid')
    gs = np.interp(np.arange(len(x)), np.arange(n) * hop + hop / 2, g).astype(np.float32); x = x * gs
    polish2.breaths = getattr(polish2, 'breaths', 0) + nb
    # de-esser (as polish)
    sos = butter(4, 5500, 'hp', fs=SR, output='sos'); hi = sosfiltfilt(sos, x).astype(np.float32); lo = x - hi
    w = int(0.004 * SR); env_h = np.sqrt(np.convolve(hi ** 2, np.ones(w) / w, 'same') + 1e-12); env_a = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, 'same') + 1e-12)
    over = np.clip(20 * np.log10(env_h / env_a) + 9.0, 0, None)
    gg = 10 ** (-over * 0.6 / 20); kk = int(0.002 * SR); gg = np.convolve(gg, np.ones(kk) / kk, 'same')
    x = lo + hi * gg.astype(np.float32)
    L = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
    board = pb.Pedalboard([pb.NoiseGate(threshold_db=L - 30.0, ratio=1.8, attack_ms=2.0, release_ms=140.0),
                           pb.Compressor(threshold_db=L - 3.0, ratio=2.0, attack_ms=20.0, release_ms=250.0),
                           pb.PeakFilter(cutoff_frequency_hz=220.0, gain_db=-4.5, q=0.8),
                           pb.PeakFilter(cutoff_frequency_hz=4000.0, gain_db=3.0, q=0.8),
                           pb.HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0, q=0.7)])
    return board(x[None, :], SR)[0].astype(np.float32)


def declick(x):
    """mouth clicks (口水音) out, nothing else touched, so the voice stays as recorded. A click is a burst of 1-3 ms in the band above
    2.5 kHz, at least 9 dB over its 30 ms surround and 4 dB over its 8 ms surround (consonant bursts are wider, so they are left alone).
    Inside speech only the click's top band is taken out (raised-cosine, 5 ms); in a pause the whole 5 ms is lowered 20 dB."""
    from scipy.signal import butter, sosfiltfilt
    x = x.astype(np.float32).copy()
    hp = sosfiltfilt(butter(4, 2500, 'hp', fs=SR, output='sos'), x).astype(np.float32)
    rms = lambda sig, w: np.sqrt(np.convolve(sig ** 2, np.ones(w) / w, 'same') + 1e-12)
    r1, r8, r30 = rms(hp, int(0.0015 * SR)), rms(hp, int(0.008 * SR)), rms(hp, int(0.030 * SR))
    full = 20 * np.log10(rms(x, int(0.030 * SR)) + 1e-9); pk = np.percentile(full, 98)
    hit = (20 * np.log10(r1 / r30) > 9.0) & (20 * np.log10(r1 / r8) > 4.0)
    idx = np.where(hit)[0]; n = 0
    if len(idx):
        ev = []; s0 = idx[0]; pv = idx[0]
        for i in idx[1:]:
            if i - pv > int(0.003 * SR): ev.append((s0, pv)); s0 = i
            pv = i
        ev.append((s0, pv))
        for a, b in ev:
            if b - a > int(0.004 * SR): continue                                  # longer than a click: a consonant
            c = (a + b) // 2; w0, w1 = max(0, c - int(0.0015 * SR)), min(len(x), c + int(0.0035 * SR))
            win = np.hanning(w1 - w0).astype(np.float32)
            if full[c] < pk - 22: x[w0:w1] *= (1 - win * (1 - 10 ** (-20 / 20)))    # in a pause: the whole click down
            else: x[w0:w1] -= hp[w0:w1] * win                                     # in speech: only its top band
            n += 1
    declick.n = getattr(declick, 'n', 0) + n
    return x


# narration: the chosen takes laid on their cues, cleaned (HPF 80 Hz), each line at the same spoken level
timing = json.load(open(f"{ROOT}/vo/audition/{voice}_timing.json"))
full, _ = sf.read(f"{ROOT}/vo/audition/{voice}_full.wav", dtype='float32')
if full.ndim > 1: full = full.mean(1)
meter = pyln.Meter(SR)
dx = np.zeros(N, np.float32)
prev_end = 0.0
placed = []
for u, t in cues:
    t = max(t, prev_end + 0.25)
    placed.append((u, t))                    # never let two lines overlap (takes differ in length)
    prev_end = t + timing[u]['dur']
    tm = timing[u]; seg = full[int(tm['start'] * SR): int((tm['start'] + tm['dur']) * SR)].copy()
    seg = hpf(seg, 80.0, 2)
    if os.environ.get('VOICE_POLISH', '0') == '1': seg = polish(seg)
    elif os.environ.get('VOICE_POLISH', '0') == '2': seg = polish2(seg)
    elif os.environ.get('VOICE_POLISH', '0') == '3': seg = declick(declick(seg))      # the original voice, mouth clicks out (two passes)
    elif os.environ.get('VOICE_POLISH', '0') == '4': seg = polish(declick(declick(seg)))   # mouth clicks out (two passes), then the light polish
    L = meter.integrated_loudness(np.stack([seg, seg], 1).astype(np.float64)) if len(seg) > SR * 0.5 else -30
    seg *= db(-22.0 - L)
    if os.environ.get('VOICE_POLISH', '0') in ('1', '2', '3', '4'):
        # crest control per line: consonant peaks no more than 11 dB over the line's loudness (transparent, ~1-3 dB on the
        # hardest consonants), then the line is set to the same loudness again, so every line keeps one level
        for _ in range(2):
            seg = limiter(seg[:, None], -22.0 + float(os.environ.get('CREST', '11')))[0][:, 0]
            L = meter.integrated_loudness(np.stack([seg, seg], 1).astype(np.float64)); seg *= db(-22.0 - L)
    if os.environ.get('SEG_FADE', '0') == '1':                                     # no hard cut of the room tone at a line's ends
        fi_, fo_ = int(0.03 * SR), int(0.08 * SR)
        seg[:fi_] *= np.sin(np.linspace(0, np.pi / 2, fi_)) ** 2; seg[-fo_:] *= np.cos(np.linspace(0, np.pi / 2, fo_)) ** 2
    i0 = int(t * SR); dx[i0:i0 + len(seg)] += seg[: N - i0]
masters, report = {}, {}
for lay in ('stereo', 'bin'):
    if lay == 'stereo':
        DX = fit(dx) * 0.707
        VR = float(os.environ.get('VOICE_ROOM', '0'))
        if VR > 0:
            Mv0 = Mix(END + 1.0); Mv0.room('hall', -1.0, END + 5, 'Opulent Ballroom', wet=-17.0, hp=150.0, pre=0.02, lp=7000.0)
            Mv0.add(dx * 1.0, 0.0, stem='DX', az=0.0, dist=1.6, rv=-22.0, room='hall', name='narration room only')
            DX = DX + VR * fit(Mv0.render('stereo', stems=('DX',))['DX'])
        MX = fit(sf.read(f"{SCORE}/mx_stereo.wav", dtype='float32')[0])
        FX = fit(sf.read(f"{PRE}/stereo/FX.wav", dtype='float32')[0]); AMB = fit(sf.read(f"{PRE}/stereo/AMB.wav", dtype='float32')[0])
    else:
        # the voice is NOT spatialised (it stays still in front): the same centred, dry narration as the stereo master,
        # with only a whisper of the hall behind it so it sits in the same space as the 3D score
        Mv = Mix(END + 1.0); Mv.room('hall', -1.0, END + 5, 'Opulent Ballroom', wet=-17.0, hp=150.0, pre=0.02, lp=7000.0)
        Mv.add(dx * 0.0 + dx, 0.0, stem='DX', az=0.0, dist=1.6, el=0.0, rv=-22.0, room='hall', name='narration room only', only='bin')
        room = fit(Mv.render('bin', stems=('DX',))['DX'])
        DX = masters_stereo_dx.copy() + float(os.environ.get('VOICE_ROOM_BIN', '0.12')) * room
        MX = fit(sf.read(f"{SCORE}/mx_bin.wav", dtype='float32')[0])
        FX = fit(sf.read(f"{PRE}/bin/FX.wav", dtype='float32')[0]); AMB = fit(sf.read(f"{PRE}/bin/AMB.wav", dtype='float32')[0])
    if lay == 'stereo': masters_stereo_dx = DX.copy()
    # score level against the voice: the score's loud passages sit ~6 LU under the narration level
    MX *= db(-27.0 - meter.integrated_loudness(MX.astype(np.float64)))
    dm = DX.sum(1)
    # from RELAX_T on (the last line, 生日快乐, sung over the score's full chord) the score gives way only RELAX of its duck
    RT = float(os.environ.get('RELAX_T', '1e9')); RX = float(os.environ.get('RELAX', '1.0'))
    wr = np.clip((np.arange(len(dm)) / SR - (RT - 0.3)) / 0.3, 0, 1).astype(np.float32)
    MX *= (duck(dm, DUCK_MX) * (1 - wr) + duck(dm, DUCK_MX * RX) * wr)[:, None]; FX *= duck(dm, DUCK_FX)[:, None]; AMB *= duck(dm, DUCK_FX)[:, None]
    # under the voice the score also gives way where the voice lives (a documentary mixer's dynamic EQ): its 0.8-3.5 kHz band is
    # carved by MIDCARVE dB while someone speaks, on the same slow envelope as the duck, so the music stays full but never masks a word
    MIDC = float(os.environ.get('MIDCARVE', '0'))
    if MIDC > 0:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, [800.0, 3500.0], 'bandpass', fs=SR, output='sos')
        mid = sosfiltfilt(sos, MX, axis=0).astype(np.float32)
        MX = MX - mid + mid * duck(dm, MIDC)[:, None]
    # keep the voice steady against everything behind it (one level for the voice; the background moves instead):
    # during each spoken line the background (score + effects + beds) is lowered as needed so the narration stays at least
    # GAP dB above it; ramps of 0.35 s in / 0.8 s out, the voice itself is never touched
    GAP = float(os.environ.get('GAP', '0'))
    if GAP > 0:
        bgain = np.ones(len(dm), np.float32)
        for (u, t) in placed:
            a, b = int(t * SR), int((t + timing[u]['dur']) * SR)
            lv = meter.integrated_loudness(DX[a:b].astype(np.float64)); lb = meter.integrated_loudness((MX + FX + AMB)[a:b].astype(np.float64))
            need = (lv - lb) - (float(os.environ.get('GAP_END', GAP)) if t >= RT - 2.5 else GAP)    # the last line: a smaller gap
            if need < 0:
                r0, r1 = int(0.35 * SR), int(0.8 * SR)
                env = np.ones(len(dm), np.float32) * 0.0
                env[a:b] = 1.0
                env[max(0, a - r0):a] = np.linspace(0, 1, a - max(0, a - r0))
                env[b:min(len(dm), b + r1)] = np.linspace(1, 0, min(len(dm), b + r1) - b)
                bgain = np.minimum(bgain, db(need * env).astype(np.float32))
        MX *= bgain[:, None]; FX *= bgain[:, None]; AMB *= bgain[:, None]
    # the background as one bus, glued by a gentle slow compressor (so score, effects and beds breathe together under the voice)
    if os.environ.get('BG_GLUE', '0') == '1':
        import pedalboard as pb
        BG0 = MX + FX + AMB; Lb = 20 * np.log10(np.sqrt((BG0 ** 2).mean()) + 1e-9)
        comp = pb.Pedalboard([pb.Compressor(threshold_db=Lb + 4.0, ratio=1.6, attack_ms=30.0, release_ms=350.0)])
        g_ = comp(BG0.T.astype(np.float32), SR).T / np.where(np.abs(BG0) > 1e-7, BG0, 1.0)
        env_ = np.clip(np.median(np.abs(g_), axis=1), 0.25, 1.0); k_ = int(0.02 * SR); env_ = np.convolve(env_, np.ones(k_) / k_, 'same').astype(np.float32)
        MX *= env_[:, None]; FX *= env_[:, None]; AMB *= env_[:, None]
    mix = DX + MX + FX + AMB
    if os.environ.get('LEVELER', '0') == '1':
        # (as in the 奶蛙 mix_v8) a slow hand on the master fader, here on the BACKGROUND only (the voice keeps one level
        # everywhere): where score + effects would run into the limiter they are eased down (at most 4 dB, held 0.5 s, smoothed
        # 0.6 s). The voice gets its own transparent peak control (its consonant peaks only, never its level), so the final limiter
        # is left with almost nothing. The gain is iterated to reach TARGET.
        sys.path.insert(0, V8); import mix_v8 as _MV
        BG = MX + FX + AMB
        def build(gdb):
            dxg = DX * db(gdb)                                  # (voice crest already controlled per line)
            lvl = _MV.leveler(BG * db(gdb) + dxg, -1.0)
            return dxg, lvl
        gdb = TARGET - meter.integrated_loudness(mix.astype(np.float64))
        for _ in range(3):
            dxg, lvl = build(gdb); y, gr = limiter(dxg + BG * db(gdb) * lvl[:, None], -1.0); gdb += TARGET - meter.integrated_loudness(y.astype(np.float64))
        dxg, lvl = build(gdb)
        st = dict(DX=dxg, MX=MX * db(gdb) * lvl[:, None], FX=FX * db(gdb) * lvl[:, None], AMB=AMB * db(gdb) * lvl[:, None])
    else:
        g = db(TARGET - meter.integrated_loudness(mix.astype(np.float64)))
        st = {k: v * g for k, v in dict(DX=DX, MX=MX, FX=FX, AMB=AMB).items()}
    m = sum(st.values())
    y, gr = limiter(m, -1.0)
    fade_n = int(2.0 * SR); y[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None]
    o = f"{ROOT}/audio/v3/out/{tag}/{'binaural' if lay == 'bin' else 'stereo'}"; os.makedirs(o, exist_ok=True)
    sf.write(f"{o}/master.wav", y, SR, subtype='PCM_24')
    for k, v in st.items():
        vv = v.copy(); vv[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None]; sf.write(f"{o}/{k}.wav", vv, SR, subtype='FLOAT')   # pre-limiter stems
    c, sm = width_stats(y)
    report[lay] = dict(limiter_max_db=round(float(gr), 2), lufs=round(float(meter.integrated_loudness(y.astype(np.float64))), 2), peak_db=round(float(20 * np.log10(np.abs(y).max() + 1e-9)), 2),
                       lr_corr=round(float(c), 3), side_over_mid_db=round(float(sm), 1))
    print(lay, report[lay], flush=True)
json.dump(report, open(f"{ROOT}/audio/v3/out/{tag}/report.json", 'w'), indent=1)
