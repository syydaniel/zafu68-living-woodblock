"""Score library for the ZAFU film from the 奶蛙 score v13 (compose_v13.py): the tune (P1-P4 and minor
forms), the Cue tempo map pinned to picture, mel / prog / lh / pad / tutti / held / swell, the drone part, and the MIDI writer.
exec(open(score_lib_v13_head.py)) at the top of a score, set FILM_END, write the music, then exec(score_lib_v13_writer.py)."""
import sys, os, json, struct, random, math
OUT = os.environ.get("SCORE_OUT", "~/Movies/ZAFU2/music/score_v8"); os.makedirs(OUT, exist_ok=True); rng = random.Random(1958)
PARTS = dict(drone=48, piano=0, cello=42, strings=48, violins=48, basses=43, horns=60, brass=57, trumpets=56, harp=46, timp=47, stacc=45, trem=44, choir=52, choirmel=53, guide=40, bd=None, cym=None)
NOTES = {p: [] for p in PARTS}; CCS = {p: [] for p in PARTS}; RAMPS = {p: [] for p in PARTS}; MARK = []; TEMPI = []
PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
QUAL = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'maj7': (0, 4, 7, 11), 'sus': (0, 5, 7), 'add9': (0, 2, 4, 7), 'madd9': (0, 2, 3, 7), '6': (0, 4, 7, 9), 'dim': (0, 3, 6)}
# the tune, in semitones from the tonic (which sits in octave 4), beats
Q = [(-5, 2), (-3, 1), (0, 1), (4, 3), (0, 1), (-3, 2), (-5, 1), (-8, 1), (-5, 4)]
A = [(0, 2), (4, 1), (7, 1), (9, 3), (7, 1), (4, 1), (0, 1), (-3, 2), (0, 4)]
BR = [(2, 1.5), (4, .5), (5, 2), (4, 1.5), (5, .5), (7, 2), (5, 1), (7, 1), (9, 1), (12, 1), (11, 3), (7, 1)]
CL = [(12, 2), (16, 1), (19, 1), (21, 2), (19, 1), (16, 1), (14, 1.5), (12, .5), (9, 1), (11, 1), (12, 4)]
Qm = [(-5, 2), (-2, 1), (0, 1), (3, 3), (0, 1), (-2, 2), (-5, 1), (-7, 1), (-5, 4)]              # the minor (pentatonic) forms
Am = [(0, 2), (3, 1), (5, 1), (7, 3), (5, 1), (3, 1), (0, 1), (-2, 2), (0, 4)]
HQ, HA, HBR, HCL = ['I', 'vi7', 'IVmaj7', 'Vsus V'], ['I', 'IV', 'vi IV', 'I'], ['ii7', 'I/3', 'IV vi', 'V'], ['I', 'IV', 'ii7 V7', 'I']
# the tune is ONE long sentence of sixteen bars, and its four phrases run into each other on upbeats (no phrase stops on a
# four-beat note before the close):
#   P1 asks (the question Q, with passing notes)          I  vi  IV-I6  V     ends on the fifth, an upbeat leads on
#   P2 answers (built on the answer A)                    I  IV  I6-ii  I     ends on the THIRD: not finished
#   P3 the climb: one cell in sequence up to the top      IV V   vi     V     ends on the dominant
#   P4 the return and the cadence                         I  IV  I64-V7 I     the only phrase that closes
P1 = [(-5, 2), (-3, 1), (0, 1), (4, 2.5), (2, .5), (0, 1), (-3, 1.5), (0, .5), (-5, 1), (-8, 1), (-5, 3), (-3, .5), (-1, .5)]
P2 = [(0, 2), (4, 1), (7, 1), (9, 2.5), (7, .5), (5, 1), (7, 1.5), (4, .5), (0, 1), (2, 1), (4, 3), (7, 1)]
P3 = [(9, 1.5), (7, .5), (9, 1), (12, 1), (14, 2.5), (12, .5), (9, 1), (12, 1.5), (9, .5), (7, 1), (4, 1), (7, 3), (9, .5), (14, .5)]
P4 = [(16, 2), (14, 1), (12, 1), (9, 2.5), (12, .5), (7, 1), (4, 1), (7, 1), (2, 1.5), (0, .5), (0, 4)]
P1m = [(-5, 2), (-2, 1), (0, 1), (3, 3), (0, 1), (-2, 1.5), (0, .5), (-5, 1), (-7, 1), (-5, 3), (-3, .5), (-1, .5)]      # (the upbeat climbs by the melodic minor: B, C sharp)
P2m = [(0, 2), (3, 1), (5, 1), (7, 2.5), (5, .5), (3, 1), (3, 1), (0, 1), (-2, 2), (0, 3), (7, 1)]
P3m = [(7, 1.5), (5, .5), (7, 1), (10, 1), (12, 2.5), (10, .5), (8, 1), (10, 1.5), (8, .5), (7, 1), (5, 1), (7, 3), (9, .5), (14, .5)]
P4m = [(15, 2), (14, 1), (12, 1), (8, 2.5), (12, .5), (5, 1), (3, 1), (7, 1), (2, 1.5), (0, .5), (0, 4)]
for _p in (P1, P2, P3, P4, P1m, P2m, P3m, P4m): assert abs(sum(n for _, n in _p) - 16) < 1e-9
HP1, HP2, HP3, HP4 = ['I', 'vi', 'IV I/3', 'Vsus V'], ['I', 'IV', 'I/3 ii', 'I'], ['IV', 'V', 'vi', 'Vsus V'], ['I', 'IV', 'I/5 V7', 'I']
HP1m, HP2m, HP3m, HP4m = ['i', 'VI', 'III', 'Vsus V'], ['i', 'III', 'VI VII', 'i'], ['i', 'VI', 'VII', 'Vsus V'], ['i', 'iv', 'i/5 V7', 'i']
HQm, HAm = ['i', 'VI', 'III', 'Vsus V'], ['i', 'III', 'VI VII', 'i']
ROM = {'I': (0, ''), 'ii7': (2, 'm7'), 'IV': (5, ''), 'IVmaj7': (5, 'maj7'), 'V': (7, ''), 'V7': (7, '7'), 'Vsus': (7, 'sus'), 'vi': (9, 'm'), 'vi7': (9, 'm7'), 'I/3': (0, ''),
       'i': (0, 'm'), 'VI': (8, ''), 'III': (3, ''), 'VII': (10, ''), 'iv': (5, 'm'), 'ii': (2, 'm'), 'I/5': (0, ''), 'i/5': (0, 'm')}


def chord(sym, key):
    """roman numeral in a key (pitch class of the tonic) -> (root pc, tones as pcs, bass pc)"""
    off, q = ROM[sym]; root = (key + off) % 12; tones = [(root + t) % 12 for t in QUAL[q]]; return root, tones, ((root + 4) % 12 if sym == 'I/3' else ((root + 7) % 12 if sym.endswith('/5') else root))


class Cue:
    """beats -> seconds through anchors [(beat, film time), ...], with a little breathing at the ends of phrases"""
    def __init__(self, anchors, phrase=16, rub=0.045):
        self.b = [a[0] for a in anchors]; self.t = [a[1] for a in anchors]; self.ph = phrase; self.rub = rub
        TEMPI.append(["%d-%d: %.0f" % (self.b[i], self.b[i + 1], 60.0 * (self.b[i + 1] - self.b[i]) / (self.t[i + 1] - self.t[i])) for i in range(len(self.b) - 1)])
    def T(self, beat):
        i = max(0, min(len(self.b) - 2, next((k for k in range(len(self.b) - 1) if beat < self.b[k + 1]), len(self.b) - 2)))
        t = self.t[i] + (beat - self.b[i]) * (self.t[i + 1] - self.t[i]) / (self.b[i + 1] - self.b[i])
        x = (beat % self.ph) / self.ph; u = min(x, 1.0 - x)             # distance to the nearest phrase boundary: slow into it, the downbeat a touch late, back in time after
        return t + (self.rub * math.cos(math.pi * u / 0.36) ** 2 if u < 0.18 else 0.0)


def note(part, p, t, d, v, human=0.010):
    if d <= 0.02 or t is None: return
    if part == 'piano': v = 50 + (v - 30) * 0.72
    NOTES[part].append((max(0.0, t + rng.uniform(-human, human)), d, int(p), int(max(1, min(127, round(v + rng.uniform(-2.5, 2.5)))))))


def cc(part, t0, t1, v0, v1, num=11):
    if t1 > t0 + 0.01: RAMPS[part].append((t0, t1, v0, v1))


def swell(part, t0, t1, lo=62, hi=104, peak=0.6):
    """a hairpin on a sustained phrase: up to the peak, down again"""
    tm = t0 + (t1 - t0) * peak; cc(part, t0, tm, lo, hi); cc(part, tm, t1, hi, lo + 6)


def ped(t0, t1): CCS['piano'].append((t0 + 0.03, 64, 127)); CCS['piano'].append((t1 - 0.06, 64, 0))


def mel(part, cue, b0, theme, key, vel, octave=0, arch=10, legato=1.04, upto=None, harm=None, mark=None, scale=1.0):
    """the tune on one instrument from beat b0 (scale = beats per written beat). harm (piano): add the third / sixth below
    on the long notes and the octave above at the top of the phrase."""
    b = b0; total = sum(x[1] for x in theme); acc = 0.0; tonic = 60 + key
    top = max(x[0] for x in theme)
    for off, n in theme:
        if upto is not None and acc >= upto - 1e-6: break
        x = acc / total; sh = arch * math.sin(math.pi * min(1.0, x * 1.2)) + (4 if n >= 3 else 0); t = cue.T(b); t1 = cue.T(b + n * scale); p = tonic + off + 12 * octave
        note(part, p, t, (t1 - t) * (legato if n < 3 else 1.0), vel + sh)
        if harm and n >= 2:                                             # the note a third or a sixth under it, from the chord of the moment
            below = sorted({(p - tone) % 12 for tone in harm(b)} & {3, 4, 8, 9, 5, 7}, key=lambda iv: (3, 4, 8, 9, 5, 7).index(iv))
            if below: note(part, p - below[0], t + 0.012, (t1 - t) * 0.98, vel + sh - 12)
        if harm and off == top and vel >= 52: note(part, p + 12, t + 0.006, (t1 - t), vel + sh - 7)
        b += n * scale; acc += n
    if mark: MARK.append(dict(t=round(cue.T(b0), 2), end=round(cue.T(b), 2), part=part, what=mark))
    return b


def prog(cue, b0, syms, key, beats_per_bar=4):
    """[(beat0, beat1, root pc, tones, bass pc), ...] for a list of bars ('IV vi' = two chords in the bar)"""
    out = []; b = b0
    for bar in syms:
        cs = bar.split(); d = beats_per_bar / len(cs)
        for s in cs:
            r, tn, bs = chord(s, key); out.append((b, b + d, r, tn, bs)); b += d
    return out


def harm_at(P):
    def f(b):
        for b0, b1, r, tn, bs in P:
            if b0 - 1e-6 <= b < b1: return [48 + x for x in tn] + [60 + x for x in tn]
        return []
    return f


def lh(cue, P, vel, style="broken", octave=2, v1=None):
    """the piano's left hand over a progression: broken chords in eighths (root, fifth, tenth ...), or held"""
    n = len(P)
    for k, (b0, b1, r, tn, bs) in enumerate(P):
        v = vel + ((v1 - vel) * k / max(1, n - 1) if v1 is not None else 0); base = 12 * (octave + 1) + bs
        if base < 36: base += 12
        ups = sorted((x - bs) % 12 for x in tn if (x - bs) % 12); third = base + ups[0] + 12; fifth = base + (7 if 7 in ups else ups[min(1, len(ups) - 1)]); top = base + 12
        ped(cue.T(b0), cue.T(b1))
        if style == "held":
            note('piano', base, cue.T(b0), cue.T(b1) - cue.T(b0), v, human=0.003); note('piano', fifth, cue.T(b0), cue.T(b1) - cue.T(b0), v - 6, human=0.003); note('piano', third, cue.T(b0), cue.T(b1) - cue.T(b0), v - 4, human=0.003)   # struck together, not rolled (a roll reads as a harp)
        else:
            pat = [base, fifth, third, fifth, top, fifth, third, fifth] if style == "broken" else [base, fifth, top, third, fifth, top, third + 12, top]
            step = 0.5 if style != "sixteenth" else 0.25; i = 0; b = b0
            while b < b1 - 1e-6:
                note('piano', pat[i % len(pat)], cue.T(b), (cue.T(b + step) - cue.T(b)) * 1.7, v + (6 if i % 8 == 0 else (2 if i % 4 == 0 else -3)), human=0.007); b += step; i += 1


_last = {}
def voiced(part, tones, lo, hi, n=4):
    """n notes of the chord between lo and hi, as close as possible to what the part played last (smooth voice leading)"""
    cand = [p for p in range(lo, hi + 1) if p % 12 in tones]; prev = _last.get(part)
    if not prev: step = max(1, len(cand) // n); out = cand[::step][:n]
    else:
        out = []
        for q in prev:
            best = min(cand, key=lambda p: (abs(p - q), p in out)); out.append(best)
        out = sorted(set(out))
        while len(out) < n:
            rest = [p for p in cand if p not in out]
            if not rest: break
            out.append(min(rest, key=lambda p: min(abs(p - q) for q in out))); out.sort()
    _last[part] = out; return out


def pad(part, cue, P, vel, lo=52, hi=76, n=4, bass=True, hair=True, v1=None, bvel=None, deep=False):
    m = len(P)
    for k, (b0, b1, r, tn, bs) in enumerate(P):
        v = vel + ((v1 - vel) * k / max(1, m - 1) if v1 is not None else 0); t0, t1 = cue.T(b0), cue.T(b1)
        for p in voiced(part, tn[:3], lo, hi, n): note(part, p, t0, t1 - t0 + 0.12, v)
        if bass:
            note('basses', 36 + bs, t0, t1 - t0 + 0.05, (bvel or v) + 2)
            if deep and 24 + bs >= 28: note('basses', 24 + bs, t0, t1 - t0 + 0.05, (bvel or v) + 2)
    if hair and P: swell(part, cue.T(P[0][0]), cue.T(P[-1][1]), 58, 100)


def harp_up(t, tones, lo=38, hi=88, dur=0.5, vel=62):
    ps = [p for p in range(lo, hi + 1) if p % 12 in tones]
    for k, p in enumerate(ps): note('harp', p, t - dur + dur * k / len(ps), 1.8, vel + 14 * k / len(ps), human=0)


def roll(t0, t1, p=38, v0=25, v1=90, rate=11.0):
    n = max(2, int((t1 - t0) * rate))
    for i in range(n): note('timp', p, t0 + (t1 - t0) * i / n, 0.16, v0 + (v1 - v0) * (i / (n - 1)) ** 1.6, human=0.004)


def cym(t, v=90, drum=True):
    NOTES['cym'].append((t, 4.0, 49, v))
    if drum: NOTES['bd'].append((t, 2.0, 48, min(127, v + 6)))
def cymroll(t0, t1, v=80): NOTES['cym'].append((t0, t1 - t0, 55, v))
def tam(t, v=100): NOTES['cym'].append((t, 8.0, 57, v))
def clip(t_cut, t_from):
    """everything that began after t_from stops at t_cut (the dead stops)"""
    for part in NOTES: NOTES[part] = [(t, min(d, t_cut - t), p, v) if t_from <= t < t_cut and t + d > t_cut else (t, d, p, v) for t, d, p, v in NOTES[part]]
def tutti(cue, P, vel, parts=('strings', 'brass', 'choir'), timp=True):
    for part in parts: pad(part, cue, P, vel - (8 if part != 'strings' else 0), lo=(48 if part == 'brass' else 52), hi=(69 if part == 'brass' else 79), n=4, bass=(part == 'strings'), hair=False, bvel=vel + 4, deep=True)
    if timp:
        for b0, b1, r, tn, bs in P: note('timp', 36 + bs if bs >= 2 else 48 + bs, cue.T(b0), 0.6, vel)



def held(part, ps, t0, t1, v, lo_cc=None, hi_cc=None):
    """a held chord / note on a part, with an optional hairpin"""
    for p in ps: note(part, p, t0, t1 - t0, v)
    if lo_cc is not None: swell(part, t0, t1, lo_cc, hi_cc)

