"""浙江农林大学 sound design v21 for the picture of work/v16k/edit18.py (the living woodblock print), with the 奶蛙 engine (eng / snd).
  - one continuous air under the whole film (Wind 5, a clean stretch); each place adds only clean stretches of its own recordings
    with long crossfades, so the cuts never change the room abruptly; every bed arrives before its picture (先闻其声) and leaves
    with a long fade;
  - birds only from the singing phrases of "Birds FX 02", the whistles of "Birds FX 01" and one sparrow phrase, high-passed (their
    recordings hum), distant, in the gaps of the voice; no crows or gulls ("Country Day" and "Bird Seagull" are not used);
  - real foley for the print: paper, leaves, a water drop for every ink bloom, small bells for the carved gold, a far temple bell
    for 古; glints from real small bells retuned to the score's D, glows (bell() in design_helpers.py) on notes of D major;
  - pre-laps (the arriving sound reversed through a hall, swelling into each big transition) and a warm sub under the three big
    moments;
  - binaural only ("8D"): the camera's movements carried round and over the head.
Every designed transition has its own sound; the voice level is handled in mix_final.py; everything is local (library choices:
audio/v3/LIB_SCAN_v19.md, CLAP + spectrograms).
  ~/venvs/audio/bin/python audio/v3/design_zafu_v21.py [layout ...]  (default: stereo bin)  ->  audio/v3/pre21/<layout>/{FX,AMB}.wav,
  audio/v3/pre21_cues.json"""
import sys, os, json
import numpy as np, soundfile as sf
V8 = "~/Movies/NAIWA_BEYOND/audio/v8"; sys.path.insert(0, V8)
from eng import *
from snd import *
HERE = os.path.dirname(os.path.abspath(__file__))
END = 110.0
M = Mix(END + 2.0); CUES = []; SX = 32.0
def cue(t, bus, src, note, **kw): CUES.append(dict(t=round(float(t), 3), bus=bus, src=src, note=note, **kw))
exec(open(f"{HERE}/design_helpers.py", encoding="utf-8").read())
LEAD, TAIL = 1.2, 1.5
rng = np.random.default_rng(17)

def glow(t, f, dur=2.4, x=0.0, peak=-30.0, note=None):
    one(None, t, clip=bell(f, dur, bright=0.45), peak=peak, x=x, dist=2.5, rv=-10.0, sync=0.0, note=note)

def noise_swell(dur, lp_hz, shape=1.5, rise=False):
    n = int(dur * SR); s = lpf(rng.standard_normal(n).astype(np.float32), lp_hz)
    env = (np.linspace(0, 1, n) ** 2.5) if rise else np.sin(np.linspace(0, np.pi, n)) ** shape
    return (s * env.astype(np.float32)).astype(np.float32)

def drop():
    """an ink drop meeting paper: a tiny falling chirp and a soft tick"""
    n = int(0.16 * SR); tt = np.arange(n) / SR
    f = 1400 * np.exp(-tt * 18) + 260
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 28)
    s[:int(0.004 * SR)] += lpf(rng.standard_normal(int(0.004 * SR)).astype(np.float32), 5000) * 0.6
    return s.astype(np.float32)

def press(weight=1.0):
    """a wooden type pressed into paper: a low knock and a short paper crush"""
    n = int(0.22 * SR); tt = np.arange(n) / SR
    knock = np.sin(2 * np.pi * (180 + 60 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 30) * weight
    crush = lpf(rng.standard_normal(n).astype(np.float32), 3000) * np.exp(-tt * 45) * 0.5
    return (knock + crush).astype(np.float32)

def ticks(n_ticks, dur):
    """the 68 years drawn up the stem: a quick run of tiny ticks"""
    n = int(dur * SR); s = np.zeros(n, np.float32)
    for k in range(n_ticks):
        i = int(k / n_ticks * (n - 400)); s[i:i + 300] += (hpf(rng.standard_normal(300).astype(np.float32), 3000) * np.exp(-np.arange(300) / 40.0)).astype(np.float32) * (0.5 + 0.5 * (k % 10 == 0))
    return s

def crackle(dur, density=40):
    n = int(dur * SR); s = np.zeros(n, np.float32)
    for k in range(int(density * dur)):
        i = rng.integers(0, n - 600); a = rng.uniform(0.2, 1.0)
        s[i:i + 500] += (lpf(rng.standard_normal(500).astype(np.float32), 1800) * np.exp(-np.arange(500) / 90.0) * a).astype(np.float32)
    return s * np.sin(np.linspace(0, np.pi, n)).astype(np.float32) ** 0.6


def clapper(f=1650.0, w=1.0):
    """a bamboo clapper (竹梆): two hard resonances, very short"""
    n = int(0.12 * SR); tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 70) + 0.6 * np.sin(2 * np.pi * f * 2.37 * tt) * np.exp(-tt * 110)) * w
    s[:int(0.002 * SR)] += hpf(rng.standard_normal(int(0.002 * SR)).astype(np.float32), 2000) * 0.8
    return s.astype(np.float32)

def woodfish(f=520.0):
    """a wooden fish (木鱼): hollow, rounder, a little longer"""
    n = int(0.3 * SR); tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 22) + 0.35 * np.sin(2 * np.pi * f * 1.52 * tt) * np.exp(-tt * 35)
    return s.astype(np.float32)

def accel(t0, t1, n_hits, f=1650.0, end_hit=True):
    """a run of clapper hits that speeds up from t0 to t1 (things growing faster and faster), a wooden fish on the arrival"""
    u = np.linspace(0, 1, n_hits) ** 0.55
    for k, uu in enumerate(u):
        one(None, t0 + (t1 - t0) * uu, clip=clapper(f * (1 + 0.04 * (k % 3)), 0.55 + 0.45 * uu), peak=-38.0 + 6 * uu, x=float(np.sin(k * 1.7)) * 0.4,
            dist=1.4, rv=-12.0, sync=0.0, note="a clapper run speeds up" if k == 0 else None)
    if end_hit: one(None, t1 + 0.12, clip=woodfish(), peak=-30.0, x=0.0, dist=1.3, rv=-11.0, sync=0.0, note="arrival: the wooden fish")

def knife(dur=0.35):
    """a carving knife pushing through pear wood: a short rasping scrape with a pitched edge"""
    n = int(dur * SR); tt = np.arange(n) / SR
    s = bpf(rng.standard_normal(n).astype(np.float32), 1800, 5200) if 'bpf' in globals() else hpf(lpf(rng.standard_normal(n).astype(np.float32), 5200), 1800)
    grain = (np.sin(2 * np.pi * 38 * tt) > 0.2).astype(np.float32) * 0.5 + 0.5
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.7 * (0.6 + 0.4 * np.linspace(1, 0, n))
    return (s * grain * env).astype(np.float32)

def roller(dur=1.6):
    """a printing roller across the paper: a low continuous friction that swells and passes"""
    n = int(dur * SR); s = lpf(rng.standard_normal(n).astype(np.float32), 700)
    s += 0.3 * hpf(lpf(rng.standard_normal(n).astype(np.float32), 2400), 900)
    return (s * np.sin(np.linspace(0, np.pi, n)).astype(np.float32) ** 1.3).astype(np.float32)

def tear(dur=0.55):
    """paper torn open: a run of crisp crackles over a ripping noise"""
    n = int(dur * SR); s = hpf(lpf(rng.standard_normal(n).astype(np.float32), 7000), 1200) * np.linspace(1, 0.2, n).astype(np.float32) * 0.6
    for k in range(int(dur * 120)):
        i = int(rng.uniform(0, 0.85) * n); a = rng.uniform(0.3, 1.0)
        s[i:i + 160] += (hpf(rng.standard_normal(160).astype(np.float32), 2500) * np.exp(-np.arange(160) / 25.0) * a).astype(np.float32)
    return s.astype(np.float32)

def sand(dur, rise=False):
    """pigment grains in the air: a granular hiss that swells (or rises) and settles"""
    n = int(dur * SR); s = np.zeros(n, np.float32)
    for k in range(int(dur * 900)):
        i = rng.integers(0, n - 200); a = rng.uniform(0.1, 1.0)
        s[i:i + 120] += (hpf(rng.standard_normal(120).astype(np.float32), 4000) * np.exp(-np.arange(120) / 18.0) * a).astype(np.float32)
    env = (np.linspace(0, 1, n) ** 2.2) if rise else np.sin(np.linspace(0, np.pi, n)) ** 1.2
    return (s * env.astype(np.float32)).astype(np.float32)


PAPER = "A4 Printing Paper Rattle Page Turn Tail"
DRIP = "Water Drip"                                                  # 20.50-22.50: a drop into a quiet pool (B5)
LEAVES = "Leaves Bush Rustling.M"
GUST = ("Wind 4", 48.30, 50.70)                                      # the softest whoosh in the library: a real gust
SOFTWH = ("Whoosh 09", 0.22, 2.53)

# ---- rooms (reverb), following the picture
M.room("paper", 0.0, 9.2, "Hangar One", wet=-17.0, lp=3200)
M.room("valley", 9.2, 12.9, "Valley Forest", wet=-12.0)
M.room("soil", 12.9, 16.6, "Old Forest Long", wet=-13.0, lp=3000)
M.room("history", 16.6, 24.0, "Hangar One", wet=-14.0, lp=4200)
M.room("donghu", 24.0, 37.8, "Valley Forest", wet=-12.0)
M.room("tree", 37.8, 48.3, "Old Forest Long", wet=-11.0)
M.room("campus", 48.3, 66.8, "Valley Forest", wet=-12.0)
M.room("plate", 66.8, 69.4, "Hangar One", wet=-17.0, lp=3200)
M.room("campus2", 69.4, 101.2, "Old Forest Long", wet=-11.0)
M.room("title", 101.2, 110.0, "Opulent Ballroom", wet=-15.0, lp=5000)


def win(name, a, b, hp=None, lp=None):
    """a clean stretch of a library recording, stereo"""
    x = S(name, a, b, mono=False)
    if hp: x = hpf(x, hp)
    if lp: x = lpf(x, lp)
    return x


def wbed(name, a, b, t0, t1, level, hp=None, lp=None, fi=1.6, fo=1.8, note=None):
    """a bed from one clean stretch [a, b] of a recording, looped with long crossfades, faded in before t0 and out after t1"""
    bed(name, t0, t1, level=level, fi=fi, fo=fo, pre=fi, post=fo, clip=win(name, a, b, hp, lp), note=note)


def bird(name, a, b, t, peak, x=0.0, hp=1300, dist=5.0, rv=-8.0, note=None, lp=9000):
    """one sung phrase, high-passed (the bird recordings hum), far away in the room"""
    one(name, t, a=a, b=b, peak=peak, hp=hp, lp=lp, x=x, dist=dist, rv=rv, sync=0.0, fi=0.15, fo=0.7, stem="AMB", note=note)


def drip(t, peak=-40.0, x=0.0, note=None):
    """an ink bloom: one drop into a quiet pool"""
    one(DRIP, t, a=20.50, b=22.50, peak=peak, x=x, dist=1.6, rv=-10.0, sync="onset", fi=0.01, fo=0.9, note=note)


def gust(t, peak=-40.0, x=0.0, note=None, sp=1.0, lp=4500):
    one(GUST[0], t, a=GUST[1], b=GUST[2], peak=peak, x=x, dist=2.0, rv=-11.0, sync=0.0, fi=0.5, fo=0.9, sp=sp, lp=lp, note=note)


def softwh(t, peak=-40.0, x=0.0, note=None, lp=5000):
    one(SOFTWH[0], t, a=SOFTWH[1], b=SOFTWH[2], peak=peak, x=x, dist=1.8, rv=-11.0, sync=0.0, fi=0.35, fo=0.7, lp=lp, note=note)


def paper(t, peak=-40.0, x=0.0, a=0.03, b=1.28, note=None):
    one(PAPER, t, a=a, b=b, peak=peak, x=x, dist=1.2, rv=-12.0, sync="onset", fi=0.01, fo=0.35, note=note)


def leaves(t, a, b, peak=-44.0, x=0.0, az=None, note=None):
    one(LEAVES, t, a=a, b=b, peak=peak, x=x, az=az, dist=1.8, rv=-11.0, sync=0.0, fi=0.4, fo=0.8, note=note)


import pedalboard as _pb
_HALL = _pb.Pedalboard([_pb.Reverb(room_size=0.92, damping=0.35, wet_level=1.0, dry_level=0.0, width=1.0)])


def prelap(t_arrive, clip, dur=1.4, peak=-46.0, x=0.0, lp=7000.0, note=None):
    """a pre-lap: the arriving sound sent through a long hall, reversed, so its reverb swells up and lands on t_arrive"""
    c = np.concatenate([np.asarray(clip, np.float32)[: int(0.6 * SR)], np.zeros(int(dur * SR), np.float32)])
    w = _HALL(c[None, :], SR)[0][: int((dur + 0.6) * SR)][::-1].copy()
    w = lpf(w[-int(dur * SR):], lp); w = fade(w, dur * 0.7, 0.03)
    one(None, t_arrive - dur, clip=w.astype(np.float32), peak=peak, x=x, dist=2.6, rv=-14.0, sync=0.0, fi=0.0, fo=0.02, note=note)


def sub(t, peak=-34.0, rise=0.5, decay=2.4, note=None):
    """a warm sub swell (D1 + D2) under a big moment, felt more than heard"""
    n = int((rise + decay) * SR); tt = np.arange(n) / SR
    env = np.where(tt < rise, (tt / rise) ** 2, np.exp(-(tt - rise) / (decay / 3)))
    y = (np.sin(2 * np.pi * 36.71 * tt) + 0.55 * np.sin(2 * np.pi * 73.42 * tt)) * env
    one(None, t - rise, clip=y.astype(np.float32), peak=peak, x=0.0, dist=2.0, rv=None, sync=0.0, fi=0.0, fo=0.2, lfe=-8.0, note=note)


# ---- beds: one air under everything (Wind 5, a clean stretch; louder in the dive, the mist, the bamboo, the rise)
bed("Wind 5", 0.0, 110.0, pts=[(0, -60), (1.0, -56), (5.0, -54), (6.6, -52), (7.4, -46), (8.6, -48), (9.4, -44), (10.6, -50), (12.6, -54), (16.6, -55),
                               (24.0, -52), (27.4, -52), (37.8, -53), (42.6, -50), (44.0, -47), (48.3, -50), (52.5, -55), (56.0, -48),
                               (57.0, -52), (59.6, -51), (66.8, -58), (69.4, -54), (73.0, -55), (80.0, -51), (90.8, -48), (93.8, -49),
                               (97.0, -45), (101.2, -55), (106.0, -51), (110.0, -58)],
    fi=1.0, fo=2.0, pre=0.0, post=0.0, clip=win("Wind 5", 23.0, 41.0, hp=120), note="one continuous air under the whole film")
# places (clean stretches only; Forest Day windows chosen so the same phrase is never heard twice)
wbed("Forest Day", 18.0, 27.0, 10.3, 12.9, -46.0, lp=8000, note="the valley in colour")
wbed("Forest Day", 0.5, 4.0, 12.9, 16.4, -52.0, lp=5000, fi=1.2, note="soil: the forest floor, close and quiet")
wbed("Forest Day", 29.0, 36.0, 16.8, 24.0, -56.0, lp=5000, note="the old campus: far birds")
wbed("Forest Day", 45.0, 51.0, 24.0, 27.6, -50.0, lp=7000, note="today's gate: the ginkgo avenue in the morning")
wbed("WATRLap_Summer Tennessee Lake Dock Water Ripples Wake Wave Gentle 05 Distant_ESM_CPS", 10.0, 18.0, 27.6, 37.6, -51.0, hp=100, note="the East Lake")
wbed("Forest Day", 38.0, 42.0, 33.2, 37.8, -52.0, lp=8000, note="the library's shore grows green")
wbed(LEAVES, 0.49, 10.49, 37.8, 42.4, -47.0, note="the tree: leaves")
wbed("Forest Day", 11.0, 16.0, 38.6, 42.4, -53.0, lp=8000, note="birds in the great tree")
wbed("Wind 8", 5.0, 13.0, 42.6, 48.3, -44.0, hp=110, fi=2.0, note="wind in the bamboo leaves")
wbed("Forest Day", 53.0, 58.0, 48.6, 52.6, -52.0, lp=7000, note="the campus from above")
wbed("Forest Day", 61.0, 64.0, 52.6, 55.9, -55.0, lp=6000, note="一草一木: close quiet air")
wbed("Water Stream River 01", 17.0, 25.0, 56.7, 59.5, -45.5, hp=90, fi=1.0, note="一水: the island's water")
wbed("WATRLap_Summer Tennessee Lake Dock Water Ripples Wake Wave Gentle 05 Distant_ESM_CPS", 10.0, 18.0, 59.6, 66.6, -53.0, hp=100, note="the misty lake, the library")
wbed("Forest Day", 53.0, 58.0, 69.4, 73.0, -43.0, lp=10000, fi=1.2, note="the morning forest")
wbed("Forest Day", 18.0, 27.0, 73.2, 79.8, -54.0, lp=6000, note="the motto stone in its garden")
wbed("Forest Day", 29.0, 36.0, 80.0, 87.6, -51.0, lp=8000, note="the campus in its valley")
wbed("Water Stream River 01", 35.0, 43.0, 87.8, 90.8, -48.0, hp=90, note="the swans' lake")
wbed("Wind 8", 1.0, 7.0, 93.8, 101.0, -50.0, hp=110, note="托举: air through leaves, rising")

# ---- birds: sung phrases only, far, in the gaps of the voice
bird("Birds FX 02", 0.00, 2.80, 10.95, -45.0, x=-0.4, note="the valley in colour: a song")
bird("Birds FX 01", 0.00, 1.45, 24.35, -47.0, x=0.5, hp=1700, note="today's gate: whistles")
bird("Birds FX 02", 14.70, 17.20, 33.6, -46.0, x=-0.5, note="spring on the shore")
bird("Birds FX 02", 23.50, 26.00, 39.3, -47.0, x=0.45, note="the great tree")
bird("Birds FX 02", 47.90, 49.90, 49.7, -47.0, x=-0.3, note="the campus from above")
bird("Bird Sparrow", 0.00, 2.05, 63.7, -50.0, x=0.5, hp=1500, note="the library in the morning mist")
bird("Birds FX 01", 0.00, 1.45, 70.05, -45.0, x=0.55, hp=1700, note="the morning forest")
bird("Birds FX 02", 30.90, 33.30, 71.5, -46.0, x=-0.45, note=None)
bird("Birds FX 02", 39.90, 42.10, 82.3, -48.0, x=0.3, note="the campus in its valley")

# ---- 0 - 16.6: the little planet, the ink ridge, through the mist, the samara lands, it sprouts
glow(1.0, NOTE["A5"], 3.2, x=-0.1, peak=-38.0, note="一粒种子: the planet turns")
paper(3.35, -44.0, note="the planet starts to unroll")
gust(4.3, -42.0, note="it unrolls: a breath of air")
softwh(6.95, -40.0, lp=4000, note="the dive along the ink ridge")
gust(9.25, -38.0, sp=0.9, lp=3500, note="down into the mist")
one("Shimmer Motion", 10.15, a=0.09, b=3.5, sp=1.018, peak=-44.0, x=0.1, dist=2.2, rv=-9.0, sync=0.0, fi=0.4, fo=0.8, note="the valley appears, taking colour")
paper(10.45, -44.0, x=0.2, note="the colour tears in like paper")
one("Eagle Wing Flaps.M", 11.0, a=0.0, b=1.6, peak=-48.0, lp=3000, az=[(0.0, -30.0, 2.0), (1.6, 10.0, 1.4)], rv=-11.0, sync=0.0, fi=0.2, fo=0.5, note="the samara spins down")
glow(12.42, NOTE["D6"], 1.8, x=0.0, peak=-42.0, note="it lands")
glow(13.3, NOTE["A5"], 2.0, x=-0.1, peak=-41.0, note="萌芽: the shoot breaks out")
glow(14.3, NOTE["D6"], 2.4, x=0.05, peak=-41.0, note="生长: the first leaves open")
one("Shimmer Motion", 14.35, a=0.09, b=2.4, sp=1.018, peak=-50.0, x=0.0, dist=2.0, rv=-9.0, sync=0.0, fi=0.5, fo=0.9, note=None)
one("MAGMisc_Magic Christmas Bells 2_344 Audio_Christmas", 15.9, a=2.0, b=3.4, peak=-46.0, x=0.0, dist=1.8, rv=-10.0, sync=0.0, fi=0.3, fo=0.6, note="the seedling lifts off as grains of light")
one("Bell Clang Resonance Low.M", 16.42, a=0.0, b=5.0, peak=-41.0, x=0.0, dist=2.0, rv=-10.0, sync="onset", fo=2.0, note="the grains gather, the first old gate bleeds out (a bowl-like bell, E)")

# ---- 16.6 - 24.0: 天目林学院 -> its building -> 浙江林学院
softwh(17.75, -42.0, lp=3500, note="through the pines")
one("Shimmer Motion", 18.2, a=0.09, b=2.2, sp=1.018, peak=-46.0, x=0.2, dist=2.0, rv=-10.0, sync=0.0, fi=0.3, fo=0.6, note="the building rises out of its scaffolding")
leaves(19.95, 6.0, 8.0, -46.0, x=-0.3, note="trees grow in front of it")
softwh(21.7, -44.0, lp=4000, note="on to the 浙江林学院 gate")
paper(23.35, -40.0, note="today: the old print washes off")
glow(24.15, NOTE["D6"], 2.2, x=0.1, peak=-39.0, note="今天: today's gate is printed")
leaves(24.8, 14.5, 16.5, -48.0, az=[(0.0, -40.0, 2.0), (2.0, 40.0, 2.0)], note="ginkgo leaves drift past the gate")
glow(25.4, NOTE["A5"], 2.6, x=0.0, peak=-40.0, note="浙江农林大学 on the stone")

# ---- 24.0 - 37.8: the East Lake grows up
drip(26.95, -38.0, x=0.05, note="ink blooms from the end of the avenue: the land in 2000")
one("Water Stream River 01", 28.9, a=17.0, b=18.6, peak=-44.0, hp=90, x=0.0, dist=1.8, rv=-11.0, sync=0.0, fi=0.6, fo=0.8, note="the lake fills")
gust(29.85, -46.0, x=0.3, note="the cranes rise")
paper(30.8, -39.0, note="the building site washes off the sheet")
glow(31.95, NOTE["D6"], 3.4, x=0.0, peak=-37.0, note="the library is printed in its place")
glow(32.0, NOTE["A5"], 3.4, x=0.0, peak=-40.0, note=None)
gust(35.15, -46.0, x=0.0, note="a blue winter passes")
leaves(36.15, 6.0, 8.0, -47.0, x=0.4, note="autumn leaves")
one("MAGMisc_Magic Christmas Bells 2_344 Audio_Christmas", 37.2, a=5.0, b=6.4, peak=-46.0, x=0.0, dist=1.8, rv=-10.0, sync=0.0, fi=0.3, fo=0.6, note="today's woods lift off as grains of light")
one("Bell Clang Resonance Low.M", 37.68, a=0.0, b=4.0, peak=-43.0, x=0.0, dist=2.0, rv=-10.0, sync="onset", fo=1.8, note="they settle as one seedling")

# ---- 37.8 - 48.3: the seedling grows into the great maple; the bamboo
gust(38.7, -44.0, x=0.0, sp=0.85, note="the tree grows, the camera pulls back")
for t, xx in ((39.75, -0.4), (40.6, 0.4)):
    leaves(t, 14.5, 16.0, -45.0, x=xx, note="a sweep of falling leaves" if xx < 0 else None)
for k, t in enumerate((41.7, 41.85, 42.0)):
    glow(t, [NOTE["A5"] * 2, NOTE["D6"] * 2, NOTE["Fs5"] * 2][k], 1.2, x=-0.3 + 0.3 * k, peak=-43.0, note="硕果满枝" if k == 0 else None)
one(None, 42.72, clip=press(1.0), peak=-38.0, x=0.0, dist=1.0, rv=-11.0, sync=0.0, note="the bamboo shoot breaks the soil")
one(None, 42.78, clip=crackle(0.7, 34), peak=-46.0, x=0.0, dist=1.0, rv=-12.0, sync=0.0, note=None)
one("Shimmer Motion", 43.3, a=0.09, b=5.0, sp=1.018, peak=-46.0, x=0.0, dist=2.0, rv=-9.0, sync=0.0, fi=0.8, fo=1.4, note="the bamboo races up: light through the leaves")
gust(44.6, -46.0, x=0.0, sp=0.9, note=None)
leaves(46.6, 0.49, 2.0, -44.0, az=[(0.0, -50.0, 1.4), (1.5, 50.0, 1.4)], note="leaves at the top")
drip(47.95, -40.0, note="ink blooms: the library from above")

# ---- 48.3 - 73.0: the campus, the collage, mountain and water, 古 and 今, the plate, the forest
one("Paper Movement on a Roll.M", 52.6, a=1.44, b=1.87, peak=-44.0, x=0.0, dist=1.2, rv=-12.0, sync="onset", note="the campus becomes the first card")
for k, (L, (az0, el0)) in enumerate(zip([48.3 + j * 11.3 / 16 for j in range(7, 11)], [(75.0, 0.0), (-75.0, 0.0), (0.0, -45.0), (75.0, 10.0)])):
    one("Paper Movement on a Roll.M", L - 0.62, a=1.44, b=1.87, peak=-47.0, az=[(0.0, az0, 1.8, el0), (0.55, 0.0, 1.3, 0.0)], rv=-12.0, sync=0.0, fi=0.05, fo=0.2,
        note="a plant print slides in from where it comes" if k == 0 else None)
    paper(L - 0.08, -42.0 + (k % 2), x=0.0, a=0.03, b=0.6, note="a plant print lands" if k == 0 else None)
glow(55.75, NOTE["A5"] * 2, 1.4, x=0.2, peak=-42.0, note="snow white becomes cloud")
drip(56.48, -40.0, x=0.3, note="一水: water spreads from the island")
gust(59.1, -44.0, sp=0.9, note="the lake mist")
glow(59.75, NOTE["A5"] / 4, 3.2, x=-0.2, peak=-35.0, note="木化石: a deep old stone resonance")
one("Bell Church 2", 59.95, a=0.86, b=7.61, sp=0.9902, peak=-47.0, lp=3000, x=-0.3, dist=6.0, rv=-7.0, sync="onset", fo=2.5, note="古: a far temple bell (D)")
one("Shimmer Motion", 62.35, a=0.09, b=2.6, sp=1.018, peak=-48.0, x=0.2, dist=2.2, rv=-9.0, sync=0.0, fi=0.4, fo=0.8, note="今: the library in the morning mist")
paper(66.3, -40.0, note="the plate is printed")
for t in (67.25, 67.5, 67.75, 68.0, 68.25):
    one("Bell Small", t + 0.15, a=0.11, b=1.6, sp=0.9817 * (1.0 if t < 67.6 else 1.1225), peak=-52.0, x=0.0, dist=1.6, rv=-11.0, sync="onset", fo=0.8, note=None)
drip(68.95, -40.0, x=-0.5, note="ink spreads from the seed: a forest")
glow(70.0, NOTE["D6"], 2.8, x=0.3, peak=-43.0, note="light shafts in the forest")
glow(72.55, NOTE["D6"], 2.6, x=0.1, peak=-42.0, note="the forest light swells: the motto stone comes out of it")
glow(72.65, NOTE["A5"], 2.6, x=-0.1, peak=-45.0, note=None)

# ---- 73 - 108.6: the motto carved in gold, the university, welcome, lift, the title
CARVE = json.load(open(f"{HERE}/../../work/v18/carve_time.json"))["t"] if os.path.exists(f"{HERE}/../../work/v18/carve_time.json") else 74.57
for k in range(4):
    one("Bell Small", CARVE + 0.25 * k + 0.12, a=0.11, b=2.6, sp=0.9817 * [1.0, 1.1225, 0.8909, 1.0][k], peak=-45.0, x=0.45 - 0.3 * k, dist=1.4, rv=-10.0, sync="onset", fo=1.0,
        note="each carved character fills with gold, right to left" if k == 0 else None)
one("Shimmer Motion", 79.4, a=0.09, b=2.4, sp=1.018, peak=-48.0, x=0.0, dist=2.2, rv=-9.0, sync=0.0, fi=0.5, fo=0.9, note="the gold of the ginkgo spreads in")
drip(87.45, -41.0, x=0.2, note="ink blooms: the swans")
glow(87.6, NOTE["A5"], 1.8, x=0.0, peak=-42.0, note=None)
gust(90.3, -44.0, sp=0.9, note="the lake mist: the lake in snow")
leaves(91.7, 6.0, 7.6, -48.0, x=0.2, note="winter gives way to autumn")
softwh(93.55, -40.0, lp=5000, note="托举: the camera lifts")
one("Wing Flaps Multiple Birds Fly Away", 97.45, a=0.33, b=3.09, peak=-44.0, hp=150, az=[(0.0, -20.0, 2.0), (2.7, 30.0, 3.5)], rv=-9.0, sync=0.0, fi=0.1, fo=0.8, note="birds lift off: 托举你的未来")
drip(100.7, -41.0, note="the clouds turn to ink")
# ---- the birthday card: the rings grow 101.0-106.0, the name is printed on 浙江农林大学 (103.98), on 生日快乐 (106.0, the full chord)
# the rim is reached: a ring of light runs out, leaves lift off and swirl round the listener, glints sparkle above, a far bird answers
_grow = noise_swell(5.2, 2600, rise=True) * 0.5 + to_peak(S("Shimmer Motion", 0.09, 5.29, mono=True), 1.0) * np.linspace(0.2, 1.0, int(5.2 * SR)).astype(np.float32)
one(None, 100.9, clip=_grow, peak=-43.0, az=[(0.0, -40.0, 2.2, 0.0), (2.6, 40.0, 2.0, 15.0), (5.2, 0.0, 1.6, 25.0)], rv=-10.0, sync=0.0, fi=0.8, fo=0.4,
    note="the rings grow outward: a warm swell rising round the listener")
one("Bell Clang Resonance Low.M", 101.35, a=0.0, b=5.0, peak=-47.0, x=0.0, dist=2.4, rv=-9.0, sync="onset", fo=2.5, note="the pith: one low, deep tone")
one(None, 103.98, clip=press(0.6), peak=-42.0, x=0.0, dist=1.2, rv=-12.0, sync=0.0, note="浙江农林大学: the name is printed")
paper(103.98, -46.0, note=None)
one("Paper Movement on a Roll.M", 104.85, a=1.44, b=1.87, peak=-50.0, az=[(0.0, -30.0, 1.6), (0.4, 30.0, 1.6)], rv=-12.0, sync=0.0, fi=0.05, fo=0.2, note="1958 – 2026")
gust(105.75, -44.0, sp=0.9, lp=5000, note="生日快乐: a breath of air as the rim is reached")
for k, (f, d_, pk) in enumerate(((NOTE["D6"], 4.0, -38.0), (NOTE["A5"], 4.0, -41.0), (NOTE["Fs6"], 3.6, -44.0))):
    glow(105.98 + 0.03 * k, f, d_, x=(-0.25, 0.25, 0.0)[k], peak=pk, note="生日快乐: a chord of light in D, with the score's last chord" if k == 0 else None)
_lv = to_peak(fade(S(LEAVES, 6.0, 8.6, mono=True), 0.15, 0.9), -40.0)
one(None, 106.05, clip=_lv, peak=-42.0, az=[(0.0, -60.0, 1.8, 10.0), (1.2, 60.0, 1.8, 25.0), (2.6, 140.0, 2.4, 10.0)], rv=-11.0, sync=0.0, note="leaves lift off the rim and swirl")
for k in range(7):                                                         # glints: tiny bells scattered above, in D
    f_ = [1.0, 1.1225, 1.3348, 1.0, 1.4983, 1.1225, 1.3348][k]
    azg = [-50.0, 35.0, -15.0, 70.0, 10.0, -80.0, 55.0][k]
    one("Bell Small", 106.12 + 0.17 * k, a=0.11, b=1.8, sp=0.9817 * f_, peak=-50.0 - (k % 3), az=[(0.0, azg, 2.2, 35.0)], rv=-10.0, sync="onset", fo=0.9,
        note="glints of gold flash as the wave passes" if k == 0 else None)
one("Paper Handle.M", 106.15, a=3.41, b=3.97, peak=-52.0, x=0.0, dist=1.4, rv=-12.0, sync="onset", note="10.25: the sprigs unfold")
_lf = to_act(fade(loop_to(S(LEAVES, 14.5, 16.5, mono=True), int(3.5 * SR)), 0.6, 1.4), -52.0)
one(None, 106.6, clip=_lf, peak=-50.0, az=[(0.0, 30.0, 2.5, 50.0), (3.5, -40.0, 2.5, 40.0)], rv=-10.0, sync=0.0, note="leaves falling all round, above")
bird("Birds FX 02", 0.00, 2.80, 107.55, -49.0, x=0.5, note="a far morning bird answers")

# ---- pre-laps into the big transitions (the arriving sound, reversed through a hall), and a warm sub under the three big moments
_drip = S(DRIP, 20.50, 22.50); _bowl = S("Bell Clang Resonance Low.M", 0.0, 3.0); _shim = speed(S("Shimmer Motion", 0.09, 2.0), 1.018)
prelap(10.15, _shim, 1.3, -48.0, note="pre-lap: the valley's shimmer swells up through the mist")
prelap(16.42, _bowl, 1.6, -45.0, note="pre-lap: the bowl of the first old gate swells in reverse")
prelap(24.15, bell(NOTE["D6"], 1.0, bright=0.4), 1.2, -49.0, note="pre-lap: today's gate")
prelap(31.95, bell(NOTE["D6"], 1.0, bright=0.4), 1.2, -49.0, note="pre-lap: the library")
prelap(37.68, _bowl, 1.4, -46.0, note="pre-lap: the seedling")
for t_ in (26.95, 47.95, 56.48, 68.95, 87.45, 100.7):
    prelap(t_, _drip, 1.1, -47.0, note="pre-lap: a reversed drop into each ink bloom" if t_ == 26.95 else None)
prelap(72.55, bell(NOTE["D6"], 1.0, bright=0.4), 1.2, -49.0, note="pre-lap: the forest light")
prelap(101.35, _bowl, 1.5, -45.0, note="pre-lap: the birthday card")
sub(16.5, -36.0, 0.6, 2.6, note="sub: the first old gate appears")
sub(42.75, -33.0, 0.25, 2.2, note="sub: the bamboo shoot bursts out")
sub(106.0, -32.0, 0.8, 3.6, note="sub: 生日快乐, the full chord")

# ---- 8D (binaural only): the movement follows the camera
def orb(clip, t, keys, rv=-12.0, name='', stem='FX'):
    M.add(clip, t, stem=stem, az=keys, rv=rv, name=name, only='bin')
_sh = to_peak(bell(NOTE['A5'], 4.0, bright=0.6), -34.0)
orb(_sh, 1.0, orbit(0.2, 3.4, az0=-60.0, period=3.2, sense=1.0, el=20.0, dist=2.0, ramp=1.0), name='the little planet turns round you')
_gl = to_peak(S("MAGMisc_Magic Christmas Bells 2_344 Audio_Christmas", 8.0, 10.0), -40.0)
orb(_gl, 15.8, orbit(0.0, 1.8, az0=0.0, period=1.8, sense=1.0, el=10.0, dist=1.5, ramp=1.0), name='grains of light swirl round, into the first gate')
_bd = to_peak(fade(hpf(S("Birds FX 02", 14.70, 17.20, mono=True), 1300), 0.3, 0.8), -44.0)
orb(_bd, 49.2, orbit(0.0, 2.4, az0=-90.0, period=6.0, sense=1.0, el=40.0, dist=6.0, ramp=1.0), rv=-8.0, name='a bird above the campus')
_wr = to_act(fade(loop_to(hpf(S('Wind 5', 33.0, 41.0, mono=True), 120), int(4.0 * SR)), 0.8, 1.2), -40.0)
orb(_wr, 94.5, orbit(0.2, 3.6, az0=0.0, period=4.0, sense=1.0, el=30.0, dist=3.0, ramp=1.0), name='rising air turns round')
# the camera's own movements carried into 3D
_air = lambda a, b, lvl: to_act(fade(hpf(S('Wind 5', a, b, mono=True), 140), 0.5, 0.8), lvl)
orb(_air(23.0, 26.2, -39.0), 6.3, [(0.0, 0.0, 3.0, 0.0), (1.4, 0.0, 1.6, 55.0), (2.2, 160.0, 1.8, 35.0), (3.2, 180.0, 3.0, 10.0)], name='the dive: air rushes over your head and behind')
for t_, (a0, a1) in ((9.3, (-50.0, 140.0)), (58.95, (-40.0, 150.0)), (90.15, (50.0, -140.0))):
    orb(_air(27.0, 28.6, -41.0), t_, [(0.0, a0, 2.4, 5.0), (0.8, (a0 + a1) / 2, 1.2, 20.0), (1.6, a1, 2.6, 5.0)], name='through the mist: the air passes through you')
_gr = to_peak(S("MAGMisc_Magic Christmas Bells 2_344 Audio_Christmas", 4.0, 6.0), -42.0)
orb(_gr, 37.0, orbit(0.0, 1.8, az0=180.0, period=1.8, sense=-1.0, el=15.0, dist=1.5, ramp=1.0), name='today\'s woods lift off as grains round you')
_bm = to_act(fade(loop_to(hpf(S('Wind 8', 5.0, 13.0, mono=True), 200), int(5.2 * SR)), 0.8, 1.0), -42.0)
orb(_bm, 43.0, [(0.0, 20.0, 2.0, -35.0), (2.0, -20.0, 1.8, 10.0), (4.0, 10.0, 2.0, 55.0), (5.2, 0.0, 2.6, 70.0)], name='the bamboo: from your feet up over your head')
_wb = to_peak(fade(hpf(S("Birds FX 01", 0.0, 1.45, mono=True), 1700), 0.1, 0.4), -47.0)
orb(_wb, 71.0, [(0.0, -130.0, 6.0, 30.0)], rv=-8.0, name='a bird behind, up in the canopy')
orb(_wb, 72.3, [(0.0, 120.0, 7.0, 25.0)], rv=-8.0, name='another answers on the right')
_sw = to_act(fade(S('Water Stream River 01', 36.0, 39.0, mono=True), 0.6, 1.0), -48.0)
orb(_sw, 88.0, orbit(0.2, 2.6, az0=-90.0, period=5.0, sense=1.0, el=-25.0, dist=2.2, ramp=1.0), name='the lake laps round you')

out = f"{HERE}/pre21"
json.dump(CUES, open(f"{out}_cues.json", "w"), ensure_ascii=False, indent=1)
for lay in (sys.argv[1:] or ["stereo", "bin"]):
    st = M.render(lay, stems=("FX", "AMB"))
    d = f"{out}/{lay.replace('.', '')}"; os.makedirs(d, exist_ok=True)
    for s, y in st.items():
        sf.write(f"{d}/{s}.wav", y[: int((END + 1.0) * SR)], SR, subtype="FLOAT"); print(lay, s, "peak %.1f dBFS" % peak_db(y), flush=True)
print(len(CUES), "cues")
