"""ZAFU copy of the 奶蛙 score mixer (mix_score.py). The score in a hall: the dry parts of the score (<SCORE_DIR>/dry/<part>.wav)
seated like an orchestra and placed in ONE real concert hall (an Apple B-format impulse response, HALL below: decoded to stereo,
and all round the listener in 5.1), with the same engine as the film's sound (audio/v8/eng.py), so music and picture share one space model.
  strings wide across the stage, first violins left, celli and basses right, horns left behind, brass right behind,
  harp far left, celesta right, timpani at the back, the piano in front in the middle (its own stereo image kept).
Depth = how much of the part goes into the hall. Cymbals come from the sound library (a real crash; reversed for swells).
  SCORE_DIR=... ~/venvs/audio/bin/python music/mix_zafu.py  ->  <SCORE_DIR>/mx_stereo.wav, mx_51.wav, ride.npy, groups/*.wav"""
import sys, os, json
import numpy as np, soundfile as sf
R = "~/Movies/NAIWA_BEYOND"; sys.path.insert(0, f"{R}/audio/v8")
from eng import *
SC = os.environ.get("SCORE_DIR", "~/Movies/ZAFU2/music/score"); info = json.load(open(f"{SC}/score.json")); FILM_END = info["end"]; DUR = FILM_END + 6.0
#          target level of the part's loudest ten seconds (dB), azimuth, width (None = a point), hall send dB, low-pass
SEAT = dict(piano=(-19.0, 0.0, 28.0, -10.0, None), strings=(-16.0, 0.0, 50.0, -8.5, None), violins=(-15.5, -26.0, 24.0, -8.5, None), trem=(-19.0, 0.0, 46.0, -10.0, None),
            stacc=(-18.5, 4.0, 40.0, -13.0, None), cello=(-19.5, 18.0, 10.0, -9.0, None), basses=(-19.0, 34.0, 12.0, -13.0, None),
            horns=(-17.0, -20.0, 16.0, -6.0, None), brass=(-15.5, 22.0, 26.0, -8.0, None), trumpets=(-18.0, 10.0, 14.0, -8.0, None), choir=(-18.5, 0.0, 42.0, -5.0, None), choirmel=(-16.5, 0.0, 34.0, -5.0, None), vocal=(-14.0, 0.0, 16.0, -10.0, None),
            harp=(-30.0, -40.0, 8.0, -9.0, None), timp=(-18.5, 6.0, 12.0, -10.0, None), bd=(-21.0, -4.0, 10.0, -11.0, None), cym=(-22.0, -10.0, 26.0, -7.0, None))
SEAT.update(songpiano=(-15.0, 0.0, 30.0, -11.0, None))                     # the ending song's piano (music/v8/song11, laid in by make_song_parts())
SEAT.update(drone=(-25.0, 0.0, 80.0, -3.0, 4500.0))                           # the ambient bed (pedal fifths), far, wide, dark
SEAT.update(song=(-14.0, 0.0, 60.0, -20.0, None))                          # the sung ending song (ACE-Step take, its own room; a little of the hall to join the film), music/v8/lay_song.py
# ZAFU: parts that only play in quiet scenes must not be raised to a loud target (the night hush, the motto pulse)
SEAT.update(trem=(-30.0, 0.0, 46.0, -8.0, None), stacc=(-24.0, 4.0, 40.0, -12.0, None), brass=(-18.5, 22.0, 26.0, -8.0, None))
# the bass drum and the cymbal only sound a few times (three accents, a few swells): a sparse part measured by its loudest ten
# seconds would be raised far too much (+18 dB), so they are seated by ear against the strings
SEAT.update(bd=(-33.0, -4.0, 10.0, -11.0, None), cym=(-33.0, -10.0, 26.0, -7.0, None))
SEAT = {k: v for k, v in SEAT.items() if os.path.exists(f"{SC}/dry/{k}.wav")}
GROUP = dict(piano="piano", strings="strings", violins="strings", trem="strings", pizz="strings", stacc="strings", cello="strings", basses="strings", horns="brass", brass="brass", trumpets="brass",
             choir="choir", choirmel="choir", vocal="vocal", harp="colour", celesta="colour", timp="percussion", bd="percussion", cym="percussion")
GROUP.update(songpiano="piano", song="vocal"); GROUP.update(drone="strings")
# a warm, long hall (measured: Opulent Ballroom decays in 3.3 s at 250 Hz, 3.2 s at 1 kHz, 1.1 s at 8 kHz; Prince Hall Three: 2.1 s and thin in
# the bass), 22 ms before the hall answers so the players stay clear in front of it, the return rolled off above 8.5 kHz
HALL = os.environ.get("HALL", "Opulent Ballroom"); WET = float(os.environ.get("WET", "-6.5")); PRE = float(os.environ.get("PRE", "0.022"))


def top10(x):
    m = x.mean(1); h = SR; e = np.array([np.sqrt((m[i:i + h] ** 2).mean()) for i in range(0, len(m) - h, h)]); return 20 * np.log10(np.sort(e)[-10:].mean() + 1e-9)


_G = {}
def build(only=None):
    M = Mix(DUR); M.room("hall", -1.0, DUR + 60, HALL, wet=WET, hp=110.0, pre=PRE, lp=8500.0)
    for p, (tgt, az, w, rv, lp) in SEAT.items():
        if only and GROUP[p] != only: continue
        x, sr = sf.read(f"{SC}/dry/{p}.wav", dtype="float32", always_2d=True); assert sr == SR; x = x[: int(DUR * SR)]
        if p not in _G: _G[p] = tgt - top10(x); print("   %-9s gain %+.1f dB" % (p, _G[p]), flush=True)
        x = x * db(_G[p])
        if lp: x = lpf(x, lp, 1)
        M.add(x[:, :2], 0.0, stem="MX", az=az, width=w, rv=rv, room="hall", name=p)      # the sections keep their own stereo picture, narrowed and turned to their seat
    return M


def ride(y, knee=-19.0, ratio=0.55, top=10.0, ceil=-14.0, down=0.5, most=8.0):
    """the conductor's hand: passages quieter than the knee are brought up by half the distance (at most `top` dB), slowly
    (2.5 s windows), so the solo piano of the quiet scenes is heard over the wind and the climaxes keep their size"""
    m = y[:, :2].mean(1); h = SR // 4; w = 10
    e = np.array([np.sqrt((m[i:i + h] ** 2).mean() + 1e-14) for i in range(0, len(m) - h, h)]); e = np.sqrt(np.convolve(e ** 2, np.ones(w) / w, "same")); L = 20 * np.log10(e + 1e-9)
    g = np.where(L < -62.0, 0.0, np.clip((knee - L) * ratio, 0.0, top)) - np.clip((L - ceil) * down, 0.0, most)       # and the loudest pages a little down (the limiter is not a conductor)
    g = np.convolve(np.pad(g, (8, 8), mode="edge"), np.hanning(17) / np.hanning(17).sum(), "same")[8:-8]
    return np.interp(np.arange(len(m)), np.arange(len(g)) * h + h / 2, g).astype(np.float32)


if __name__ == "__main__":
    M = build(); n = int((FILM_END + 4.0) * SR); G = None
    for lay in ("stereo", "5.1"):
        y = M.render(lay, stems=("MX",))["MX"][:n]
        if G is None: G = db(ride(y)); np.save(f"{SC}/ride.npy", G[::480])
        y = y * G[:, None]; sf.write(f"{SC}/mx_{lay.replace('.', '')}.wav", y, SR, subtype="FLOAT")
        print(lay, "peak %.1f dBFS" % peak_db(y), "rms %.1f" % (20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)), (("L/R corr %.2f, side/mid %.1f dB" % width_stats(y)) if lay == "stereo" else ""), flush=True)
    os.makedirs(f"{SC}/groups", exist_ok=True)
    for g in ("piano", "strings", "brass", "choir", "vocal", "colour", "percussion"):
        y = build(g).render("stereo", stems=("MX",))["MX"][:n] * G[:, None]; sf.write(f"{SC}/groups/{g}.wav", y, SR, subtype="FLOAT"); print("group", g, "peak %.1f" % peak_db(y), flush=True)
