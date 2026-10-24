"""ZAFU copy of the 奶蛙 binaural score mixer: the score for headphones in 3D, with sound arriving from all round the listener
("8D"). The same dry parts, balance and hall as mix_zafu.py, but every section is a pair of sources at its own
place round the listener, with height (eng.py layout "bin": Apple's HRTF, the hall decoded to twelve directions round and above):
  you sit in the middle of the orchestra: piano just in front, strings spread wide in front, violins left, celli and basses right,
  horns behind left and brass behind right, a little up; the shimmer (tremolo) behind and high; the chorus a dome over the sides
  and the back; the drone all round, turning very slowly.
"8D" turns (ORBITS): at a few peaks a section goes round the head (eng.orbit), the rest stays where it sits.
  SCORE_DIR=... ~/venvs/audio/bin/python music/mix_zafu_bin.py  ->  <SCORE_DIR>/mx_bin.wav, groups_bin/*.wav
(uses <SCORE_DIR>/ride.npy and mx_stereo.wav from mix_zafu.py: run that first)"""
import sys, os
import numpy as np, soundfile as sf
R = "~/Movies/NAIWA_BEYOND"; sys.path.insert(0, f"{R}/audio/v8"); sys.path.insert(0, "~/Movies/ZAFU2/music")
from eng import *
import mix_zafu as MS

SC = MS.SC; DUR = MS.DUR
#            azimuth (+ right), half width, elevation (+ up)
# you sit INSIDE the orchestra, so the space behind the head is filled too. The tune stays in front (piano, violins, celli); the harmony
# and the air go round and behind: horns behind left, brass behind right, the tremolo behind and high, the chorus a dome over the sides and the back
SEAT3D = dict(piano=(0.0, 22.0, -5.0), strings=(0.0, 80.0, 5.0), violins=(-45.0, 16.0, 8.0), trem=(180.0, 70.0, 30.0), stacc=(0.0, 50.0, 0.0),
              cello=(40.0, 10.0, -5.0), basses=(60.0, 12.0, -10.0), horns=(-125.0, 18.0, 12.0), brass=(125.0, 24.0, 12.0), trumpets=(150.0, 12.0, 20.0),
              choir=(0.0, 120.0, 35.0), choirmel=(180.0, 40.0, 30.0), vocal=(0.0, 12.0, 0.0), harp=(-70.0, 8.0, 20.0), timp=(0.0, 14.0, -5.0),
              bd=(0.0, 10.0, -10.0), cym=(180.0, 30.0, 35.0), songpiano=(0.0, 30.0, 0.0), drone=(90.0, 90.0, 0.0), song=(0.0, 45.0, 0.0))
# (part, start s, end s, period s, sense, elevation) in film seconds: the roots (strings), the bamboo (horns), the night sky
# (chorus dome turns, shimmer the other way), the finale (chorus round once)
ORBITS = [("strings", 28.6, 36.0, 6.0, 1.0, 5.0),                                     # the roots: the strings turn once, slowly
          ("horns", 42.6, 48.0, 5.0, 1.0, 15.0),                                       # 像一株竹, 不断超越: the horns go round, rising
          ("choir", 59.6, 66.8, 9.0, 1.0, 35.0), ("trem", 59.6, 66.8, 7.0, -1.0, 40.0),   # the stars: a dome turning over the head
          ("choir", 93.8, 104.0, 8.0, 1.0, 30.0)]                                      # 托举你的未来: the chorus round, up into the title
DRONE_TURN = 40.0                                                                      # the drone turns once in 40 s, all film long


def path_for(part, az0, el):
    orb = [o for o in ORBITS if o[0] == part]
    if part == "drone":
        T = np.arange(0.0, DUR + 1.0, 0.5); return [(float(t), az0 + 360.0 * t / DRONE_TURN, 2.5, el) for t in T]
    if not orb: return az0
    keys = [(0.0, az0, 2.5, el)]; a = az0
    for _, t0, t1, per, sense, e in sorted(orb, key=lambda o: o[1]):
        k = orbit(t0, t1, az0=a, period=per, sense=sense, el=e, dist=2.5)
        keys += [x for x in k[1:] if x[0] >= t0 - 0.2]; a = k[-1][1]
        keys.append((t1 + 2.5, a, 2.5, el))                                            # back to its seat's height after the turn
    return keys


def build3d(only=None):
    M = Mix(DUR); M.room("hall", -1.0, DUR + 60, MS.HALL, wet=MS.WET, hp=110.0, pre=MS.PRE, lp=8500.0)
    for p, (tgt, az, w, rv, lp) in MS.SEAT.items():
        if only and MS.GROUP[p] != only: continue
        x, sr = sf.read(f"{SC}/dry/{p}.wav", dtype="float32", always_2d=True); assert sr == SR; x = x[: int(DUR * SR)]
        if p not in MS._G: MS._G[p] = tgt - MS.top10(x)
        x = x * db(MS._G[p])
        if lp: x = lpf(x, lp, 1)
        a3, w3, e3 = SEAT3D.get(p, (az, w, 0.0))
        M.add(x[:, :2], 0.0, stem="MX", az=path_for(p, a3, e3), width=w3, el=e3, rv=rv, room="hall", name=p)
    return M


if __name__ == "__main__":
    n = int((MS.FILM_END + 4.0) * SR); g = np.load(f"{SC}/ride.npy"); G = np.interp(np.arange(n), np.arange(len(g)) * 480, g).astype(np.float32)
    y = build3d().render("bin", stems=("MX",), verbose=True)["MX"][:n] * G[:, None]
    import pyloudnorm as pyln                                                          # keep the stereo balance: the 3D score as loud as the stereo one
    st, _ = sf.read(f"{SC}/mx_stereo.wav", dtype="float32"); mt = pyln.Meter(SR); LM = mt.integrated_loudness(st[:n].astype(np.float64)) - mt.integrated_loudness(y.astype(np.float64))
    G = G * db(LM); y = y * db(LM); print("bin matched to the stereo score: %+.2f dB" % LM, flush=True)
    sf.write(f"{SC}/mx_bin.wav", y, SR, subtype="FLOAT"); print("bin peak %.1f dBFS" % peak_db(y), "L/R corr %.2f, side/mid %.1f dB" % width_stats(y), flush=True)
    os.makedirs(f"{SC}/groups_bin", exist_ok=True)
    for grp in ("piano", "strings", "brass", "choir", "vocal", "colour", "percussion"):
        y = build3d(grp).render("bin", stems=("MX",))["MX"][:n] * G[:, None]; sf.write(f"{SC}/groups_bin/{grp}.wav", y, SR, subtype="FLOAT"); print("group", grp, "peak %.1f" % peak_db(y), flush=True)
