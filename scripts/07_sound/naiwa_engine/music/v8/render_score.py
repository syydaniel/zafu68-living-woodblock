"""Render every part of the score on its own, dry, at 48 kHz:
  piano        Salamander concert grand (Yamaha C5, 16 velocity layers, CC BY 3.0 Alexander Holm) through FluidSynth
  everything else  the Sonatina Symphonic Orchestra's section recordings through music/v8/sso.py (CC Sampling Plus 1.0)
The expression curves of the score (controller 11, from compose_v13.py) become a gain curve on the part.
  ~/venvs/audio/bin/python music/v8/render_score.py [part ...]  ->  music/v8/score/dry/<part>.wav"""
import os, sys, subprocess, json
import numpy as np, soundfile as sf
R = "~/Movies/NAIWA_BEYOND"; S = os.environ.get("SCORE_DIR", f"{R}/music/v8/score"); SF = f"{R}/music/sf"; sys.path.insert(0, f"{R}/music/v8")
from sso import Inst, SR, sample, ROOT
PIANO = f"{SF}/SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"
SFZ = dict(drone="Strings - Performance/All Strings Sustain.sfz", strings="Strings - Performance/All Strings Sustain.sfz", violins="Strings - Performance/1st Violins Sustain.sfz", cello="Strings - Performance/Cello Solo Sustain.sfz",
           basses="Strings - Performance/Basses Sustain.sfz", trem="Strings - Performance/All Strings Tremolo.sfz", pizz="Strings - Performance/All Strings Pizzicato.sfz",
           stacc="Strings - Performance/All Strings Staccato.sfz", horns="Brass - Performance/Horns Sustain.sfz", brass="Brass - Performance/All Brass Sustain.sfz",
           trumpets="Brass - Performance/Trumpets Sustain.sfz", timp="Percussion/Timpani.sfz", harp="Concert Harp.sfz", celesta="Percussion/Celeste.sfz",
           choir="Chorus - Performance/Large Chorus.sfz", choirmel="Chorus - Performance/Mixed Chorus.sfz", guide="Strings - Performance/All Strings Sustain.sfz", bd="Percussion/Bass Drum & Snare.sfz", cym="Percussion/Cymbals & Tamtam.sfz")
# how the players start and leave a note (attack, release in seconds): bowed and blown lines join each other instead of starting hard
ENV = dict(drone=(1.1, 2.4), strings=(0.16, 1.0), violins=(0.11, 0.85), cello=(0.07, 0.6), basses=(0.22, 1.1), horns=(0.06, 0.75), brass=(0.05, 0.65), trumpets=(0.04, 0.55), choir=(0.45, 1.5), choirmel=(0.15, 0.6), guide=(0.06, 0.4), trem=(0.09, 0.7))
J = json.load(open(f"{S}/notes.json")); info = json.load(open(f"{S}/score.json")); TOTAL = info["end"] + 8.0
parts = sys.argv[1:] or [p for p in J["notes"] if J["notes"][p]]
os.makedirs(f"{S}/dry", exist_ok=True)
for p in parts:
    out = f"{S}/dry/{p}.wav"; notes = J["notes"][p]
    if p == "piano":
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.7", "-r", "48000", "-o", "synth.reverb.active=0", "-o", "synth.chorus.active=0", "-o", "synth.polyphony=512", "-T", "wav", "-O", "s24", "-F", out, PIANO, f"{S}/{p}.mid"], check=True)
        print(p, "Salamander grand (FluidSynth)", flush=True); continue
    I = Inst(SFZ[p], attack=ENV[p][0], release=ENV[p][1]) if p in ENV else Inst(SFZ[p]); lo = min(r["_lo"] for r in I.regions); hi = max(r["_hi"] for r in I.regions); nn = []
    if p == "cym":
        rollp = f"{ROOT}/Samples/Percussion/cymbal_roll-a-f.wav"; x, sr, _ = sample(rollp); m = np.abs(x).max(1); h = sr // 20
        e = np.array([m[i:i + h].max() for i in range(0, len(m) - h, h)]); tpk = float(np.argmax(e)) * h / sr                 # the roll's own peak
        for t, d, kind, v in notes:
            if kind == 49: nn.append((t - 0.02, 4.0, 55, v))                    # piatti
            elif kind == 55: nn.append((t + d - tpk, tpk + 3.0, 48, v))         # a roll that peaks where the swell should arrive
            else: nn.append((t - 0.03, 8.0, 57, v))                             # tam-tam
        print("   cymbal roll peaks %.2f s into its sample" % tpk)
    else:
        for t, d, k, v in notes:
            while k < lo: k += 12
            while k > hi: k -= 12
            nn.append((t, d, k, v))
    y = I.render(nn, TOTAL)
    cc = [(t, v) for t, num, v in J["cc"].get(p, []) if num == 11]
    if cc:                                                                  # expression: 100 = as played
        tt = np.arange(len(y)) / SR; g = np.interp(tt, [c[0] for c in cc], [c[1] for c in cc], left=100.0); y = y * ((np.maximum(g, 1.0) / 100.0) ** 1.5).astype(np.float32)[:, None]
    sf.write(out, y, SR, subtype="FLOAT")
    a = np.abs(y).max(1) > 1e-4; print(p, os.path.basename(SFZ[p]), len(nn), "notes, peak %.1f dBFS, active rms %.1f" % (20 * np.log10(np.abs(y).max() + 1e-9), 20 * np.log10(np.sqrt((y[a] ** 2).mean()) + 1e-9) if a.any() else -180), flush=True)
