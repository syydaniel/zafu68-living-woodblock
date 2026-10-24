"""浙江农林大学 score v10, timed to the picture of work/v16k/edit18.py, written with the 奶蛙 score v13 machinery
(music/score_lib_v13_head.py):
  - ONE tune of four phrases (P1 asks, P2 answers, P3 climbs, P4 closes), long sentences that run across the cuts; the full close
    only once, on the emblem
  - FIVE paragraphs that follow the story, inside each the accompaniment never stops; a bed (soft far fifths, the drone) glues them
  - no celesta, no mallets, no timpani rolls, no harp runs (they read as mallets); every sustained note swells; strings move by
    the nearest voices; the piano plays with two hands
  - the particle transitions are the picture's, not the music's: the music does not hit every one, it carries through them;
    the sound design gives each its air and chime
Key D major (warm morning gold); one tempo family (68-80 BPM).

  film s       picture / narration                            music
  0-3.8        一粒种子会如何成长？ (words of light)          the bed from nothing; the tune's first four notes, piano alone
  3.8-16.6     the seed flies down to 天目山, the sprout       P1 (asks) on piano, left hand held, then strings creep in
  16.6-27.4    the gates of history, today's gate             P2 (answers) on piano, a cello line under it; 今天 swells to I as the
                                                              iron gate blows away as leaves (23.4-24.6)
  27.4-42.6    the East Lake site, roots, 像一棵树             P3 (the climb) on celli over flowing piano, rising with the buildings,
                                                              strings growing, violins join
  42.6-48.3    像一株竹：坚韧不拔，不断超越                    the horns take the top of the climb; IV - Vsus - V held into ...
  48.3-59.6    整个校园就是一座植物园, leaves, the lake         ... I: P1 again, bright, violins + piano broken chords (the arrival)
  59.6-66.8    博览古今，学参天地 (stars)                      hush: high strings, the chorus, single high piano notes
  66.8-73.0    研究一粒种子 (DNA), 守护森林                     P2's second half on violins, a crescendo into
  73.0-80.0    求真敬业，肯干实干能干                          a firm pulse (low strings, brass); three accents where the narrator
                                                              says 肯干 / 实干 / 能干 (76.9 / 77.8 / 78.85)
  80.0-93.8    这是一所大学…欢迎你来这里扎根、生长             P3 in full (horns, strings, chorus), under the voice
  93.8-108.6   托举你的未来, 浙江农林大学                       P4, everyone, the only full close, on the emblem; the last chord rings
  SCORE_OUT=music/score_v10 python3 music/score_zafu_v10.py   ->  <SCORE_OUT>/{<part>.mid, notes.json, score.json}"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{HERE}/score_lib_v13_head.py", encoding="utf-8").read())
FILM_END = 108.6
D = 2

# the picture's clock: (beat, film second)
c = Cue([(0, 0.0), (4, 3.8), (20, 16.6), (36, 27.4), (52, 42.6), (58, 48.3), (74, 59.6), (82, 66.8), (90, 73.0), (99, 80.0), (115, 93.8),
         (131, 106.0), (135, 110.0)], phrase=16, rub=0.035)

# ================================================================ A. the seed and its history (0 - 27.4)
for k, (off, bb, d) in enumerate(((-5, 0.6, 1.2), (-3, 1.6, 0.8), (0, 2.4, 0.8), (4, 3.2, 3.0))):          # the tune's first four notes, alone
    note('piano', 62 + off + 12, c.T(bb), c.T(bb + d) - c.T(bb), 38 + 4 * k, human=0)
ped(c.T(0.5), c.T(4.2)); MARK.append(dict(t=0.5, end=3.8, part="piano", what="一粒种子: the tune's first four notes, alone"))
P = prog(c, 4, HP1, D)
lh(c, P, 30, "held")
mel('piano', c, 4, P1, D, 44, octave=1, arch=7, harm=harm_at(P), mark="the seed flies to 天目山, the sprout: P1 asks (piano)")
pad('strings', c, prog(c, 12, ['IV I/3', 'Vsus V'], D), 22, lo=55, hi=74, n=3, bvel=24, v1=34)               # strings creep in under the sprout
P = prog(c, 20, HP2, D)
lh(c, P, 36, "broken")
mel('piano', c, 20, P2, D, 50, octave=1, arch=8, harm=harm_at(P), mark="the gates of history: P2 answers (piano)")
mel('cello', c, 20, [(-12, 4), (-15, 4), (-8, 2), (-10, 2), (-12, 4)], D, 46, octave=0, arch=6)              # a slow line under it
swell('cello', c.T(20), c.T(36), 60, 96)
pad('strings', c, P, 30, lo=55, hi=76, n=4, v1=44)
t0, t1 = 23.4, 24.6                                                                                           # 今天: the iron gate blows away as leaves
tb = c.T(32)                                                                                                    # the bar of I
for p in (64, 67, 71, 76): note('violins', p, t0, tb - t0 + 0.05, 52)                                            # ii as the light crosses ...
for p in (62, 66, 69, 74, 78): note('violins', p, tb, 1.9, 58)                                                   # ... I as it arrives
cc('violins', t0, tb, 30, 100); cc('violins', tb, tb + 1.9, 100, 40)
MARK.append(dict(t=t0, end=t1, part="violins", what="今天，它是浙江农林大学: the leaves blow, a swell into I"))

# ================================================================ B. 扎根 / 像一棵树 / 像一株竹 (27.4 - 48.3): the climb
P = prog(c, 36, ['IV', 'V', 'vi', 'Vsus V'], D)
lh(c, P, 40, "broken", v1=56)
pad('strings', c, P, 34, lo=52, hi=74, n=4, v1=60, deep=True)
mel('cello', c, 36, P3, D, 54, octave=-1, arch=12, mark="roots, 像一棵树: P3 climbs (celli)")
swell('cello', c.T(36), c.T(52), 62, 116, peak=0.9)
mel('violins', c, 44, P3[6:], D, 50, octave=0, arch=8)                                                       # the violins join the second half of the climb (the tree)
swell('violins', c.T(44), c.T(52), 50, 104, peak=0.95)
# 像一株竹 (42.6 - 48.3): the horns hold the top of the climb, IV - Vsus - V, and the strings lift into the arrival
P = prog(c, 52, ['IV', 'Vsus V'], D, 3)
pad('strings', c, P, 62, lo=55, hi=79, n=5, deep=True, v1=74)
held('horns', [57, 62, 69], c.T(52), c.T(55), 66); held('horns', [57, 64, 69], c.T(55), c.T(56.5), 72); held('horns', [57, 61, 64, 69], c.T(56.5), c.T(58) + 0.15, 78)
swell('horns', c.T(52), c.T(58), 60, 110, peak=0.92)
mel('violins', c, 52, [(12, 2), (14, 1), (19, 3)], D, 62, octave=0, arch=6, mark="像一株竹，不断超越: the violins reach the top")
lh(c, P, 48, "broken", v1=58)
cymroll(c.T(56.4), c.T(58) + 0.02, 66)                                                                       # a soft cymbal swell into the arrival (no roll on drums)

# ================================================================ C. the botanical garden, learning, stars, research (48.3 - 73.0)
P = prog(c, 58, HP1, D)
lh(c, P, 46, "broken")
pad('strings', c, P, 40, lo=55, hi=79, n=4, v1=36)
mel('violins', c, 58, P1, D, 58, octave=1, arch=8, mark="整个校园就是一座植物园: arrival on I, P1 bright (violins + piano)")
mel('piano', c, 58, P1, D, 48, octave=1, arch=6, harm=harm_at(P))
swell('violins', c.T(58), c.T(74), 56, 100, peak=0.4)
# 博览古今，学参天地 (59.6 - 66.8): hush. The high strings hold, the chorus breathes, single high piano notes like stars
P = prog(c, 74, ['IV', 'I/3'], D)
for b0, b1, r, tn, bs in P:
    for p in voiced('trem', tn[:3], 62, 81, 4): note('trem', p, c.T(b0), c.T(b1) - c.T(b0) + 0.1, 28)
    for p in voiced('choir', tn[:3], 55, 72, 3): note('choir', p, c.T(b0), c.T(b1) - c.T(b0) + 0.2, 30)
    note('basses', 36 + bs, c.T(b0), c.T(b1) - c.T(b0), 30)
swell('trem', c.T(74), c.T(82), 34, 62, peak=0.6); swell('choir', c.T(74), c.T(82), 34, 66, peak=0.55)
for k, (bb, p) in enumerate(((74.5, 86), (76.0, 81), (77.5, 88), (79.0, 83), (80.5, 90))):
    note('piano', p, c.T(bb), 2.2, 42 - k, human=0)
ped(c.T(74.3), c.T(82)); MARK.append(dict(t=59.6, end=66.8, part="trem", what="博览古今，学参天地: the hush, stars (high strings, chorus, piano)"))
# 研究一粒种子 / 守护一座森林 (66.8 - 73.0): P2's second half on the violins, the piano flowing again, a crescendo into the motto
P = prog(c, 82, ['I/3 ii', 'Vsus V'], D)
lh(c, P, 42, "broken", v1=58)
pad('strings', c, P, 40, lo=55, hi=77, n=4, v1=66)
mel('violins', c, 82, P2[6:], D, 52, octave=1, arch=8, mark="研究 / 守护: the answer's second half (violins), building")
swell('violins', c.T(82), c.T(90), 52, 112, peak=0.95)

# ================================================================ D. 求真敬业，肯干实干能干 (73.0 - 80.0)
P = prog(c, 90, ['I', 'IV', 'I/5 V'], D, 3)
for b0, b1, r, tn, bs in P:
    for p in voiced('strings', tn[:3], 52, 76, 4): note('strings', p, c.T(b0), c.T(b1) - c.T(b0) + 0.08, 70)
    note('basses', 36 + bs, c.T(b0), c.T(b1) - c.T(b0), 74); note('basses', 24 + bs if bs >= 4 else 36 + bs, c.T(b0), c.T(b1) - c.T(b0), 70)
    for i in range(int(round((b1 - b0) / 0.5))):                                                             # a firm walking pulse in the low strings (bowed, not staccato)
        bb = b0 + i * 0.5; note('cello', 38 + bs if bs < 10 else 26 + bs, c.T(bb), (c.T(bb + 0.5) - c.T(bb)) * 0.92, 58 + (8 if i % 2 == 0 else 0), human=0.004)
mel('horns', c, 90, [(4, 2), (7, 1), (9, 3), (7, 3)], D, 64, octave=-1, arch=4, mark="求真敬业: the horns state it")
for k, (t, chd) in enumerate(((76.9, (55, 59, 62, 67)), (77.8, (57, 62, 66, 69)), (78.85, (57, 61, 64, 69)))):   # 肯干 实干 能干: three accents on the harmony (IV, I/5, V)
    for p in chd: note('brass', p, t, 0.55, 86 + 6 * k)
    note('basses', chd[0] - 12, t, 0.6, 96); NOTES['bd'].append((t, 1.5, 48, 74 + 8 * k))
MARK.append(dict(t=76.9, end=79.4, part="brass", what="肯干 实干 能干: three accents where each word lands"))
cymroll(c.T(97.6), c.T(99) + 0.02, 70)

# ================================================================ E. 这是一所大学 … 欢迎 … 托举你的未来 … 浙江农林大学 (80.0 - 108)
P = prog(c, 99, HP3, D)
tutti(c, P, 74, parts=('strings', 'horns', 'choir'), timp=False)
lh(c, P, 54, "broken", v1=66)
mel('violins', c, 99, P3, D, 70, octave=1, arch=10, mark="这是一所大学 … 欢迎你来这里扎根、生长: P3 in full (strings, horns, chorus)")
mel('cello', c, 99, P3, D, 64, octave=-1, arch=10)
for part in ('strings', 'horns', 'choir'): cc(part, c.T(99), c.T(115), 70, 104)
P = prog(c, 115, HP4, D)
tutti(c, P, 92, parts=('strings', 'horns', 'brass', 'choir'), timp=False)
lh(c, P, 64, "broken")
mel('violins', c, 115, P4, D, 84, octave=1, arch=10, mark="托举你的未来 → 浙江农林大学: P4, the only full close, on the emblem")
mel('horns', c, 115, P4, D, 80, octave=-1, arch=8)
mel('piano', c, 115, P4, D, 62, octave=1, arch=6, harm=harm_at(P))
cymroll(c.T(129.6), c.T(131) + 0.02, 74)
# the last chord: D major, everyone, rings into the title and fades with the picture
te = c.T(131)
for part, ps, v in (('strings', (50, 57, 62, 66, 69, 74), 76), ('horns', (50, 57, 62, 66), 66), ('choir', (57, 62, 66, 69), 64), ('basses', (26, 38), 72), ('violins', (74, 78, 81), 62)):
    held(part, ps, te, 113.0, v)
    cc(part, te, te + 1.2, 110, 100); cc(part, te + 1.2, 113.0, 100, 18)
for p in (50, 62, 66, 69, 74): note('piano', p, te, 6.0, 52)
ped(te, 113.0)
cym(te, 82, drum=False)

# ================================================================ THE BED: soft far fifths under every paragraph (the dominant's A-E in D),
# the tonic fifth D-A under the hush and for the last chord
BED = [(0.4, 27.4, (57, 64), 0.55), (27.4, 59.6, (57, 64), 0.55), (59.6, 66.8, (50, 57), 0.55), (66.8, 106.0, (57, 64), -0.4), (106.0, 112.0, (50, 57), 0.0)]
for t0_, t1_, fifth, ov in BED:
    for p_ in fifth: note('drone', p_, t0_ - 0.25, t1_ - t0_ + 0.25 + ov, 40, human=0)
cc('drone', 0.4, 5.0, 10, 90)

exec(open(f"{HERE}/score_lib_v13_writer.py", encoding="utf-8").read())
