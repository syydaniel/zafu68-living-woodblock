"""MIDI + notes.json + score.json writer and the silence / tempo report (from 奶蛙 compose_v13.py)."""
# =============================================================================================== write the files
def vlq(n):
    n = max(0, int(n)); a = [n & 127]; n >>= 7
    while n:
        a.append((n & 127) | 128); n >>= 7
    return bytes(reversed(a))


def expression(part):
    rs = sorted(RAMPS[part]); pts = []; ns = sorted(NOTES[part])
    for i, (t0, t1, v0, v1) in enumerate(rs):
        if pts and t0 < pts[-1][0]: t0 = pts[-1][0] + 0.001                                    # overlapping ramps: the later one takes over
        if t1 <= t0: continue
        if pts and t0 - pts[-1][0] > 0.02: pts.append((t0 - 0.02, pts[-1][1]))
        pts += [(t0, v0), (t1, v1)]
        nxt = rs[i + 1][0] if i + 1 < len(rs) else 1e9
        ring = max([t + d for t, d, p, v in ns if t < t1 + 0.05 and t + d > t1 - 0.05] + [t1]) + 1.3      # the notes under the ramp, and their release
        if nxt > ring + 0.6: pts += [(ring, v1), (ring + 0.5, 100)]
    return pts


for part in PARTS:
    pts = expression(part)
    for (ta, va), (tb, vb) in zip(pts, pts[1:]):
        n = max(1, int((tb - ta) * 12))
        for i in range(n): CCS[part].append((ta + (tb - ta) * i / n, 11, int(round(max(0, min(127, va + (vb - va) * i / n))))))
    if pts: CCS[part].append((pts[-1][0], 11, int(pts[-1][1])))
TPQ = 480; TEMPO = 500000; TPS = TPQ / 0.5; END = FILM_END + 6.0; info = {}
for part, prg in PARTS.items():
    ev = []
    for t, d, p, v in NOTES[part]: ev.append((t, 1, 'on', p, v)); ev.append((t + d, 0, 'off', p, 0))
    for t, num, val in CCS[part]: ev.append((t, 2 if num != 64 or val == 0 else 0.5, 'cc', num, val))
    ev.sort(key=lambda e: (e[0], e[1])); ch = 9 if prg is None else 0; data = bytearray(b'\x00\xff\x51\x03' + TEMPO.to_bytes(3, 'big'))
    if prg is not None: data += b'\x00' + bytes([0xC0 | ch, prg])
    data += b'\x00' + bytes([0xB0 | ch, 7, 110]) + b'\x00' + bytes([0xB0 | ch, 11, 100]) + b'\x00' + bytes([0xB0 | ch, 91, 0]) + b'\x00' + bytes([0xB0 | ch, 93, 0]); last = 0
    for t, _, kind, a, b in ev:
        tick = int(round(t * TPS)); data += vlq(tick - last); last = tick; data += bytes([{'on': 0x90, 'off': 0x80, 'cc': 0xB0}[kind] | ch, a & 127, b & 127])
    data += vlq(max(0, int(round(END * TPS)) - last)) + b'\xff\x2f\x00'
    open(f"{OUT}/{part}.mid", "wb").write(b'MThd' + struct.pack('>IHHH', 6, 1, 1, TPQ) + b'MTrk' + struct.pack('>I', len(data)) + bytes(data))
    info[part] = dict(notes=len(NOTES[part]))
MARK.sort(key=lambda m: m["t"])
json.dump(dict(notes={k: [(round(t, 4), round(d, 4), p, v) for t, d, p, v in sorted(n)] for k, n in NOTES.items()}, cc={k: [(round(t, 4), num, val) for t, num, val in sorted(x)] for k, x in CCS.items()}), open(f"{OUT}/notes.json", "w"))
json.dump(dict(end=FILM_END, film=FILM_END, parts=info, programs=PARTS, statements=MARK, cym=[(round(t, 3), round(d, 3), p, v) for t, d, p, v in NOTES['cym']]), open(f"{OUT}/score.json", "w"), ensure_ascii=False, indent=1)
print("score v12:", {k: v["notes"] for k, v in info.items()})
for m in MARK: print("  %7.2f - %7.2f  %-8s %s" % (m["t"], m["end"], m["part"], m["what"]))

# what the ear cannot check here, checked by counting: the silences (only the intended ones should be listed), the tempi, quiet entries
import numpy as np
g = np.zeros(int(FILM_END * 10) + 80, int); ex = {}
for part in PARTS:
    cs = sorted((t, v) for t, num, v in CCS[part] if num == 11); ex[part] = cs
    for t, d, p, v in NOTES[part]: g[int(t * 10): int((t + d) * 10) + 1] += 1
sil = []; i = 0
while i < int(FILM_END * 10):
    if g[i] == 0:
        j = i
        while j < len(g) and g[j] == 0: j += 1
        if j - i >= 8: sil.append((i / 10, j / 10))
        i = j
    else: i += 1
print("silences (no note sounding) of 0.8 s and more:", ", ".join("%.1f-%.1f" % x for x in sil))
print("tempi (BPM between anchors):")
for t in TEMPI: print("   ", "  ".join(t))
low = []
for part in PARTS:
    cs = ex[part]
    if not cs: continue
    tt = [c_[0] for c_ in cs]; vv = [c_[1] for c_ in cs]
    for t, d, p, v in NOTES[part]:
        a = np.interp(t, tt, vv, left=100); b = np.interp(t + min(d, 1.0), tt, vv, left=100)
        if max(a, b) < 30 and d > 0.3: low.append("%s %.1f (cc %d)" % (part, t, a))
print("notes entering under expression 30 with no rise:", ", ".join(low[:40]) or "none")

