"""Phrase timing of the narration, so the captions keep pace with the voice. Per-character timing of this voice is not
reliable (slow delivery, third-tone syllables nearly as quiet as breaths), but its pauses are: every line is cut at its punctuation
into phrases, and each phrase starts where the voice comes out of a pause. The pauses inside a line are matched in order to its
commas (the pause nearest to where the comma falls by syllable count, longer pauses preferred); a phrase ends where the voice goes
quiet before the next one. Characters inside a phrase are spread evenly between its start and end (for the texts that reveal one
character at a time); the captions show a whole phrase as it begins.
  .venv_ai/bin/python work/v18/align_chunks.py  ->  work/v18/chunks.json  {u: [[phrase, t_start, t_end, [t per char]], ...]}"""
import sys, json, importlib.util as ilu
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).resolve().parents[2]
s = ilu.spec_from_file_location('edit18', ROOT / 'work/v16k/edit18.py'); M = ilu.module_from_spec(s); sys.modules['edit18'] = M; s.loader.exec_module(M)
E17 = M.E17
PUNCT = '，。、,.！？? '
TEXT = {'u01': '一粒种子会如何成长？', 'u12': '在这里，我们博览古今，我们学参天地', 'u18': '浙江农林大学'}
for u, lines in E17.LINES.items(): TEXT[u] = '，'.join(lines)
HOP = 0.01


def energy():
    sr, x = wavfile.read(ROOT / 'audio/v3/out/v18/stereo/DX.wav'); x = x.astype(np.float32)
    x = resample_poly(x.mean(1) if x.ndim > 1 else x, 1, 3); sr //= 3
    f, t, Z = stft(x, sr, nperseg=400, noverlap=400 - int(sr * HOP), nfft=512, boundary=None)
    P = np.abs(Z) ** 2; band = (f >= 150) & (f <= 4000)
    e = 10 * np.log10(P[band].sum(0) + 1e-10)
    return np.convolve(e, np.ones(3) / 3, 'same')


def phrases(u):
    out, cur = [], ''
    for c in TEXT[u]:
        if c in PUNCT:
            if cur: out.append(cur); cur = ''
        else: cur += c
    if cur: out.append(cur)
    return out


def main():
    es = energy(); ref = np.percentile(es, 98)
    loud = es > ref - 18                                            # speech (breaths and room tone stay below)
    quiet = es < ref - 24
    on = M.CUE_ONSET; us = sorted(on, key=on.get); res = {}
    for k, u in enumerate(us):
        g = int((on[u] - 0.15) / HOP)
        while not loud[g]: g += 1
        while es[g - 1] < es[g] - 0.8 and es[g - 1] > ref - 45: g -= 1     # the foot of the first rise
        t0 = g * HOP
        nxt = on[us[k + 1]] - 0.1 if k + 1 < len(us) else len(es) * HOP - 0.1
        # the line ends at its last loud frame before the next line (breaths are not loud)
        last = g
        for h in range(g, int(nxt / HOP)):
            if loud[h]: last = h
        t1 = last * HOP + 0.04
        # pauses inside the line: quiet runs of at least 0.12 s
        pz = []; h = g
        while h < last:
            if quiet[h]:
                h2 = h
                while h2 < last and not loud[h2]: h2 += 1
                if (h2 - h) * HOP >= 0.12: pz.append((h * HOP, h2 * HOP))
                h = h2 + 1
            else: h += 1
        ph = phrases(u); n = [len(p) for p in ph]; tot = sum(n)
        starts = [t0]
        if len(ph) > 1:
            exp = [t0 + (t1 - t0) * sum(n[:j]) / tot for j in range(1, len(ph))]
            # choose one pause per comma, in order: closest to where the comma falls, longer pauses preferred
            best = None
            import itertools
            for comb in itertools.combinations(range(len(pz)), min(len(ph) - 1, len(pz))):
                if len(comb) < len(ph) - 1: continue
                sc = sum(abs(pz[c][1] - x) / 0.6 - (pz[c][1] - pz[c][0]) / 0.3 for c, x in zip(comb, exp))
                if best is None or sc < best[0]: best = (sc, comb)
            if best is None:
                starts += exp; print('  ', u, 'not enough pauses: by syllable count')
            else: starts += [pz[c][1] for c in best[1]]
        ends = []
        for j in range(len(ph)):
            if j + 1 < len(ph):
                pe = [p_ for p_ in pz if abs(p_[1] - starts[j + 1]) < 1e-6]
                ends.append(pe[0][0] if pe else starts[j + 1] - 0.15)
            else: ends.append(t1)
        rows = []
        for p_, a, b in zip(ph, starts, ends):
            m = len(p_); ts = [round(a + (b - a - 0.18) * i / max(m - 1, 1) * (m - 1) / m, 3) for i in range(m)] if m > 1 else [round(a, 3)]
            rows.append([p_, round(a, 3), round(b, 3), ts])
        res[u] = rows
        print(u, ' | '.join(f'{p_} {a:.2f}-{b:.2f}' for p_, a, b, _ in rows))
    json.dump(res, open(ROOT / 'work/v18/chunks.json', 'w'), ensure_ascii=False, indent=0)
    print('->', ROOT / 'work/v18/chunks.json')


if __name__ == '__main__':
    main()
