"""Elements for the DaVinci assembly (the final edit of the film is done in DaVinci Resolve), from the shot list of edit18.film(),
written to renders/v18/elements:
  shots/S##_<tag>.mov      every code shot (and every AI shot not generated yet, as its keyframe stand-in), ProRes 422 HQ, with
                           HANDLE seconds before and after for dissolves
  ai/A##_<stem>.mov        every generated AI clip, conformed: native pixels, retimed onto its slot at 25 fps, with handles
  trans/T##_<type>.mov     the signature transitions (ink bloom, whip, zoom) prerendered from both shots
  trans/X####_<type>.mov   the signature joins inside the era long take (overlapping windows merged into one file)
  overlay_text.mov         every caption, label and seal on transparency (ProRes 4444 with alpha), full length
  plan.json                what goes where (record times, sources, in points, speeds), read by the Resolve build script
  .venv_ai/bin/python work/v16k/export18.py [shots|ai|trans|chainx|overlay|plan|all] [only=1,3,5]"""
import sys, json, subprocess
from pathlib import Path
import numpy as np, cv2
import importlib.util as ilu

ROOT = Path(__file__).resolve().parents[2]
s = ilu.spec_from_file_location('edit18', ROOT / 'work/v16k/edit18.py'); M = ilu.module_from_spec(s); sys.modules['edit18'] = M; s.loader.exec_module(M)
E = M.E; C = M.C; W, H, FPS = 1920, 1080, 25
OUTD = ROOT / 'renders/v18/elements'; (OUTD / 'shots').mkdir(parents=True, exist_ok=True); (OUTD / 'trans').mkdir(exist_ok=True)
HANDLE = 0.6
SIGNATURE = ('ink', 'whip', 'zoom', 'pigment', 'particles3d', 'mist', 'reprint', 'leafwipe', 'light', 'lumakey')                      # prerendered; 'luma' becomes a Resolve cross dissolve


def enc(path, alpha=False):
    pix, prof, ip = ('yuva444p10le', '4', 'rgba64le') if alpha else ('yuv422p10le', '3', 'rgb48le')
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', ip, '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'prores_ks', '-profile:v', prof, '-pix_fmt', pix, str(path)], stdin=subprocess.PIPE)


def put(p, img):
    p.stdin.write((np.clip(img, 0, 1) * 65535 + 0.5).astype('<u2').tobytes())


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    only = None
    for a in sys.argv[2:]:
        if a.startswith('only='): only = {int(x) for x in a[5:].split(',')}
    t0, t1, S, labels, texts, trans = M.film()
    AIS = {}                                                          # shot index -> AIShot (to know which are real AI files)
    for i, sh in enumerate(S):
        fn = sh[2]; obj = getattr(fn, '__self__', None)
        if isinstance(obj, E.AIShot): AIS[i] = obj
    plan = {'fps': FPS, 'shots': [], 'trans': [], 'overlay': 'overlay_text.mov'}
    # the era long take is one shot in the Python cut; in the DaVinci timeline each of its AI segments is its own clip
    flat = []
    for i, (a, b, fn) in enumerate(S):
        obj = getattr(fn, '__self__', None)
        if isinstance(obj, M.Chain):
            for (sa, sb, sh) in obj.segs: flat.append((sa, sb, sh.frame, sh))
            # chain joins; overlapping signature windows are merged into one continuous file (one track in DaVinci)
            wins = []
            for at, spec in obj.xf.items():
                kind, d, kw = (('luma', spec, {}) if not isinstance(spec, tuple) else (spec[0], spec[1], spec[2] if len(spec) > 2 else {}))
                if kind not in SIGNATURE:
                    plan['trans'].append({'j': -1, 'at': at, 'dur': d, 'type': kind}); continue
                wins.append([at - d / 2, at + d / 2, [at], kind])
            wins.sort()
            merged = []
            for w in wins:
                if merged and w[0] < merged[-1][1] + 1e-6:
                    merged[-1][1] = max(merged[-1][1], w[1]); merged[-1][2] += w[2]; merged[-1][3] = 'merged'
                else: merged.append(w)
            for t_s, t_e, ats, kind in merged:
                f = OUTD / 'trans' / f'X{int(round(ats[0] * 10)):04d}_{kind}.mov'
                for at in ats:
                    plan['trans'].append({'j': -1, 'at': at, 'dur': t_e - t_s, 'type': kind, 'file': str(f), 'start': t_s, 'part_of': ats[0]})
                if what in ('trans', 'chainx', 'all'):
                    p = enc(f)
                    for k in range(int(round((t_e - t_s) * FPS))):
                        put(p, obj.frame(t_s + k / FPS))
                    p.stdin.close(); p.wait(); print('chain trans', ats, kind, f'{t_s:.2f}-{t_e:.2f}', flush=True)
        else:
            flat.append((a, b, fn, AIS.get(i)))
    plan['flat'] = True
    for i, (a, b, fn, ai) in enumerate(flat):
        real_ai = ai is not None and ai.path.exists()
        tag = (ai.path.stem if ai else getattr(getattr(fn, '__self__', None), '__class__', type(fn)).__name__).replace(' ', '')
        ent = {'i': i, 'a': a, 'b': b, 'tag': tag}
        if real_ai:
            # conformed copy: the AI pixels untouched (native 1280x720), only retimed onto the slot at 25 fps, with handles
            pa, pb = max(t0, a - HANDLE), min(t1, b + HANDLE)
            f = OUTD / 'ai' / f'A{i:02d}_{ai.path.stem}.mov'; f.parent.mkdir(exist_ok=True)
            ent.update(kind='ai', file=str(f), file_start=pa, file_end=pb, src=str(ai.path), src_in=ai.s0, src_out=ai.s1)
            if what in ('ai', 'all') and (only is None or i in only):
                raw = ai.raw; fps_s = ai.fps; n = len(raw); hh, ww = raw[0].shape[:2]
                gain = None
                if ai.path.stem in ('T1b', 'S7', 'AI-9', 'E2', 'AI-10'):          # seasons / growth: no sudden colour jumps
                    mu = np.array([cv2.resize(f_, (64, 36), interpolation=cv2.INTER_AREA).reshape(-1, 3).mean(0) for f_ in raw], np.float32)
                    k_ = int(round(fps_s * 0.8)) | 1; pad = k_ // 2
                    sm = np.stack([np.convolve(np.pad(mu[:, c], pad, mode='edge'), np.ones(k_) / k_, mode='valid') for c in range(3)], 1)
                    gain = np.clip(sm / np.maximum(mu, 1), 0.8, 1.25)
                p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{ww}x{hh}', '-r', str(FPS), '-i', '-',
                                      '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', str(f)], stdin=subprocess.PIPE)
                for k in range(int(round((pb - pa) * FPS))):
                    t = pa + k / FPS
                    st = ai.src_time(t) if hasattr(ai, 'src_time') else ai.s0 + (ai.s1 - ai.s0) * (t - a) / (b - a)
                    x = min(max(st * fps_s, 0), n - 1); i0 = int(x); fr = x - i0; i1 = min(i0 + 1, n - 1)
                    if hasattr(ai, 'pic'): img = ai.pic(t)                        # flow in-betweens, shutter, live past the ends
                    else: img = raw[i0] if fr < 0.02 else cv2.addWeighted(raw[i0], 1 - fr, raw[i1], fr, 0)
                    win = {'T1b': [(2.7, 3.4), (3.9, 4.9), (5.15, 6.6)]}.get(ai.path.stem, [])
                    for w0, w1 in win:                                             # sudden season jumps: a short temporal blend
                        if w0 <= st <= w1:
                            kk = int(round(0.3 * fps_s)); lo_, hi_ = max(0, i0 - kk), min(n, i0 + kk + 1)
                            img = np.mean(np.stack(raw[lo_:hi_]).astype(np.float32), 0).astype(np.uint8)
                    if gain is not None:
                        g_ = gain[i0] * (1 - fr) + gain[i1] * fr
                        img = np.clip(img.astype(np.float32) * g_[None, None, :], 0, 255).astype(np.uint8)
                    if ai.path.stem == 'AI-10' and 2.6 < st < 4.6:            # the one dull grey cloud bank: lift it to match its neighbours
                        g_ = 1 + 0.22 * np.sin(np.pi * (st - 2.6) / 2.0); img = np.clip(img.astype(np.float32) * g_, 0, 255).astype(np.uint8)
                    z_ = ai.zoom_at(t) if hasattr(ai, 'zoom_at') else getattr(ai, 'zoom', 1.0)
                    if z_ > 1.0:
                        ch_, cw_ = int(hh / z_), int(ww / z_); img = cv2.resize(img[(hh - ch_) // 2:(hh - ch_) // 2 + ch_, (ww - cw_) // 2:(ww - cw_) // 2 + cw_], (ww, hh), interpolation=cv2.INTER_LANCZOS4)
                    p.stdin.write(img.tobytes())
                p.stdin.close(); p.wait(); print('ai', i, ai.path.stem, f'{pa:.2f}-{pb:.2f}', flush=True)
        else:
            pa, pb = max(t0, a - HANDLE), min(t1, b + HANDLE)
            f = OUTD / 'shots' / f'S{i:02d}_{tag}.mov'
            ent.update(kind='code', file=str(f), file_start=pa, file_end=pb)
            if what in ('shots', 'all') and (only is None or i in only):
                p = enc(f)
                for k in range(int(round((pb - pa) * FPS))):
                    put(p, fn(pa + k / FPS))
                p.stdin.close(); p.wait(); print('shot', i, tag, f'{pa:.2f}-{pb:.2f}', flush=True)
        plan['shots'].append(ent)
    for j, tr in trans.items():
        bt = S[j][1]; d = tr[1]; kind = tr[0]
        ent = {'j': j, 'at': bt, 'dur': d, 'type': kind}
        kw = dict(tr[2]) if len(tr) > 2 else {}
        align = kw.pop('align', 'centre')                     # 'end': the whole transition happens before the cut and lands on B's first frame
        t_s, t_e = (bt - d, bt) if align == 'end' else (bt - d / 2, bt + d / 2)
        if kind in SIGNATURE:
            f = OUTD / 'trans' / f'T{j:02d}_{kind}.mov'; ent['file'] = str(f); ent['start'] = t_s
            if what in ('trans', 'all') and (only is None or j in only) and kind == 'particles3d':
                sys.path.insert(0, str(ROOT / 'work/v16k')); import particles3d as P3
                n = int(round(d * FPS)); A0 = S[j][2](t_s).astype(np.float32); B1 = S[j + 1][2](t_e - 1e-3 if align == 'end' else t_e).astype(np.float32)
                if align == 'end': B1 = S[j + 1][2](bt).astype(np.float32)
                As = [S[j][2](t_s + k / FPS).astype(np.float32) for k in range(n)]           # both shots keep moving under the grains
                Bs = [S[j + 1][2](t_s + k / FPS).astype(np.float32) for k in range(n)]
                p = enc(f)
                for fr_ in P3.render(A0, B1, n, A_seq=As, B_seq=Bs, **kw): put(p, fr_)
                p.stdin.close(); p.wait(); print('trans', j, kind, flush=True)
            elif what in ('trans', 'all') and (only is None or j in only):
                p = enc(f)
                for k in range(int(round(d * FPS))):
                    t = bt - d / 2 + k / FPS
                    A = S[j][2](t).astype(np.float32); B = S[j + 1][2](t).astype(np.float32)
                    put(p, C.TRANS[kind](A, B, (t - (bt - d / 2)) / d, **kw))
                p.stdin.close(); p.wait(); print('trans', j, kind, flush=True)
        plan['trans'].append(ent)
    if what in ('overlay', 'all'):
        # every text element drawn over black and over white: alpha = 1 - (white - black), colour = black / alpha
        p = enc(OUTD / 'overlay_text.mov', alpha=True)
        for f in range(int(round((t1 - t0) * FPS))):
            t = t0 + f / FPS
            k0 = np.zeros((H, W, 3), np.float32); k1 = np.ones((H, W, 3), np.float32)
            for L_ in labels: L_.draw(k0, t); L_.draw(k1, t)
            for T in texts: T.draw(k0, t); T.draw(k1, t)
            al = np.clip(1 - (k1 - k0).mean(-1), 0, 1)
            col = np.where(al[..., None] > 1e-4, k0 / np.maximum(al[..., None], 1e-4), 0)
            p.stdin.write((np.clip(np.dstack([col, al]), 0, 1) * 65535 + 0.5).astype('<u2').tobytes())
            if f % 250 == 0: print('overlay', f, flush=True)
        p.stdin.close(); p.wait()
    json.dump(plan, open(OUTD / 'plan.json', 'w'), ensure_ascii=False, indent=1)
    print('plan ->', OUTD / 'plan.json')


if __name__ == '__main__':
    main()
