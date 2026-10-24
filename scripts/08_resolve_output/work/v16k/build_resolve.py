"""DaVinci Resolve assembly of the film from <EL>/plan.json (default renders/v17/elements; run inside Resolve's scripting:
`resolve` is given). Timeline TL_NAME at 3840x2160 25 fps:
  V1 picture: code shots and AI shots (AI 720p clips with Super Scale), cross dissolves where the plan says 'luma'
  V2 / V3 signature transitions (ink bloom, whip, zoom, pigment grains) prerendered, laid over their joins, and the alpha
          overlays (two tracks, so overlapping pieces never collide)
  V4 captions, labels and seals (ProRes 4444 alpha)
  V5 one xuan paper texture over everything (multiply, low opacity)
  A1 旁白  A2 配乐  A3 音效  A4 环境 (the DX / MX / FX / AMB stems of the mix)"""
import json, os

ROOT = os.path.expanduser('~/Movies/ZAFU2')
EL = globals().get('EL', f'{ROOT}/renders/v17/elements')
TL_NAME = globals().get('TL_NAME', 'R9_装配')
plan = json.load(open(f'{EL}/plan.json'))
FPS = plan['fps']
# three.js effect pieces (libtv_jobs/fx/README.md), placed on exact timeline frames
FXD = globals().get('FXD', f'{ROOT}/libtv_jobs/fx/out')
if os.path.exists(f'{FXD}/FX_E3a_page.mov'):                      # 2001 -> 2002: the print turns like a page (in place of the pigment knots at 31.4; NO_PAGE keeps the knots)
    for t in plan['trans']:
        if abs(t['at'] - 31.4) < 1e-6 and t.get('file') and t['type'] == 'pigment' and not globals().get('NO_PAGE'):
            t.update(file=f'{FXD}/FX_E3a_page.mov', start=774 / FPS, dur=23 / FPS, type='page')
OVERLAYS = [(f'{FXD}/FX_T1b_leaves.mov', 987, 44)] if os.path.exists(f'{FXD}/FX_T1b_leaves.mov') else []
pm = resolve.GetProjectManager()
project = pm.GetCurrentProject()
assert project.GetName() == globals().get('PROJECT', 'ZAFU_68_v16'), project.GetName()
mp = project.GetMediaPool()
root = mp.GetRootFolder()


def folder(name):
    for f in root.GetSubFolderList():
        if f.GetName() == name: return f
    return mp.AddSubFolder(root, name)


def imp(fld, paths):
    mp.SetCurrentFolder(fld)
    have = {c.GetClipProperty('File Path'): c for c in fld.GetClipList()}
    out = {}
    need = [p for p in paths if p not in have]
    if need:
        for c in mp.ImportMedia(need) or []:
            have[c.GetClipProperty('File Path')] = c
    for p in paths: out[p] = have.get(p)
    return out


SUF = globals().get('BIN_SUFFIX', '')
f_code, f_ai, f_tr, f_txt, f_snd, f_pap = (folder(n + SUF) for n in ('代码片段', 'AI镜头', '转场', '字幕', '声音', '纸纹'))
# start clean: delete any timeline of this name, then every clip of the picture bins (files on disk may have been re-rendered)
for i in range(project.GetTimelineCount(), 0, -1):
    t_ = project.GetTimelineByIndex(i)
    if t_ and t_.GetName() == TL_NAME: mp.DeleteTimelines([t_])
for f_ in (f_code, f_ai, f_tr, f_txt, f_snd):
    cl = f_.GetClipList()
    if cl: mp.DeleteClips(cl)
shots = plan['shots']
code_items = imp(f_code, [s['file'] for s in shots if s['kind'] == 'code'])
ai_items = imp(f_ai, [s['file'] for s in shots if s['kind'] == 'ai'])
for it in list(ai_items.values()) + list(code_items.values()):
    if it: it.SetClipProperty('Super Scale', '2')                       # 720p / 1080p -> 4K: 2x Super Scale (then fit)
tr_items = imp(f_tr, [t['file'] for t in plan['trans'] if t.get('file')] + [o[0] for o in OVERLAYS])
for o in OVERLAYS:
    it_ = tr_items.get(o[0])
    if it_: it_.SetClipProperty('Alpha mode', 'Straight')
txt = imp(f_txt, [f'{EL}/overlay_text.mov'])[f'{EL}/overlay_text.mov']
AUDIO = globals().get('AUDIO', f'{ROOT}/audio/v3/out/v17/stereo')
stems = [f'{AUDIO}/{n}.wav' for n in ('DX', 'MX', 'FX', 'AMB')]
snd = imp(f_snd, stems)
paper_png = f'{EL}/paper_4k.png'
pap = imp(f_pap, [paper_png])[paper_png] if os.path.exists(paper_png) else None

name = TL_NAME
for i in range(1, project.GetTimelineCount() + 1):
    tl = project.GetTimelineByIndex(i)
    if tl and tl.GetName() == name: mp.DeleteTimelines([tl]); break
mp.SetCurrentFolder(root)
tl = mp.CreateEmptyTimeline(name)
project.SetCurrentTimeline(tl)
for _ in range(4): tl.AddTrack('video')
for _ in range(3): tl.AddTrack('audio', 'stereo')
for k, n in enumerate(('画面', '特殊转场', '特殊转场2', '字幕', '纸纹'), 1): tl.SetTrackName('video', k, n)
busy = {2: [], 3: []}
def free_track(f0, f1):
    for k in (2, 3):
        if all(f1 <= a or f0 >= b for a, b in busy[k]): busy[k].append((f0, f1)); return k
    return 3
for k, n in enumerate(('旁白', '配乐', '音效', '环境'), 1): tl.SetTrackName('audio', k, n)
T0 = tl.GetStartFrame()
fr = lambda t: int(round(t * FPS))
log = []
# V1 picture
items = []
for s in shots:
    it = (code_items if s['kind'] == 'code' else ai_items).get(s['file'])
    if it is None: log.append(f"missing {s['file']}"); continue
    a, b = s['a'], s['b']
    sf = fr(a - s['file_start']); ef = sf + fr(b) - fr(a)          # endFrame is exclusive
    r = mp.AppendToTimeline([{'mediaPoolItem': it, 'startFrame': sf, 'endFrame': ef, 'trackIndex': 1, 'recordFrame': T0 + fr(a), 'mediaType': 1}])
    items.append((s, r[0] if r else None))
    if not r: log.append(f"append failed {s['tag']}")
# cross dissolves at 'luma' joins
by_end = {round(s['b'], 3): ti for s, ti in items if ti}
for t in plan['trans']:
    if t['type'] == 'luma':
        ti = by_end.get(round(t['at'], 3))
        if ti:
            ok = ti.AddTransition({'type': 'Cross Dissolve', 'category': 'simple', 'position': 'end', 'alignment': 'center', 'duration': max(2, fr(t['dur']))})
            log.append(f"dissolve {t['at']}: {bool(ok)}")
# V2 signature transitions
placed = set()
for t in plan['trans']:
    if not t.get('file') or t['file'] in placed: continue
    placed.add(t['file'])
    it = tr_items.get(t['file'])
    if it is None: log.append(f"missing {t['file']}"); continue
    n = fr(t['start'] + t['dur']) - fr(t['start'])
    trk = free_track(fr(t['start']), fr(t['start']) + n)
    r = mp.AppendToTimeline([{'mediaPoolItem': it, 'startFrame': 0, 'endFrame': n, 'trackIndex': trk, 'recordFrame': T0 + fr(t['start']), 'mediaType': 1}])
    log.append(f"trans {t['type']} {t['at']}: {bool(r)}")
for fpath, f0, n in OVERLAYS:                                     # alpha overlays over the picture (under the captions)
    it = tr_items.get(fpath)
    r = mp.AppendToTimeline([{'mediaPoolItem': it, 'startFrame': 0, 'endFrame': n, 'trackIndex': free_track(f0, f0 + n), 'recordFrame': T0 + f0, 'mediaType': 1}]) if it else None
    log.append(f'overlay fx {os.path.basename(fpath)}: {bool(r)}')
# V4 captions (alpha), V5 paper
if txt:
    r = mp.AppendToTimeline([{'mediaPoolItem': txt, 'trackIndex': 4, 'recordFrame': T0, 'mediaType': 1}]); log.append(f'overlay {bool(r)}')
if pap:
    r = mp.AppendToTimeline([{'mediaPoolItem': pap, 'startFrame': 0, 'endFrame': fr(globals().get('END', 108)), 'trackIndex': 5, 'recordFrame': T0, 'mediaType': 1}])
    if r:
        r[0].SetProperty('CompositeMode', resolve.COMPOSITE_MULTIPLY); r[0].SetProperty('Opacity', 35.0)
    log.append(f'paper {bool(r)}')
# audio stems
for k, p in enumerate(stems, 1):
    it = snd.get(p)
    if it:
        r = mp.AppendToTimeline([{'mediaPoolItem': it, 'trackIndex': k, 'recordFrame': T0, 'mediaType': 2}]); log.append(f'audio {k} {bool(r)}')
result = {'timeline': tl.GetName(), 'v1_items': len([1 for _, x in items if x]), 'log': log}
