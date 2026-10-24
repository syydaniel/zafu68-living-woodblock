"""Image jobs through the Codex CLI: codex/jobs/<batch>.json -> codex/out/<id>.png (skips existing).
  python3 codex/run_batch.py <batch> [--par 4] [--only id1,id2]
Each job: {"id", "refs": [paths relative to the project root], "prompt", "transparent": bool, "aspect": "16:9"|"3:2"|"1:1"|"2:3"}.
The image model only reconstructs/cleans/separates REAL photos (refs); it never invents history.
"""
import json, subprocess, sys, time, shutil, concurrent.futures as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex'
batch = sys.argv[1]
par = int(sys.argv[sys.argv.index('--par') + 1]) if '--par' in sys.argv else 4
only = set(sys.argv[sys.argv.index('--only') + 1].split(',')) if '--only' in sys.argv else None
jobs = json.loads((ROOT / f'codex/jobs/{batch}.json').read_text())
OUT = ROOT / 'codex/out'; OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / f'codex/out/{batch}.log'

COMMON = ('No text, no letters, no Chinese characters, no logos, no watermarks, no signs, no captions, no borders, no frames. '
          'No distorted or melted shapes; architecture must keep straight, clean, correct lines.')
# for real inscriptions (gate stones, the motto stone): keep the characters that are in the photo, add nothing
COMMON_TEXT = ('Do not add any new text, logos, watermarks, captions, borders or frames. Keep existing carved characters exactly as photographed. '
               'No distorted or melted shapes; architecture must keep straight, clean, correct lines.')


def run(job):
    jid = job['id']
    dst = OUT / f'{jid}.png'
    if dst.exists():
        return jid, 'exists', 0
    wd = ROOT / f'codex/work/{jid}'; wd.mkdir(parents=True, exist_ok=True)
    refs = [str(ROOT / r) for r in job.get('refs', [])]
    lab = ''.join(f'Image {i + 1} is a REAL photograph used as the reference. ' for i, _ in enumerate(refs[:1]))
    if len(refs) > 1:
        lab = ' '.join(f'Image {i + 1}: real reference photograph.' for i in range(len(refs))) + ' '
    if job.get('labels'):
        lab = ' '.join(f'Image {i + 1}: {t}' for i, t in enumerate(job['labels'])) + ' '
    tr = ('Ask the image tool for a genuinely transparent background (real alpha channel) and keep that alpha in the saved PNG. '
          if job.get('transparent') else '')
    asp = job.get('aspect', '16:9')
    prompt = (f'Use your built-in image generation tool to make exactly one image. Do not write code and do not use the CLI fallback. '
              f'{lab}{job["prompt"]} {COMMON_TEXT if job.get("keep_text") else COMMON} Aspect ratio {asp}, highest resolution the tool allows. {tr}'
              f'When done, copy the generated PNG into the current directory as {jid}.png and reply with that path only.')
    cmd = [CODEX, 'exec', '--skip-git-repo-check', '-s', 'workspace-write', '-c', 'model_reasoning_effort="low"']
    for r in refs:
        cmd += ['-i', r]
    cmd += ['--', prompt]
    t0 = time.time()
    with open(wd / 'run.log', 'w') as lf:
        p = subprocess.run(cmd, cwd=wd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=900)
    src = wd / f'{jid}.png'
    if src.exists():
        shutil.copy(src, dst)
        (OUT / f'{jid}.json').write_text(json.dumps(job, ensure_ascii=False, indent=1))
        return jid, 'ok', time.time() - t0
    return jid, f'FAIL rc={p.returncode}', time.time() - t0


todo = [j for j in jobs if not only or j['id'] in only]
with cf.ThreadPoolExecutor(par) as ex, open(LOG, 'a') as log:
    for jid, st, dt in ex.map(run, todo):
        line = f'{time.strftime("%H:%M:%S")} {jid:24s} {st} {dt:.0f}s'
        print(line, flush=True); log.write(line + '\n'); log.flush()
