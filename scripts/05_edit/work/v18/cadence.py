"""The 4-frame cadence of the generated clips: the video model makes its frames in groups of four, so inside a clip
the camera does not move evenly (one step in four is short, the next long; T1b even repeats a frame every four) and one frame in
four is sharper than the others. Measured here per clip, once:
  steps[i]  mean optical-flow displacement from frame i-1 to i (DIS medium, 640 px wide, textured pixels only)
  sharp[i]  Laplacian variance of frame i (640 px)
  sigma[i]  the Gaussian blur (in source pixels) that brings a too-sharp frame down to its neighbours' median sharpness
edit18.AIShotMap uses steps to place frames at their true times (cumulative smoothed motion); sharp and sigma are kept for reference.
  .venv_ai/bin/python work/v18/cadence.py  ->  work/v18/cadence.json"""
import json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'libtv_jobs/out'
CLIPS = ['AI-1', 'O1', 'AI-G', 'S1', 'S2', 'S3', 'S4', 'E2', 'E3a', 'S7', 'T1b', 'AI-4', 'AI-5', 'AI-7', 'AI-8', 'AI-9', 'AI-10']


def lapvar(g): return float(cv2.Laplacian(g, cv2.CV_32F).var())


def main():
    res = json.load(open(ROOT / 'work/v18/cadence.json')) if (ROOT / 'work/v18/cadence.json').exists() else {}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for name in CLIPS:
        if name in res and '--redo' not in sys.argv: continue
        cap = cv2.VideoCapture(str(OUT / f'{name}.mp4')); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        h, w = fr[0].shape[:2]; W2 = 640; H2 = int(round(h * W2 / w)); sc = w / W2
        gs = [cv2.cvtColor(cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in fr]
        steps = [0.0]
        for i in range(1, len(gs)):
            fl = dis.calc(gs[i - 1], gs[i], None); mag = np.linalg.norm(fl, axis=-1)
            tex = cv2.Laplacian(gs[i - 1], cv2.CV_32F) ** 2 > 20                      # flat paper and sky say nothing
            steps.append(float(mag[tex].mean() if tex.sum() > 500 else mag.mean()))
        sharp = [lapvar(g) for g in gs]
        sigma = []
        for i, g in enumerate(gs):
            lo, hi = max(0, i - 4), min(len(gs), i + 5)
            tgt = float(np.median([sharp[j] for j in range(lo, hi) if j != i]))
            s_ = 0.0
            if sharp[i] > 1.10 * tgt:
                a, b = 0.0, 2.5                                                         # sigma at 640 px, bisection
                for _ in range(14):
                    mid = (a + b) / 2
                    if lapvar(cv2.GaussianBlur(g, (0, 0), mid)) > tgt: a = mid
                    else: b = mid
                s_ = (a + b) / 2 * sc                                                   # in source pixels
            sigma.append(round(s_, 3))
        st = np.array(steps[1:]); ms = np.convolve(np.pad(st, 4, mode='edge'), np.ones(9) / 9, 'valid')
        mov = ms > 0.6; idx = np.arange(1, len(steps))
        rat = [round(float((st[mov] / ms[mov])[idx[mov] % 4 == j].mean()), 2) if (mov & (idx % 4 == j)).any() else None for j in range(4)]
        print(f'{name:6s} {len(fr)} frames, mean step {np.mean(steps):.2f}px, step ratio by frame mod 4 {rat}, softened {sum(1 for x in sigma if x > 0)}', flush=True)
        res[name] = dict(steps=[round(x, 4) for x in steps], sharp=[round(x, 1) for x in sharp], sigma=sigma)
        json.dump(res, open(ROOT / 'work/v18/cadence.json', 'w'))


if __name__ == '__main__':
    main()
