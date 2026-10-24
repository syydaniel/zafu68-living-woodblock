"""find sudden changes: per frame, the mean abs difference to the previous frame (on 1/4-size grey) and the jump in mean colour;
a frame is a jump when its difference is far above the local level (a hard cut or a flash), listed with its time"""
import sys, numpy as np, cv2
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); prev = None; d = []; col = []
while True:
    ok, fr = cap.read()
    if not ok: break
    s = cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    col.append(s.reshape(-1, 3).mean(0))
    if prev is not None: d.append(np.abs(s - prev).mean())
    else: d.append(0)
    prev = s
d = np.array(d); col = np.array(col)
med = np.array([np.median(d[max(0, i - 12):i + 13]) for i in range(len(d))])
dc = np.r_[0, np.abs(np.diff(col, axis=0)).max(1)]
print(f'frames {len(d)} fps {fps}')
for i in range(1, len(d)):
    if d[i] > 0.06 and d[i] > 3.5 * med[i] + 0.01 or dc[i] > 0.045:
        print(f'{i / fps:7.2f}s  diff {d[i]:.3f}  local {med[i]:.3f}  colour jump {dc[i]:.3f}')
