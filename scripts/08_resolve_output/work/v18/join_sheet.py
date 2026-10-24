"""one row per join: frames at -0.6 .. +0.6 s around it (320x180), the join time written on the row"""
import sys, cv2, numpy as np
src, out = sys.argv[1], sys.argv[2]
joins = [float(x) for x in sys.argv[3].split(',')]
offs = [-0.6, -0.36, -0.12, 0.0, 0.12, 0.36, 0.6]
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS)
need = sorted({int(round((j + o) * fps)) for j in joins for o in offs})
frames = {}; i = 0; k = 0
while k < len(need):
    ok, fr = cap.read()
    if not ok: break
    if i == need[k]: frames[i] = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA); k += 1
    i += 1
rows = []
for j in joins:
    r = np.hstack([frames.get(int(round((j + o) * fps)), np.zeros((180, 320, 3), np.uint8)) for o in offs])
    cv2.putText(r, f'{j:.2f}', (6, 22), 0, 0.7, (0, 0, 255), 2)
    rows.append(r)
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
