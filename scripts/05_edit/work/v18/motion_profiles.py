"""per-frame motion (90th percentile optical flow, px per source frame at 480 px width) of every AI clip -> work/v18/motion.json"""
import cv2, numpy as np, json, glob, os
out = {}
for f in sorted(glob.glob('libtv_jobs/out/*.mp4')):
    name = os.path.basename(f)[:-4]
    cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS); prev = None; m = [0.0]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY)
        if prev is not None: m.append(float(np.percentile(np.linalg.norm(dis.calc(prev, g, None), axis=-1), 90)))
        prev = g
    out[name] = {'fps': fps, 'm': m}
    print(name, len(m), round(float(np.mean(m)), 2), flush=True)
json.dump(out, open('work/v18/motion.json', 'w'))
