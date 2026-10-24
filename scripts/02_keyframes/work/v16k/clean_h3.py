"""H3 without the distant high-rise housing: local LaMa patch only, every other pixel untouched -> K_ai2_library_first_clean"""
import numpy as np, cv2
from PIL import Image
from simple_lama_inpainting import SimpleLama
src = np.asarray(Image.open('libtv_jobs/refs/T2_library_woodblock_1280x720.png').convert('RGB'))
hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
m = np.zeros(src.shape[:2], np.uint8)
bx0, by0, bx1, by1 = 226, 170, 402, 201
sub = hsv[by0:by1, bx0:bx1]
m[by0:by1, bx0:bx1] = ((sub[..., 2] > 175) & (sub[..., 1] < 70)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); m = cv2.dilate(m, np.ones((7, 7), np.uint8))
m[186:207, 224:300] = 1                                          # the podium row of the same complex
x0, y0, x1, y1 = 100, 60, 612, 316
crop = src[y0:y1, x0:x1]
out = np.asarray(SimpleLama()(Image.fromarray(crop), Image.fromarray(m[y0:y1, x0:x1] * 255)))[:crop.shape[0], :crop.shape[1]]
res = src.copy(); mm = m[y0:y1, x0:x1, None] > 0
res[y0:y1, x0:x1] = np.where(mm, out, crop)
Image.fromarray(res).save('libtv_jobs/refs/K_ai2_library_first_clean_1280x720.png')
print('changed px', int((np.abs(res.astype(int) - src).sum(-1) > 0).sum()), 'mask px', int(m.sum()))
big = lambda a: cv2.resize(a[140:240, 180:460], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
Image.fromarray(np.vstack([big(src), big(res)])).save('work/v16k/_h3_clean_check.png')
