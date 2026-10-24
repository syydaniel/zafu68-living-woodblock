"""keyframe sources: real frame -> 16:9 1920x1080 (crop chosen per source) -> depth -> code layout (tools/woodblock.py)"""
import sys, subprocess
from PIL import Image
SRC = {  # name: (file, crop box in source px or None for centre 16:9)
 'orbit1': ('footage/orbit_library/00001.jpg', None), 'orbit744': ('footage/orbit_library/00744.jpg', None),
 'orbit3345': ('footage/orbit_library/03345.jpg', None), 'dive1199': ('footage/dive_library/01199.jpg', None),
 'dive3597': ('footage/dive_library/03597.jpg', None),
 'grass43': ('footage/grass_light/00043.jpg', (0, 300, 4096, 2604)),
 'push1': ('footage/avenue_push/00001.jpg', (0, 500, 4096, 2804)), 'push120': ('footage/avenue_push/00120.jpg', (0, 500, 4096, 2804)),
 'island_autumn': ('work/v14/refs_real/island_autumn_top.png', None), 'island_snow': ('work/v14/refs_real/island_snow_top.png', None),
 'avenue_aerial': ('work/v14/refs_real/avenue_aerial.png', None), 'campus1': ('footage/campus_aerial/00001.jpg', None),
 'lib_mist': ('work/v14/refs_real/lib_mist.png', (268, 0, 1651, 778)), 'ginkgo_backlit': ('work/v14/refs_real/ginkgo_backlit.png', (268, 0, 1651, 778)),
 'topdown479': ('footage/topdown_lake/00479.jpg', None),
 'lotus': ('work/v14/refs_real/lotus.png', (268, 0, 1651, 778)), 'canopy_top': ('work/v14/refs_real/canopy_top.png', (268, 0, 1651, 778)),
 'motto': ('work/v14/refs_real/motto_stone.png', (268, 0, 1651, 778)), 'xy04': ('work/v16k/real/xy04.png', None), 'xy43': ('work/v16k/real/xy43.png', None),
 'bamboo': ('work/v16k/real/bamboo_v1.png', None),
}
names = sys.argv[1:] or list(SRC)
for n in names:
    f, box = SRC[n]
    im = Image.open(f).convert('RGB')
    if box is None:
        w, h = im.size; nh = round(w * 9 / 16)
        box = (0, (h - nh) // 2, w, (h - nh) // 2 + nh) if nh <= h else ((w - round(h * 16 / 9)) // 2, 0, (w + round(h * 16 / 9)) // 2, h)
    im.crop(box).resize((1920, 1080), Image.LANCZOS).save(f'work/v16k/real/{n}.png')
paths = [f'work/v16k/real/{n}.png' for n in names]
subprocess.run(['.venv_ai/bin/python', 'tools/depth.py', *paths, '--out', 'work/v16k/real/depth'], check=True, capture_output=True)
for n in names:
    r = subprocess.run(['.venv_ai/bin/python', 'tools/woodblock.py', f'work/v16k/real/{n}.png', f'work/v16k/real/depth/{n}_d.png', f'work/v16k/lay/{n}'], capture_output=True, text=True)
    print(n, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
