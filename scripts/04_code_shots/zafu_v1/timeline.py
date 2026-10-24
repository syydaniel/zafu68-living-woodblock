"""Master timeline shared by picture and music. 90 BPM, 4/4: one beat = 20 frames, one bar = 80 frames at 30 fps."""
BPM = 90
BEAT = 60 / BPM          # 0.6667 s
BAR = 4 * BEAT           # 2.6667 s
FPS = 30


def bar(b):
    return b * BAR


# sections (bars)
SEC = {
    'roots':   0,    # seed falls, luminous roots, names through time
    'bamboo':  8,    # HIT: shoot breaks the soil, rise through the bamboo forest
    'canopy':  14,   # break through the canopy
    'land':    16,   # HIT: flight over Tianmu hills and the lake
    'ink':     20,   # the landscape turns into an ink painting
    'herb':    24,   # herbarium plate, the campus is a botanical garden
    'micro':   30,   # into the leaf: cells, chloroplasts, carbon
    'rise':    36,   # build
    'climax':  38,   # HIT: lights rise, tree of light, motto
    'title':   46,   # HIT: ginkgo leaf, letterbox opens, name
    'end':     51,   # end of picture (136 s)
}
T = {k: bar(v) for k, v in SEC.items()}
END = T['end']

# exact moments inside sections (seconds)
SEED_FALL = 1.5
SEED_LAND = 3.0
NAME_1958 = bar(3)          # 天目林学院
NAME_1966 = bar(4.5)        # 浙江林学院
NAME_2010 = bar(6)          # 浙江农林大学
