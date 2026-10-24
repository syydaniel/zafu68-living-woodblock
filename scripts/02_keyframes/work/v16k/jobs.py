"""Keyframe job lists for the Codex CLI image runner -> codex/jobs/v16_K<batch>.json
(one shared style block, fixed style anchors: H3 clean + T1 lawn + T1 swans).
  python3 work/v16k/jobs.py A|B|C|D"""
import json, sys
STY = ['libtv_jobs/refs/K_ai2_library_first_clean_1280x720.png', 'codex/out/T1_lawn_golden.png', 'codex/out/T1_swans.png']
STY_L = ['style anchor 1 (the film\'s woodblock print style: carving, palette, paper); do not copy its content or composition.',
         'style anchor 2 (same print style, golden side-backlight); do not copy its content or composition.',
         'style anchor 3 (same print style); do not copy its content or composition.']
STYLE = ('a refined full-colour Jiangnan water-print woodblock print (江南水印木刻) in exactly the same print style, palette and paper as the style anchors: '
         'hand-pulled print, carved ink outlines, flat printed colour blocks with slight misregistration, faint wood grain in flat areas, xuan paper fibres, '
         'no 3D shading, no glossy highlights, no photographic detail, the sky left mostly as warm paper with a soft printed wash.')
LIGHT = 'Light: late golden afternoon, low sun outside the frame at the upper left, warm side-backlight, soft shadows falling to the lower right.'
KEEP = ('Registration is the priority: keep Image 1\'s exact canvas and framing; every building, ridge, shoreline, road, bridge and tree mass stays at the '
        'same position and size; do not crop, zoom, extend, tilt or re-compose; do not add or remove buildings.')
CLEAN = 'Remove every person, car, bicycle, sign board, lamp post and banner. No people, no cars, no text, no seals, no signatures.'
MV = 'work/v16k/K_library_multiview.png'


def job(jid, real, prompt, layout=True, extra=(), extra_l=()):
    refs, labels = [], []
    if layout:
        refs.append(f'work/v16k/lay/{real}/woodblock.png'); labels.append('a rough woodblock layout made by code from the real photo (Image 2); it fixes the exact composition and every outline.')
    refs.append(f'work/v16k/real/{real}.png'); labels.append('the real photograph of this place, only to understand what each shape is.' if layout else 'the real photograph of this place; it fixes the exact composition.')
    for r, l in zip(extra, extra_l): refs.append(r); labels.append(l)
    refs += STY; labels += STY_L
    return {'id': jid, 'refs': refs, 'labels': labels, 'aspect': '16:9', 'prompt': f'Make one image: Image 1 finished as {STYLE} {KEEP} {prompt} {LIGHT} {CLEAN}'}


A = [
 {'id': 'K_library_multiview', 'aspect': '16:9',
  'refs': ['work/v16k/real/orbit1.png', 'work/v16k/real/orbit744.png', 'work/v16k/real/dive1199.png', 'work/v16k/real/orbit3345.png'] + STY,
  'labels': ['real aerial photo: the ring-shaped library seen from the front three-quarter, across the lake.', 'real aerial photo: the same library from close above its open courtyard side.',
             'real aerial photo: the same library seen straight from above.', 'real aerial photo: the same library seen from behind, the lake beyond it.'] + STY_L,
  'prompt': f'Make one model sheet of this ONE ring-shaped library building, drawn as {STYLE} Four panels in a 2 x 2 grid on one sheet of paper, separated by plain paper gutters: '
            'top left: front three-quarter view from the lake side (as Image 1); top right: side view toward the open courtyard (as Image 2); bottom left: straight top-down view (as Image 3); '
            'bottom right: back view (as Image 4). Every panel shows exactly the same building: a C-shaped ring of white floors around a round courtyard, the big curved white roof slab '
            'with its skylight rows, the broad stepped plaza on the open side; keep the real proportions, number of floors and window rhythm from the photographs. Only the building and a '
            'little ground and trees around it in each panel, no other buildings. ' + LIGHT + ' ' + CLEAN},
 job('K_ai3_trunk_first', 'grass43', 'A low camera at grass level: the bases of two big tree trunks rising out of fine lawn, the trunks carved in warm ochre and ink with bark lines, the grass as fine carved strokes in mineral green and gold.'),
 job('K_ai3_canopy_last', 'ginkgo_backlit', 'Looking up into a backlit tree canopy against the low sun: replace the ginkgo leaves with palmate MAPLE leaves (槭树, five-lobed) and hang many clusters of paired winged maple samaras (翅果, two wings joined at the seed) from the twigs, glowing translucent gold and ochre against the light; the sun itself as a soft disc of bare paper. Keep the branch structure and leaf masses where they are in Image 1.'),
 job('K_ai6_path_first', 'push1', 'A quiet tree-lined campus road under big broadleaf trees, low camera on the road, sun spots on the asphalt drawn as printed light patches, backlight through the leaves.'),
 job('K_ai6_path_last', 'push120', 'The same tree-lined campus road further along, low camera on the road, sun glowing through the trees at the end, light patches on the road.'),
 job('K_ai9_luming_autumn_last', 'island_autumn', 'Straight top-down view of the lake with the small round wooded island and its zigzag footbridge, the lakeside road on the left; autumn: golden ginkgo along the road, red and orange maples, the lake as flat mineral blue-green with fine carved ripples.'),
 job('K_ai7_campus_mountains_last', 'avenue_aerial', 'Aerial view along the long golden ginkgo avenue in autumn, the campus buildings and the mountains beyond; the ginkgo crowns as carved golden clumps.'),
 job('K_ai10_campus_aerial_first', 'campus1', 'Aerial view over the campus buildings toward the green mountains, a wide band of paper sky above the ridge.'),
]
B = [
 job('K_ai4_library_high_last', 'dive3597', 'Straight top-down view of the ring-shaped library, its round courtyard and stepped plaza, the curving roads, lawns and dense trees all around filling the frame. Image 3 (model sheet, bottom-left panel) shows how this exact building looks from above in this print style.', extra=[MV], extra_l=['the model sheet of this library in the film\'s print style (structure reference).']),
 job('K_ai5_library_mist_first', 'lib_mist', 'Early morning: the library across the still lake with its full reflection, low mist lying on the water and along the far shore, red metasequoias on the left; leave the upper third of the frame as calm paper sky for vertical text. Light: soft early sun from the upper left.'.replace('', ''), extra=[MV], extra_l=['the model sheet of this library in the film\'s print style (structure reference).']),
]
TI = 'codex/out/T1_tianmu_ink.png'
H3C = 'libtv_jobs/refs/K_ai2_library_first_clean_1280x720.png'
C = [
 {'id': 'K_build_drawing', 'aspect': '16:9', 'refs': [H3C, MV],
  'labels': ['the finished woodblock print of the campus; it fixes the exact composition, every building outline, the ridges and the shoreline.', 'the model sheet of the ring-shaped library (structure reference).'],
  'prompt': 'Make one image: the SAME view as Image 1, as an architect\'s construction drawing (施工图) printed in pine-soot ink on warm xuan paper, before anything is built. '
            'Registration is the priority: every building outline, the ring library, the ridges and the lake shoreline stay exactly where they are in Image 1; do not crop, zoom or re-compose. '
            'All buildings are fine ink line drawings only: floor lines, column grid, the curved roof slab of the ring library as construction lines, a few thin scaffolding and crane-free guide lines; no colour, no fill. '
            'The mountains and far ranges are faint grey ink washes with carved ridge lines; the lake is an empty outline with a few horizontal hatching strokes; trees are only sparse dotted circles. '
            'Hand-pulled woodblock line print, xuan paper fibres, lots of empty paper. No people, no cars, no text, no numbers, no dimension labels, no seals, no signatures.'},
 job('K_plant_lotus', 'lotus', 'Close view of lotus: broad round leaves and one open pink lotus flower above the water, the flower in a soft pale rose (a little red ochre over paper).', layout=False),
 job('K_plant_ginkgo', 'ginkgo_backlit', 'Backlit fan-shaped ginkgo leaves on dark twigs against the low sun, leaves glowing gold and ochre, the sun as a soft disc of bare paper.', layout=False),
 job('K_plant_canopy', 'canopy_top', 'Straight down onto a mixed forest canopy: crowns of many species as carved clumps in mineral green, ochre and a few red-ochre autumn crowns.', layout=False),
 job('K_plant_redleaf_snow', 'xy04', 'Close view of a twig of small red-brown leaves holding fresh snow, the background a soft dark green wash, a few falling snowflakes as paper dots. Light: soft winter daylight.', layout=False),
 job('K_plant_blossom_snow', 'xy43', 'Close view of wintersweet branches with pale yellow blossoms under fresh snow, the snow as bare paper with fine carved edges. Light: soft winter daylight.', layout=False),
 job('K_island_summer', 'topdown479', 'Straight top-down view of the lake, the round wooded island on the right and the zigzag stone footbridge; summer: deep green trees, the water flat mineral green-blue with fine carved ripples.'),
 job('K_motto_stone', 'motto', 'A long low natural boulder standing on a lawn in front of dense trees. IMPORTANT: the front face of the boulder is smooth and completely BLANK, with no characters, no carving and no marks at all (text is added later).', layout=False),
 job('K_bamboo_up', 'bamboo', 'Looking straight up a bamboo grove: tall culms with their nodes converging toward the bright sky, fine leaves at the top; NO motion streaks, crisp carved culms in mineral green and ochre, the sky as paper.', layout=False),
 {'id': 'K_tianmu_colour', 'aspect': '16:9', 'refs': [TI, 'work/v16t/T1/tianmu_clouds_169.png'] + STY,
  'labels': ['the ink-only print of the Tianmu cloud sea; it fixes the exact composition and every carved line.', 'the real photograph, for the real colours of the forest and the mountains.'] + STY_L,
  'prompt': f'Make one image: Image 1 printed again with the colour blocks added, as {STYLE} Keep every line, ridge, cloud edge and the forest of Image 1 exactly where it is; only add colour: mineral green and blue on the forest and the ranges, the cloud sea left as paper with a faint warm wash. {LIGHT} {CLEAN}'},
]
D = [
 {'id': 'K_ai9_luming_snow_first', 'aspect': '16:9', 'refs': ['codex/out/K_ai9_luming_autumn_last.png', 'work/v16k/real/xy356.png'] + STY[:1],
  'labels': ['the autumn print of this exact view; it fixes the composition: the road, the bridge, the island, the shoreline must not move by a single pixel.', 'a real photograph of the same lake in snow, only for how snow lies on it.', STY_L[0]],
  'prompt': 'Make one image: Image 1 printed again as WINTER after fresh snow, same print style. Registration is the priority: keep every outline of Image 1 exactly in place (road, zigzag bridge, island, shoreline, buildings, tree positions); do not crop, zoom or re-compose. Snow is bare white paper on the road, the bridge, the roofs and the ground; trees are bare dark ink branches dusted with snow; the lake is dark ink-blue and still, with a few thin floes. Soft grey winter daylight. No people, no cars, no text, no seals, no signatures.'},
 {'id': 'K_ai10_clouds_ink_last', 'aspect': '16:9', 'refs': ['codex/out/K_ai10_campus_aerial_first.png'] + STY[:2],
  'labels': ['the campus print the camera rises from (same style, same light).', STY_L[0], STY_L[1]],
  'prompt': f'Make one image in exactly the print style of Image 1: the camera has risen high above the campus into the sky at golden hour. A great bank of golden clouds fills the upper two thirds, lit from the upper left; toward the frame edges the clouds turn from printed colour into loose pine-soot ink wash on xuan paper, like a woodblock dissolving into an ink painting; at the bottom only a few pale ink strokes of distant mountains remain, the campus far below and almost gone. No sun disc, no people, no text, no seals, no signatures.'},
 {'id': 'K_ai6_path_first_seedling', 'aspect': '16:9', 'refs': ['codex/out/K_ai6_path_first.png', 'codex/out/K_ai3_canopy_last.png'] + STY[:1],
  'labels': ['the print of the tree-lined road; keep everything in it exactly where it is.', 'the film\'s maple print: how the maple leaves and the winged samaras are drawn.', STY_L[0]],
  'prompt': 'Make one image: Image 1 with ONE new element in the foreground, same print style: at the lower left, very close to a camera that sits on the ground, a young maple seedling has just come up from the leaf litter, two narrow seed leaves and its first pair of small five-lobed true leaves, backlit gold; beside it lies the split husk of the winged samara it grew from. The seedling is sharp and large (about a third of the frame height); the road and the trees behind stay exactly as in Image 1, slightly softer. No people, no cars, no text, no seals, no signatures.'},
 job('K_ai7_ginkgo_first', 'ai7_close_layout', 'A drone skimming just above the crowns of the golden ginkgo avenue: the gold crowns fill the frame as carved clumps, a thin strip of the avenue road between them, a few campus roofs at the edge; same palette and light as the style anchors.'),
]
batch = sys.argv[1]
json.dump({'A': A, 'B': B, 'C': C, 'D': D}[batch], open(f'codex/jobs/v16_K{batch}.json', 'w'), ensure_ascii=False, indent=1)
print(len({'A': A, 'B': B, 'C': C, 'D': D}[batch]), 'jobs')
