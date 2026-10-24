"""EDL variant for the bamboo hero shot inside the woodblock film. Same shot and camera as edl.py, full 16:9 frame
(no letterbox), no grain, no text (the film sets its own typography). Render: engine.py video 24.6 29.8 --edl edl_v3b --notext"""
from edl import *          # noqa: F401,F403
import edl as _E

_E.GRADE_DEFAULT.update(uAspect=1.7778, uGrain=0.0)
for _t, _d in _E.GRADE_KEYS:
    if 'uGrain' in _d:
        _d['uGrain'] = 0.0
    if 'uAspect' in _d:
        _d['uAspect'] = 1.7778
