#!/bin/bash
# score v10: compose -> render every part dry (Salamander grand + Sonatina orchestra, 48 kHz) -> hall mix (stereo) -> 8D mix
cd ~/Movies/ZAFU2
export SCORE_DIR=~/Movies/ZAFU2/music/score_v10; export SCORE_OUT=~/Movies/ZAFU2/music/score_v10
python3 music/score_zafu_v10.py > music/score_v10_compose.log 2>&1
~/venvs/audio/bin/python ~/Movies/NAIWA_BEYOND/music/v8/render_score.py > music/score_v10_render.log 2>&1 || { echo RENDER_FAIL; exit 1; }
~/venvs/audio/bin/python music/mix_zafu.py > music/score_v10_mix.log 2>&1 || { echo MIX_FAIL; exit 1; }
~/venvs/audio/bin/python music/mix_zafu_bin.py > music/score_v10_mixbin.log 2>&1 || { echo MIXBIN_FAIL; exit 1; }
echo SCORE_V10_DONE
