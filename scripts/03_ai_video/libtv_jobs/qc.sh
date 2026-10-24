#!/bin/zsh
# QC for one LibTV output: probe, contact sheet (every 6 frames), first-frame vs input similarity.
# usage: qc.sh out/<id>.mp4 refs/<first_frame>.png
set -e
vid=$1; ref=$2
base=${vid:r}
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames:format=duration -of compact "$vid"
ffmpeg -loglevel error -y -i "$vid" -vf "select='not(mod(n\,6))',scale=320:-1,tile=6x6:padding=4" -frames:v 1 "${base}_sheet.jpg"
ffmpeg -loglevel error -y -i "$vid" -vf "select='eq(n\,0)'" -frames:v 1 "${base}_f0.png"
ffmpeg -loglevel error -y -sseof -0.05 -i "$vid" -frames:v 1 -update 1 "${base}_flast.png"
ffmpeg -loglevel error -i "${base}_f0.png" -i "$ref" -lavfi "[0]scale=1280:720[a];[1]scale=1280:720[b];[a][b]ssim" -f null - 2>&1 | grep -o "All:[0-9.]*" || true
