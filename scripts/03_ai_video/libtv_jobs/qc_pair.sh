#!/bin/zsh
# Download one LibTV clip and QC it against both keyframes.
# usage: qc_pair.sh <id> <url> refs/<first>.png refs/<last>.png
set -e
cd ${0:A:h}
id=$1; url=$2; first=$3; last=$4
vid=out/$id.mp4
[[ -s $vid ]] || curl -sSL -o $vid "$url"
zsh qc.sh $vid $first | sed "s/^All:/first SSIM /"
ffmpeg -loglevel error -i out/${id}_flast.png -i $last -lavfi "[0]scale=1280:720[a];[1]scale=1280:720[b];[a][b]ssim" -f null - 2>&1 | grep -o "All:[0-9.]*" | sed "s/^All:/last SSIM /" || true
# audio must be absent
ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 $vid | sed "s/^/AUDIO STREAM: /" || true
