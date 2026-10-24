#!/bin/bash
# frame sequences from the original campus footage (~/footage/campus) for the three.js compositor: JPEG q2, max 4096 wide, native fps
cd ~/footage/campus
OUT=~/Movies/ZAFU2/footage
ex() {  # name src start dur
  [ -f "$OUT/$1/done" ] && { echo "skip $1"; return; }
  mkdir -p "$OUT/$1"
  nice -n 10 ffmpeg -v error -y -ss "$3" -t "$4" -i "$2" -vf "scale='min(4096,iw)':-2" -q:v 2 "$OUT/$1/%05d.jpg" && touch "$OUT/$1/done"
  echo "$1 $(ls $OUT/$1/*.jpg | wc -l) frames"
}
ex sprout_clover "三叶草4.mov" 0 10.84
ex avenue_season "10.mov" 0 4.6
ex dive_library DJI_0238.MP4 0 120
ex topdown_lake DJI_0208.MP4 0 72
ex macro_narcissus "水仙花.mov" 0 14.2
ex macro_pinecone "松果特写.mov" 0 3.2
ex macro_drop "水滴.mov" 0 3.6
ex stars_library 111.mp4 0 9.875
ex sunset_library "日落图啊书馆.mov" 0 8.44
ex orbit_library DJI_0285.MP4 0 111.6
ex swans DSCF9176.MOV 0 45
ex avenue_push zhangshutuijin.mov 0 6.04
ex donghu_sunrise "东湖日出.mov" 0 19.1
ex hill_sunrise "日出03.mov" 0 16.84
ex grass_light "草地光影.mov" 0 15.9
ex campus_aerial DJI_0280.MP4 0 15.8
ex sky_clouds "西湖44.mov" 0 6.88
echo EXTRACT_DONE
