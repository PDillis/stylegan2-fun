#!/bin/bash

# Run in a terminal:
# ./projection_video.sh 5 real

# 5 will be the run number and on it, real images were projected using the official
# repository (e.g., you have, in the results subdirectory, 00005-project-real-images)

# Some nice bash editing commands:
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Check if there are exactly 2 arguments by the user, otherwise, show usage:
if [ $# -ne 1 ] || ! [ "$1" -eq "$1" ] 2> /dev/null; then
	echo -e "Wrong arguments! Usage: \n\t${BOLD}\$ ./projection_video.sh n${NORMAL} \nwhere ${BOLD}n${NORMAL}, is the run number (an integer: 0, 1, 2, ...)."
	exit 1
fi

# Run dir:
printf -v NRUN "%05d" "$1"

REAL="$PWD"/results/"$NRUN"-project-real-images
GEN="$PWD"/results/"$NRUN"-project-generated-images

# We will check which of these paths actually exist and assign RUN to it; this
# way, we will avoid having to ask the user to input it:
if [ -d "$REAL" ]; then
	RUN="$REAL"
elif [ -d "$GEN" ]; then
	RUN="$GEN"
else
	echo -e "The directories ${RED}"$REAL"${NC} and ${RED}"$GEN"${NC} do not exist. Check if your run ${BLUE}${BOLD}"$NRUN"${NORMAL}${NC} was actually a projection and not something else."
	exit 1
fi

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# We need all the images that were projected (real will have img, generated will have seed):
IMGS=($(find "$RUN" -name '*step0001.png' -exec basename \{} .png \; | cut -d'-' -f 1))

# Thus, for all of the images:
for IMG in "${IMGS[@]}"
do
	# Result dir, which we then move the 1000 projection images to:
	RES="$RUN"/"$IMG"
	mkdir "$RES"
	mv "$RUN"/"$IMG"*.png "$RES"

	# Number of frames will determine our framerate (default: 1000 frames at 50 fps: 20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))

	# Make left video, with bottom box printing frame number (can be easily removed):
	echo -e "\vMaking \vleft \vvideo \vof \v$IMG"
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$IMG"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video of the static target image:
	echo -e "\vMaking \vright \vvideo \vof \v$IMG"
	ffmpeg -y -loop 1 -i "$RES"/"$IMG"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Target image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t "$DUR" -pix_fmt yuv420p "$RES"/right.mp4

	# Combine left and right videos:
	echo -e "\vCombining \vvideos \vof \v$IMG"
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 20 -preset veryfast "$RES"/"$IMG"-projecting.mp4

	# Delete right and left videos (remove if you wish to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
