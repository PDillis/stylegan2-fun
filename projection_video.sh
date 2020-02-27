#!/bin/bash

# Run in a terminal:
# ./projection_video.sh 5

# 5 will be the run number where either real or generated images were projected
# using the official repository (e.g., you have, in the results subdirectory,
# 00005-project-real-images or 00005-project-generated-images)

# Some nice bash text editing commands:
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Check if there is exactly 1 argument by the user, otherwise, show usage:
if [ $# -ne 1 ] || ! [ "$1" -eq "$1" ] 2> /dev/null; then
	echo -e "Wrong arguments! Usage: \n\t${BOLD}\$ ./projection_video.sh n${NORMAL} \nwhere ${BOLD}n${NORMAL}, is the run number (an integer: 0, 1, 2, ...)."
	exit 1
fi

# Get the run number dir:
printf -v NRUN "%05d" "$1"

# The projection was of either real or generated images, so we will check which
# of these actually exists:
REAL="$PWD"/results/"$NRUN"-project-real-images
GEN="$PWD"/results/"$NRUN"-project-generated-images

# Whichever does exist will be our RUN:
if [ -d "$REAL" ]; then
	RUN="$REAL"
    echo -e "Projecting a real image!"
elif [ -d "$GEN" ]; then
	RUN="$GEN"
    echo -e "Projecting a generated image!"
else
	echo -e "The directories ${RED}"$REAL"${NC} and ${RED}"$GEN"${NC} do not exist. Check if your run ${BLUE}${BOLD}"$NRUN"${NORMAL}${NC} was actually a projection and not something else."
	exit 1
fi

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# We need all the image names that were projected; while real will have img and
# generated will have seed, they will all have in common *step0001.png, so that
# is what we will look for and extract the first part to get all the image names:
IMGS=($(find "$RUN" -name '*step0001.png' -exec basename \{} .png \; | cut -d'-' -f 1))

# Thus, for all of the images:
for IMG in "${IMGS[@]}"
do
	# Result dir, which we then move the 1000 projection images to:
	RES="$RUN"/"$IMG"
	mkdir "$RES"
	mv "$RUN"/"$IMG"*.png "$RES"

	# Number of frames will determine our framerate (default: 1000 frames at
	# 50 fps: 20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))

	# Make left video, with bottom box printing frame number (text can be
    # removed by deleting the -vf "drawtext=..." flag):
# 	echo -e "\vMaking \vleft \vvideo \vof \v$IMG"
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$IMG"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -preset veryfast -c:v libx265 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video of the static Target image (text can be removed by
    # deleting the -vf "drawtext=..." flag):
	echo -e "\vMaking \vright \vvideo \vof \v$IMG"
    TEXT=""$NAME" image"
	ffmpeg -y -loop 1 -i "$RES"/"$IMG"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Target Image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -preset veryfast -c:v libx265 -t "$DUR" -pix_fmt yuv420p "$RES"/right.mp4

	# Combine left and right videos:
	echo -e "\vCombining \vvideos \vof \v$IMG"
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 20 -preset veryfast "$RES"/"$IMG"-projection.mp4

	# Delete right and left videos (remove line if you wish to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
