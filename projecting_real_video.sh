#!/bin/bash

# Run in a terminal:
# ./projecting_real_video.sh 5

# 5 will be the run number (e.g., you have, in the results subdirectory, 00005-project-real-images)

# Run dir:
printf -v RUN "%05d" "$1"
RUN="$PWD"/results/"$RUN"-project-real-images

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# We need all the images that were projected:
IMGS=($(find "$RUN" -name '*step0001.png' -exec basename \{} .png \; | cut -d'-' -f 1))

# Thus, for each image:
for IMG in "${IMGS[@]}"
do
	# Result dir, which we then move all our images to:
	RES="$RUN"/"$IMG"
	mkdir "$RES"
	mv "$RUN"/"$IMG"*.png "$RES"

	# Number of frames will determine our framerate (default: 1000 frames at 50 fps: 20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))

	# Make left video, with bottom box printing frame number (can be easily removed):
	echo -e "\vMaking \vleft \vvideo \vof \v$IMG"
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$IMG"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video:
	echo -e "\vMaking \vright \vvideo \vof \v$IMG"
	ffmpeg -y -loop 1 -i "$RES"/"$IMG"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Real image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t "$DUR" -pix_fmt yuv420p "$RES"/right.mp4

	# Combine left and right videos:
	echo -e "\vCombining \vvideos \vof \v$IMG"
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast "$RES"/"$IMG"-projecting.mp4

	# Delete right and left videos (remove if you wish to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
