#!/bin/bash

# input: ./projecting_generated_video.sh 5

# 5 will be the run number (e.g., 00005-project-generated-images)

# Run dir:
printf -v RUN "%05d" "$1"
RUN="$PWD"/results/"$RUN"-project-generated-images

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# We need all the seeds that were run before entering the loop:
SEEDS=($(find "$RUN" -name '*step0001.png' -exec basename \{} .png \; | cut -d'-' -f 1))

# Thus, for each seed:
for SEED in "${SEEDS[@]}"
do
	# Result dir, which we will move all our images to:
	RES="$RUN"/"$SEED"
	mkdir "$RES"
	mv "$RUN"/"$SEED"*.png "$RES"

	# Number of frames will determine our framerate (20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))

	# Make left video, with bottom box printing frame number (can be easily removed):
	echo -e "\vMaking \vleft \vvideo \vof \v$SEED"
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$SEED"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video:
	echo -e "\vMaking \vright \vvideo \vof \v$SEED"
	ffmpeg -y -loop 1 -i "$RES"/"$SEED"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Generated image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t "$DUR" -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/right.mp4

	# Combine left and right videos:
	echo -e "\vCombining \vvideos \vof \v$SEED"
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 20 -preset veryfast "$RES"/"$SEED"-projecting.mp4

	# Delete right and left videos (remove if you wish to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
