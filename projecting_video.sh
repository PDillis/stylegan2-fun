#!/bin/bash

# input: ./projecting_video.sh 1 seed0012 seed0482 .....

# $1 will be the run number, $2 onwards will be the different seeds that you have (presumably) run before.

# Run dir:
printf -v RUN "%05d" "$1"
RUN="$PWD"/results/"$RUN"-project-generated-images

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# Thus, for each seed:
for var in "${@:2}"
do
	# Result dir, which we will move all our images to:
	RES="$RUN"/"$var"
	mkdir "$RES"
	mv "$RUN"/"$var"*.png "$RES"

	# Number of frames will determine our framerate (20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))
	# Make left video, with bottom box printing frame number (can be easily removed):
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$var"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video
	ffmpeg -y -loop 1 -i "$RES"/"$var"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Generated image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t "$DUR" -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/right.mp4

	# Combine:
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 20 -preset veryfast "$RES"/"$var"-projecting.mp4

	# Delete right and left videos
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
