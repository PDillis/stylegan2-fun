#!/bin/bash

# input: ./projecting_real_video.sh 1 image0012 image0482 .....

# $1 will be the run number, $2 onwards will be the different images that you have (presumably) run before.

# Run dir:
NUM=${1?Error: No run number given}
printf -v RUN "%05d" "$NUM"
RUN="$PWD"/results/"$RUN"-project-real-images

# Thus, for each image:
for var in "${@:2}"
do
	# Result dir, which we will move all our images to:
	RES="$RUN"/"$var"
	mkdir "$RES"
	mv "$RUN"/"$var"*.png "$RES"

	# Make left video, with bottom box printing frame number (can be easily removed):
	ffmpeg -y -framerate 50 -i "$RES"/"$var"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

	# Make right video:
	ffmpeg -y -loop 1 -i "$RES"/"$var"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Real image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t 20 -pix_fmt yuv420p "$RES"/right.mp4

	# Combine left and right videos:
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 22 -preset veryfast "$RES"/"$var"-projecting.mp4

	# Delete right and left videos (remove if you want to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4

done
