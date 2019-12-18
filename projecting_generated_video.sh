#!/bin/bash

# input: ./projecting_generated_video.sh 5

# 5 will be the run number (e.g., 00005-project-generated-images)

# Run dir:
printf -v RUN "%05d" "$1"
RUN="$PWD"/results/"$RUN"-project-generated-images

# Duration of video (at 1000 frames, default gives 50fps; change at your own risk):
DUR=20

# Number of images run
NUMIMAGES=$(ls "$RUN" | grep step | wc -l)
END=$(("$NUMIMAGES"/1000 - 1))

# Start of our loop
START=0

#e.g., if we have 5000 images, we projected 5 images, so our loop will be from 0 to 4
# Thus, for each image, from image0000 to image0004:
for (( c=$((10#$START)); c<=$((10#$END)); c++ ))
do
    # We reassign c to be, from 0 to 0000
    printf -v c "%04d" "$c"
    # This will be what the file names have in common
    IMG=image"$c"
	# Result dir, which we will move all our images to:
	RES="$RUN"/"$IMG"
	mkdir "$RES"
	mv "$RUN"/"$IMG"*.png "$RES"

    # Number of frames will determine our framerate (20 second video):
	NUMFRAMES=$(($(ls -lR "$RES"/*.png | wc -l) - 1)) # we remove the target image
	FRAMERATE=$(("$NUMFRAMES" / "$DUR"))

    # Make left video, with bottom box printing frame number (can be easily removed):
    echo -e "\vMaking \vleft \vvideo \vof \v$IMG"
	ffmpeg -y -framerate "$FRAMERATE" -i "$RES"/"$IMG"-step%04d.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Step\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/left.mp4

    # Make right video:
    echo -e "\vMaking \vright \vvideo \vof \v$IMG"
	ffmpeg -y -loop 1 -i "$RES"/"$IMG"-target.png -vf "drawtext=fontfile=DejaVuSansMono.ttf: text='Generated image': x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:v libx264 -t "$DUR" -profile:v high -crf 20 -pix_fmt yuv420p "$RES"/right.mp4

    # Combine left and right videos:
    echo -e "\vCombining \vvideos \vof \v$IMG"
	ffmpeg -y -i "$RES"/left.mp4 -i "$RES"/right.mp4 -filter_complex '[0:v][1:v]hstack[vid]' -map [vid] -c:v libx264 -crf 20 -preset veryfast "$RES"/"$IMG"-projecting.mp4

	# Delete right and left videos (remove if you wish to keep them):
	rm -f "$RES"/left.mp4 "$RES"/right.mp4
done
