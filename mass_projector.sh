#!/bin/bash

# Run in a terminal:
# ./mass_projector.sh name-of-dataset /dataset/root/path /models/root/path N

# Some nice bash editing commands:
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# If wrong number of arguments:
if [ $# -ne 3 ]; then
	# Print usage:
	echo -e "Wrong amount of arguments! Usage: \n\t${BOLD}\$ ./mass_projector.sh name-of-dataset /dataset/root/path /model/root/path N${NORMAL} \nwhere ${BOLD}name-of-dataset${NORMAL} is the name of the dataset to project located at ${BOLD}/dataset/root/path${NORMAL} and ${BOLD}N${NOMRAL} is the number of images per model tick. ${BOLD}/model/root/path${NORMAL} will be the root to all your pkl files."
	exit 1
else
	# Note that if the user provides the wrong arguments, the projector won't run
	# properly:
	DATASETNAME="$1"
	DATASETROOT="$2"
    IMGPERTICK="$3"
	# For all pkl files in the model root path, run the projector of StyleGAN2:
	for filename in "$3"/*.pkl; do
		# I want the name of the model just to print it out for the user:
		fname=$(basename -- "$filename")
		echo -e "${BOLD}Projecting images with model: " "${RED}$fname${NC}${NORMAL}"
		# Rename the absolute path to the pkl file:
		printf -v PKLPATH `readlink -f $filename`
		# We run the projector for each of these pkl files (you can change --num-snapshots, but for my purposes, I only need 1):
		python run_projector.py project-real-images --network="$PKLPATH" --dataset="$DATASETNAME" --data-dir="$DATASETROOT" --num-snapshots 1 --num-images "$IMGPERTICK"
	done

fi
