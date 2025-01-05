#!/usr/bin/env bash
#
# Usage: create_video.sh <image_folder> <output_video>

# Check for exactly 2 arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <image_folder> <output_video>"
  exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_VIDEO="$2"

# Run ffmpeg to create a 60fps video from images in INPUT_FOLDER.
# This example assumes .png images. Adjust the extension if needed.
ffmpeg \
  -framerate 30 \
  -pattern_type glob -i "${INPUT_FOLDER}/*.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  "${OUTPUT_VIDEO}"
