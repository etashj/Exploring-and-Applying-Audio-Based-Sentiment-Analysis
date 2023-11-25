#!/bin/bash

# Set the input and output directories
input_dir="/Users/etashjhanji/Documents/GitHub/intelligent-emotion-based-music-selection-via-cnn/data/clips_45seconds_wav"
output_dir="/Users/etashjhanji/Documents/GitHub/intelligent-emotion-based-music-selection-via-cnn/data/clips_45seconds_wav_resamp"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all WAV files in the input directory
for input_file in "$input_dir"/*.wav; do
    # Extract the file name without the extension
    filename=$(basename -- "$input_file")
    filename_no_ext="${filename%.*}"

    # Set the output file path
    output_file="$output_dir/$filename_no_ext.wav"

    # Use ffmpeg to resample the audio to 44.1 kHz
    ffmpeg -i "$input_file" -ar 44100 "$output_file"

    echo "Resampled $filename to $output_file"
done
