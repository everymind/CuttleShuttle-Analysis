#! /usr/bin/bash
base_dir="/mnt/c/Users/taunsquared/Dropbox/CuttleShuttle/CuttleShuttle-VideoDataset-Raw/"
for animal in "$base_dir"/*
do
    for file in "$animal"/*"_all_sessions_full_vids"/*
	do
		filename=$(basename -s .avi "$file")
		filepath=$(dirname "$file")
		if [[ $filename == s* ]]; then
			echo "Updating name of $filename..."
			cp "$file" "$filepath"/$(basename "$animal")_"$filename".avi
		fi
	done
done