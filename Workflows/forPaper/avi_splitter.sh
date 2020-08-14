#! /usr/bin/bash
base_dir="/mnt/c/Users/taunsquared/Dropbox/CuttleShuttle/CuttleShuttle-VideoDataset-Raw/L1-H2013-01"
maxsize=2500000000
output_dir="/mnt/c/Users/taunsquared/Documents/thesis/CuttleShuttle/HarvardDataverse/Videos"
#for animal in "$base_dir"/*
#do
if ((1>0)); then
    for file in "$base_dir"/*"_all_sessions_full_vids"/*
	do
		actualsize=$(wc -c <"$file")
		if ((actualsize>maxsize)); then
			filename=$(basename -s .avi "$file")
			echo "$filename"
			ffmpeg -i "$file" -ss 0 -t 900 -codec copy "$output_dir"/"$filename"_part1.avi
			ffmpeg -i "$file" -ss 900 -codec copy "$output_dir"/"$filename"_part2.avi
		fi
	done
fi
#done
# select videos greater than 2.5 gb
# for each video get duration, cut into the first 15 min, then the rest