#! /usr/bin/bash
base_dir="/mnt/c/Users/taunsquared/Dropbox/CuttleShuttle/CuttleShuttle-VideoDataset-Raw/"
for animal in "$base_dir"/*
do
    animal_name=$(basename "$animal")
    echo "$animal_name"
    for date_folder in "$animal"/*
	do
        for file in "$date_folder"/*
        do
            filename=$(basename "$file")
            filepath=$(dirname "$file")
            if [[ $filename == *.csv ]] || [[ $filename == *.png ]]; then
                if [[ $filename != $animal_name*.csv ]] || [[ $filename == $animal_name*.png ]]; then
                    #echo "old name: $file"
                    #echo "new name: $filepath"/"$animal_name"_"$filename"
                    #echo "Updating name of $filename..."
                    cp "$file" "$filepath"/"$animal_name"_"$filename"
                fi
            fi
        done
	done
done