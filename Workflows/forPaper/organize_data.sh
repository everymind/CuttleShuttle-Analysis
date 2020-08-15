#! /usr/bin/bash
base_dir="/mnt/c/Users/taunsquared/Dropbox/CuttleShuttle/CuttleShuttle-VideoDataset-Raw/"
for animal in "$base_dir"/*
do
    arduino="$animal"/$(basename "$animal")_arduino_behavior_code
    moi="$animal"/$(basename "$animal")_MOI_screenshots
    csv="$animal"/$(basename "$animal")_MOI_csv
    declare -a arr=("$arduino" "$moi" "$csv")
    for new_folder in "${arr[@]}"
    do
        if [[ -d "$new_folder" ]]; then
            echo "$(basename "$new_folder") already exists"
        else
            echo "creating "$new_folder"..."
            mkdir "$new_folder"
        fi
    done
    for folder in "$animal"/*
    do
        if [[ $(basename "$folder") != $(basename "$animal")* ]]; then
            for file in "$folder"/*
            do
                filename=$(basename "$file")
                filepath=$(dirname "$file")
                if [[ $filename == $(basename "$animal")_*.csv ]]; then
                    #echo "$csv"/"$filename"
                    echo "Moving $filename to folder $(basename "$csv")..."
                    mv "$file" "$csv"/"$filename"
                fi
                if [[ $filename == $(basename "$animal")_*.png ]]; then
                    #echo "$moi"/"$filename"
                    echo "Moving $filename to folder $(basename "$moi")..."
                    mv "$file" "$moi"/"$filename"
                fi
                if [[ $filename == *_$(basename "$animal").txt ]]; then
                    #echo "$arduino"/"$filename"
                    echo "Copying $filename to folder $(basename "$arduino")..."
                    cp "$file" "$arduino"/"$filename"
                fi
            done
        fi
    done
done