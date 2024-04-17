#!/bin/bash

# Directory containing zip files
directory="."

# Function to unzip files
do_unzip() {
    zip_file="$1"
    unzip -o "$zip_file" -d "${zip_file%.*}"  # Extract into a folder named after the zip file
}

export -f do_unzip  # Export the function for parallel to use

# Count total files for progress estimation
total_files=$(find "$directory" -name '*.zip' -type f | wc -l)

# Find all zip files in the directory and pass them to parallel
find "$directory" -name '*.zip' -type f | parallel --eta -j 8 --bar do_unzip {} 

echo "All files have been extracted."
