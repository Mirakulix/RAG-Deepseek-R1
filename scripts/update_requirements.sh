#!/bin/bash

# Store the current working directory
ROOT_DIR=$(pwd)

# Find and process all requirements.txt files in subdirectories
find . -name "requirements.txt" -type f | while read -r file; do
    DIR=$(dirname "$file")
    echo "Processing $file in directory $DIR"
    cd "$DIR" || continue
    # Create backup of existing requirements.txt
    cp requirements.txt requirements.txt.backup
    echo "Created backup: ${file}.backup"
    
    # Change to the directory containing requirements.txt
    cd "$DIR" || continue
    
    # Generate new requirements.txt using pip freeze
    pip freeze > "requirements.txt"
    pip freeze >> "$ROOT_DIR/requirements.txt"
    echo "Updated requirements.txt in $DIR"


    
    # Return to root directory
    cd "$ROOT_DIR" || exit
done

# Handle root directory requirements.txt
if [ -f "$ROOT_DIR/requirements.txt" ]; then
    echo "Processing root directory requirements.txt"
    cd $ROOT_DIR || continue
    # Update root requirements.txt
    pip freeze > "$ROOT_DIR/requirements.txt"
    echo "Updated root requirements.txt"
fi

echo "Requirements processing completed"--upgrade pip
