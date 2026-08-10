#!/bin/bash
# Import data from ~/Downloads/comp-5434-2526-sem-3-project.zip

set -e

ZIP_FILE="$HOME/Downloads/comp-5434-2526-sem-3-project.zip"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

mkdir -p "$DATA_DIR"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: $ZIP_FILE not found"
    echo "Please download the data from Kaggle first."
    exit 1
fi

echo "Extracting data from $ZIP_FILE to $DATA_DIR..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR"

# Move files from subdirectory if needed
if [ -d "$DATA_DIR/comp-5434-2526-sem-3-project" ]; then
    mv "$DATA_DIR/comp-5434-2526-sem-3-project"/* "$DATA_DIR/"
    rmdir "$DATA_DIR/comp-5434-2526-sem-3-project"
fi

echo "Done. Files in data directory:"
ls -la "$DATA_DIR"
