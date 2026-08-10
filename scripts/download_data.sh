#!/bin/bash
# Download data from Kaggle API

set -e

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
COMPETITION="comp-5434-2526-sem-3-project"

mkdir -p "$DATA_DIR"

echo "Downloading data from Kaggle..."
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

ZIP_FILE="$DATA_DIR/$COMPETITION.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "Extracting..."
    unzip -o "$ZIP_FILE" -d "$DATA_DIR"
    rm "$ZIP_FILE"
fi

echo "Done. Files in data directory:"
ls -la "$DATA_DIR"
