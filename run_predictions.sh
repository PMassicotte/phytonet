#!/bin/bash

# Script to run phytonet-predict on all mission dataset folders
# Each folder gets its own CSV prediction file

# Configuration
BASE_DIR="/media/4TB/phyto_imgs/01_mission_datasets/"
MODEL_PATH="data/models/20250802-120929/best_model_20250802_135920_epoch13_acc0.95.pth"

# Get list of directories (folders only, not files)
for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        # Extract folder name without path
        folder_name=$(basename "$folder")

        # Define output CSV file
        output_file="${folder_name}.csv"

        echo "Processing folder: $folder_name"
        echo "Input: $folder"
        echo "Output: $output_file"
        echo "----------------------------------------"

        # Run the prediction command with verbose output
        uv run phytonet-predict "$folder" \
            --model-path "$MODEL_PATH" \
            --output "$output_file"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed $folder_name"
        else
            echo "✗ Failed to process $folder_name"
        fi

        echo ""
    fi
done

echo "All predictions complete."
