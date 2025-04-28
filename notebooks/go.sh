#!/bin/bash

RUNS_DIR="results/ROME_MODIFIED"  # Folder containing your runs
OUTPUT_DIR="results/ROME_MODIFIED"  # Where you want .txt files

for run_folder in "$RUNS_DIR"/*; do
    if [ -d "$run_folder" ]; then
        run_name=$(basename "$run_folder")
        output_file="${OUTPUT_DIR}/${run_name}.txt"

        echo "Running summarize for $run_name ..."
        python3 -m experiments.summarize --dir_name=ROME_MODIFIED --runs="$run_name" > "$output_file" &
    fi
done

wait  # Wait for all background processes to finish
echo "All summarizations completed."
