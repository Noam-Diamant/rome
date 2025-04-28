#!/bin/bash

RUNS_DIR="results/ROME_MODIFIED"
OLD_SUMMARY_DIR="results/ROME_MODIFIED/summaries"
NEW_SUMMARY_DIR="results/ROME_MODIFIED/summaries/new"

mkdir -p "$NEW_SUMMARY_DIR"  # Ensure the new summaries folder exists

for run_folder in "$RUNS_DIR"/*; do
    if [ -d "$run_folder" ]; then
        run_name=$(basename "$run_folder")
        old_summary_file="${OLD_SUMMARY_DIR}/${run_name}.txt"
        new_summary_file="${NEW_SUMMARY_DIR}/${run_name}.txt"

        # Check if summary exists in either old or new summary directories
        if [ ! -f "$old_summary_file" ] && [ ! -f "$new_summary_file" ]; then
            echo "Summarizing $run_name ..."
            python3 -m experiments.summarize --dir_name=ROME_MODIFIED --runs="$run_name" > "$new_summary_file" &
        else
            echo "Summary already exists for $run_name. Skipping."
        fi
    fi
done

wait  # Wait for all background summarize jobs to finish
echo "All summarizations completed."
