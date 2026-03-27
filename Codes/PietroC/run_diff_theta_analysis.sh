#!/bin/bash

# ==========================================================
# PARALLEL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

# Define the mode: 'normal' or 'close_to_90'
MODE="normal"

# Select the appropriate array of angles based on the chosen mode
if [ "$MODE" == "normal" ]; then
    ANGLES=(0 30 45 60 90)
elif [ "$MODE" == "close_to_90" ]; then
    ANGLES=(0 90 89.9 89.7 89.5 89 88.5 88 87 86)
else
    echo "Error: Unknown MODE = $MODE"
    exit 1
fi

echo "Starting automated parallel batch processing..."
echo "Selected Mode: $MODE"

# Loop through each angle in the array
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python scripts for Theta = $THETA in background..."
    
    # Pass THETA as the first argument ($1) and MODE as the second argument ($2)
    python -u Module_Analysis_Densification.py $THETA $MODE &
done

# Wait for all background processes to finish before exiting the script
wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"