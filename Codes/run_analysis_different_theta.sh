#!/bin/bash

# ==========================================================
# PARALLEL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

# Define the array of angles to process
ANGLES=(0 30 45 60 90)

echo "Starting automated parallel batch processing..."

# Loop through each angle in the array
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python script for Theta = $THETA in background..."
    
    # Use python -u to force standard output to be unbuffered
    # The '&' at the end pushes the job to the background (parallel execution)
    python -u Complete_Fidelity_and_Trace_Distance_Analysis.py $THETA &
done

# Wait for all background processes to finish before exiting
wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"
