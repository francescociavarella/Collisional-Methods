#!/bin/bash

# ==========================================================
# PARALLEL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

# Define the array of angles to process
ANGLES=(0 30 45 60 90)   # normal
# ANGLES=(90 89.9 89.5 89 88 87))  # Close to 90 deg

echo "Starting automated parallel batch processing..."

# Loop through each angle in the array
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python script for Theta = $THETA in background..."
    
    # Use python -u to force standard output to be unbuffered
    # The '&' at the end pushes the job to the background (parallel execution)
    #python -u Complete_Fidelity_and_Trace_Distance_Analysis.py $THETA &
    python -u Sx_Sy_Sz_exp_value_analysis.py $THETA &
done

# Wait for all background processes to finish before exiting
wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"
