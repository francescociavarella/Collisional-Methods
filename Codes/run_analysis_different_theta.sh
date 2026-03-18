#!/bin/bash

# ==========================================================
# BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

# Define the array of angles to process
ANGLES=(0 15 30 45 60 75 90)

echo "Starting automated batch processing..."

# Loop through each angle in the array
for THETA in "${ANGLES[@]}"
do
    echo "------------------------------------------------"
    echo "Executing Python script for Theta = $THETA"
    echo "------------------------------------------------"
    
    # Call the Python script and pass the angle as an argument
    python process_single_theta.py $THETA
    
    # Optional: check if the Python script crashed and stop the loop if it did
    if [ $? -ne 0 ]; then
        echo "❌ Error detected during Theta = $THETA. Stopping the batch."
        exit 1
    fi
done

echo "🎉 ALL ANGLES PROCESSED SUCCESSFULLY!"
