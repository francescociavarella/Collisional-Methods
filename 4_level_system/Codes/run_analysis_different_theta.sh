#!/bin/bash

# ==========================================================
# PARALLEL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

# Define the mode: 'normal' or 'close_to_90'
# MODE="close_to_90"

# # Select the appropriate array of angles based on the chosen mode
# if [ "$MODE" == "normal" ]; then
#     ANGLES=(0 30 45 60 90)
# elif [ "$MODE" == "close_to_90" ]; then
#     ANGLES=(0 90 89.9 89.7 89.5 89 88.5 88 87 86)
# else
#     echo "Error: Unknown MODE = $MODE"
#     exit 1
# fi

# ANGLES=(0 0.001 0.005 0.1 0.2 0.5 0.7 1 3)
ANGLES=(0.1 0.2 0.5 0.7 1 3)

# Loop through each angle in the array
for THETA in "${ANGLES[@]}"
#do
#    echo "Launching Python scripts for Theta = $THETA in background..."
#    
    # Raggruppa calcolo e plot in un unico job in background per ogni angolo
#    (
#       python -u Generic_Populations_Analysis.py $THETA
#        python -u Generic_Fidelity_and_Trace_Distance_Analysis.py $THETA
#    ) &

#done

do
    echo "Launching Python scripts for Theta = $THETA in background..."
    
    # Run only the populations analysis in the background for each angle
    python -u Generic_Populations_Analysis.py $THETA &

done

# Wait for all background groups to finish
wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"