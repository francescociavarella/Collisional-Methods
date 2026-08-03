# #!/bin/bash

# # ==========================================================
# # PARALLEL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# # ==========================================================

# # Define the mode: 'normal' or 'close_to_90'
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

# echo "Starting automated parallel batch processing..."
# echo "Selected Mode: $MODE"

# # Loop through each angle in the array
# for THETA in "${ANGLES[@]}"
# do
#     echo "Launching Python scripts for Theta = $THETA in background..."
    
#     # Pass THETA as the first argument ($1) and MODE as the second argument ($2)
#     # The '&' runs them concurrently in the background
#     #python -u Intermediate/Plot_rho_complete.py $THETA $MODE &
#     #python -u Complete_Fidelity_and_Trace_Distance_Analysis.py $THETA $MODE &
#     python -u Complete_Sx_Sy_Sz_exp_value_analysis.py $THETA $MODE &
# done

# # Wait for all background processes to finish before exiting the script
# wait

# echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"

#!/bin/bash

# ANGLES=(0.0 30.0 45.0 60.0 90.0)
ANGLES=(89.9 89.7 89.5 89 88.5 88 87 86)

PYTHON_SCRIPT="generic_rho_trajectories.py" 
PLOT_SCRIPT="Plot_generic.py"    
EIGENSTATE_SCRIPT="Eigenstate_Analysis.py"
HEATMAP_SCRIPT="Heatmap_Tomography.py"      
VARIANCE_SCRIPT="Variance_and_Trace_Distance_Analysis.py" 
TIME_SLICE_SCRIPT="Time_Slice_Distribution.py"
TOTAL_VARIANCE_SCRIPT="Total_Variance_Analysis.py"

export NUMBA_NUM_THREADS=12  # 12/number of angles to run in parallel (adjust as needed)

echo "Starting automated parallel batch processing..."
echo "Number of Numba threads per job set to: $NUMBA_NUM_THREADS"

# Loop attraverso gli angoli
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python scripts for Theta = $THETA in background..."
    
    # (python -u $PYTHON_SCRIPT $THETA "batch_run" && python -u $PLOT_SCRIPT $THETA && python -u $HEATMAP_SCRIPT $THETA && python -u $VARIANCE_SCRIPT $THETA && python -u $TIME_SLICE_SCRIPT $THETA )  # & (add & for parallel execution, may be too expensive)
    python -u $TOTAL_VARIANCE_SCRIPT $THETA # & 

done

wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"