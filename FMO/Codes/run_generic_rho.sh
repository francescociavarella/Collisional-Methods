#!/bin/bash

ANGLES=(0.0 45.0 90.0)

PYTHON_SCRIPT="generic_rho_trajectories.py" 
PLOT_SCRIPT="Plot_generic.py"    
HEATMAP_SCRIPT="Heatmap_Tomography.py"       

export NUMBA_NUM_THREADS=6  # 12/number of angles to run in parallel (adjust as needed)

echo "Starting automated parallel batch processing..."
echo "Number of Numba threads per job set to: $NUMBA_NUM_THREADS"

# Loop attraverso gli angoli
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python scripts for Theta = $THETA in background..."
    
    # ( python -u $PYTHON_SCRIPT $THETA "batch_run" && python -u $PLOT_SCRIPT $THETA && python -u $HEATMAP_SCRIPT $THETA ) # &   (add & for parallel execution, may be too expensive)
    python -u $HEATMAP_SCRIPT $THETA "batch_run" # &

done

wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"