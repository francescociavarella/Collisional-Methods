#!/bin/bash

ANGLES=(0.0 0.1 0.5 1.0 45.0 90.0)
# ANGLES=(0.1 0.5 1.0)

PYTHON_SCRIPT="generic_rho_trajectories.py" 
PLOT_SCRIPT="Plot_generic.py"    
EIGENSTATE_SCRIPT="Eigenstate_Analysis.py"
HEATMAP_SCRIPT="Heatmap_Tomography.py"      
VARIANCE_SCRIPT="Variance_and_Trace_Distance_Analysis.py" 
TIME_SLICE_SCRIPT="Time_Slice_Distribution.py"
TOTAL_VARIANCE_SCRIPT="Total_Variance_Analysis.py"
TRACE_DISTANCE_SCALING="Trace_Distance_Scaling.py"

export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMBA_NUM_THREADS=1 # 12/number of angles to run in parallel (adjust as needed)

echo "Starting automated parallel batch processing..."
echo "Number of Numba threads per job set to: $NUMBA_NUM_THREADS"

# Loop attraverso gli angoli
for THETA in "${ANGLES[@]}"
do
    echo "Launching Python scripts for Theta = $THETA in background..."
    
    # (python -u $PYTHON_SCRIPT $THETA "batch_run" && python -u $PLOT_SCRIPT $THETA && python -u $HEATMAP_SCRIPT $THETA && python -u $VARIANCE_SCRIPT $THETA && python -u $TIME_SLICE_SCRIPT $THETA )  # & (add & for parallel execution, may be too expensive)
    # python -u $VARIANCE_SCRIPT $THETA # & 
    python -u $TRACE_DISTANCE_SCALING
done

wait

echo "ALL PARALLEL CALCULATIONS COMPLETED SUCCESSFULLY"