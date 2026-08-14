#!/bin/bash

# ==========================================================
# SEQUENTIAL BASH SCRIPT TO AUTOMATE PYTHON ANALYSIS
# ==========================================================

ANGLES=(90 0 30)
# ANGLES=(0)


echo "Starting automated sequential batch processing..."

for THETA in "${ANGLES[@]}"
do
    echo "Running Python scripts sequentially for Theta = $THETA..."
    # Run the calculation, then the plot only if it succeeds
    # python -u CM_generic_rho_only.py $THETA && python -u Plot_generic.py $THETA
    python -u Plot_generic.py $THETA
done

echo "ALL SEQUENTIAL CALCULATIONS COMPLETED SUCCESSFULLY"