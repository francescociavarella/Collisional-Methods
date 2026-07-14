#!/bin/bash

ANGLES=(0 45 90)

export NUMBA_NUM_THREADS=12

echo "Avvio dei calcoli paralleli sulla workstation..."

# Loop per ogni angolo
for THETA in "${ANGLES[@]}"
do
    echo "Lancio calcolo per Theta = $THETA gradi"
    
    #python -u CM_generic_rho_only.py $THETA &
    python -u Plot_generic.py $THETA &
done

# Aspetta che tutti i processi in background finiscano
wait

echo "TUTTI I CALCOLI SONO STATI COMPLETATI CON SUCCESSO"