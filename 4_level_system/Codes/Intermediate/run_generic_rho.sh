#!/bin/bash

ANGLES=(90)

export NUMBA_NUM_THREADS=12

echo "Avvio dei calcoli e dei plot in sequenza sulla workstation..."

# Loop per ogni angolo
for THETA in "${ANGLES[@]}"
do
    echo "Lancio calcolo e plot per Theta = $THETA gradi"
    
    # Esegue prima il calcolo e, appena ha finito, lancia in automatico il plot
    ( python -u CM_generic_rho_only.py $THETA && python -u Plot_generic.py $THETA ) &
done

# Aspetta che tutti i processi in background finiscano
wait

echo "TUTTI I CALCOLI E I PLOT SONO STATI COMPLETATI CON SUCCESSO"