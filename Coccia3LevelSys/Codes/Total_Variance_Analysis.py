#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# ==========================================
# 1. LETTURA ARGOMENTI DA BASH
# ==========================================
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1])
else:
    print("Nessun angolo fornito. Esecuzione di test con phi = 90.0")
    phi_deg = 90.0

# CONVERSIONE IN RADIANTI (perché i file sono salvati così!)
phi_rad = np.radians(phi_deg)

# Formattazione della stringa usando i RADIANTI con 4 cifre decimali
phi_str = f"{phi_rad:.4f}".replace(".", "p")

print("=" * 50)
print(f"ANALISI VARIANZA TOTALE - Angolo: {phi_deg}° (File cercato: phi{phi_str})")
print("=" * 50)

# ==========================================
# 2. CARICAMENTO DATI
# ==========================================
results_dir = "../Results/Data/Complete_rho/"

# Cerca i file che corrispondono esattamente all'angolo richiesto
search_pattern = os.path.join(results_dir, f"result_phi{phi_str}_*.npz")
npz_files = glob.glob(search_pattern)

if not npz_files:
    raise FileNotFoundError(f"Nessun file trovato in {results_dir} con il pattern:\n{search_pattern}")

# Se ci sono più file (es. con dt o N_traj diversi), prende l'ultimo generato
file_path = max(npz_files, key=os.path.getctime)
print(f"Caricamento dati dal file:\n -> {os.path.basename(file_path)}")

data = np.load(file_path)
rho_tot_all = data['rho_tot_all'] # Forma: (3, 3, n_times, N_traj)
times = data['times']
N_traj = data['N_traj']
dt = data['dt']

# Estrazione della dinamica esatta per confronto (Lindblad Master Equation)
# Shape attesa: (n_times, N, N)
if 'rho_list_lindblad' in data:
    rho_exact = data['rho_list_lindblad']
else:
    print("Warning: 'rho_list_lindblad' non trovata nel file npz. Confronto teorico non disponibile.")
    rho_exact = None

n_sites = rho_tot_all.shape[0]

# ==========================================
# 3. CREAZIONE CARTELLA DI OUTPUT
# ==========================================
# Crea una cartella specifica per l'angolo analizzato
output_base = "../Results/Plot/Total_Variance/"
output_dir = os.path.join(output_base, f"Phi_{phi_deg}")
os.makedirs(output_dir, exist_ok=True)
print(f"Cartella di output creata/verificata:\n -> {output_dir}")

# ==========================================
# 4. CALCOLO DELLA VARIANZA TOTALE
# ==========================================
mean_pop = np.zeros((n_sites, len(times)))
var_C = np.zeros((n_sites, len(times)))
var_Q = np.zeros((n_sites, len(times)))
var_Tot = np.zeros((n_sites, len(times)))

for site in range(n_sites):
    # Estraiamo la popolazione del sito m per tutti i tempi e le traiettorie
    pop = np.real(rho_tot_all[site, site, :, :])
    
    # Valore atteso totale (Popolazione media)
    mean_p = np.mean(pop, axis=1)
    mean_pop[site, :] = mean_p
    
    # VARIANZA CLASSICA/STATISTICA
    v_C = np.var(pop, axis=1, ddof=0)
    var_C[site, :] = v_C
    
    # VARIANZA QUANTISTICA
    v_Q = np.mean(pop - pop**2, axis=1)
    var_Q[site, :] = v_Q
    
    # VARIANZA TOTALE
    v_Tot = v_C + v_Q 
    var_Tot[site, :] = v_Tot

# Salvataggio numerico delle varianze
out_data_file = os.path.join(output_dir, f"variances_phi{phi_str}.npz")
np.savez_compressed(
    out_data_file,
    times=times,
    mean_pop=mean_pop,
    var_C=var_C,
    var_Q=var_Q,
    var_Tot=var_Tot,
    phi_deg=phi_deg,
    dt=dt,
    N_traj=N_traj
)
print(f"Dati numerici salvati in:\n -> {os.path.basename(out_data_file)}")

# ==========================================
# 5. PLOTTING & EXPORT DELLE FIGURE
# ==========================================
plt.style.use('default')
colors_pop = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Colori classici per le popolazioni

# ------------------------------------------
# FIGURA 1: Dinamica delle Popolazioni Medie
# ------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))

for site in range(n_sites):
    ax1.plot(times, mean_pop[site, :], label=f'Sito $|{site}\\rangle$', color=colors_pop[site], lw=2)

ax1.set_title(rf'Dinamica delle Popolazioni Medie ($\theta = {phi_deg}^\circ$)', fontsize=14)
ax1.set_xlabel('Tempo (fs)', fontsize=12)
ax1.set_ylabel('Popolazione $\\langle P_m \\rangle$', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()

fig1_path = os.path.join(output_dir, f"Populations_phi{phi_str}.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)

# ------------------------------------------
# FIGURA 2: Teorema della Varianza Totale
# ------------------------------------------
fig2, axes = plt.subplots(n_sites, 1, figsize=(10, 2.5 * n_sites), sharex=True)
if n_sites == 1:
    axes = [axes]

for site in range(n_sites):
    ax = axes[site]
    
    # Se abbiamo i dati teorici, plottiamo la linea tratteggiata nera di riferimento
    if rho_exact is not None:
        # Essendo un proiettore di popolazione, la varianza esatta è <P> - <P>^2
        pop_exact = np.real(rho_exact[:, site, site])
        var_exact = pop_exact - pop_exact**2
        ax.plot(times, var_exact, color='black', linewidth=3, linestyle='--', label='Varianza Totale (Lindblad)')

    # Andamenti delle equazioni della varianza
    ax.plot(times, var_C[site, :], color='red', linewidth=2, alpha=0.8, label='Varianza Statistica ($Var_C$)')
    ax.plot(times, var_Q[site, :], color='blue', linewidth=2, alpha=0.8, label='Varianza Quantistica ($Var_Q$)')
    ax.plot(times, var_Tot[site, :], color='limegreen', linewidth=3, linestyle=':', label='Somma ($Var_C + Var_Q$)')
    
    ax.set_ylabel(f'Sito $|{site}\\rangle$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if site == 0:
        ax.set_title(rf'Teorema della Varianza Totale ($\theta = {phi_deg}^\circ$)', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)

axes[-1].set_xlabel('Tempo (fs)', fontsize=12)
fig2.tight_layout()

fig2_path = os.path.join(output_dir, f"Variance_Decomposition_phi{phi_str}.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)

print(f"Grafici PNG salvati in:\n -> {output_dir}")
print("=" * 50)
print("ANALISI COMPLETATA CON SUCCESSO!")