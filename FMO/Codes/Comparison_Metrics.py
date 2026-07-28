#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import curve_fit

# ==========================================
# FUNZIONI MATEMATICHE E DI FIT
# ==========================================

@njit(cache=True)
def trace_distance_njit(rho, sigma):
    """
    Calcola la Trace Distance tra due matrici di densità.
    L'operazione è invariante per base, quindi possiamo usare la base eccitonica.
    """
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)
    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)

def clt_fit(N, a):
    """Funzione di fit per il Teorema del Limite Centrale (Decadimento a -0.5)."""
    return a / np.sqrt(N)

# ==========================================
# IMPOSTAZIONI INIZIALI
# ==========================================

# Parametri della simulazione da analizzare
dt = 1.0
N_traj_max = 10000
dt_str = f"{dt:.2f}".replace(".", "p")

# Gli angoli che vogliamo confrontare
angles_deg = [0.0, 90.0]
labels = {0.0: "Quantum Jump ($\\Theta = 0^\\circ$)", 90.0: "State Diffusion ($\\Theta = 90^\\circ$)"}
colors = {0.0: "red", 90.0: "blue"}

# La lista dei numeri di traiettorie per l'analisi di convergenza
N_list = [100, 200, 500, 1000, 2000, 4000, 8000, 10000]

# Percorsi delle directory
results_dir = "../Results/Data/"
output_dir = "../Results/Plot/Comparison_Metrics/"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# ESTRAZIONE DATI ED ELABORAZIONE
# ==========================================

# Dizionari per salvare i risultati per ogni angolo
td_time_dict = {angle: {} for angle in angles_deg}
mean_td_dict = {angle: [] for angle in angles_deg}
max_td_dict = {angle: [] for angle in angles_deg}

times = None
n_times = 0

print("--- Inizio analisi comparative ---")

for theta_deg in angles_deg:
    theta_str = f"{theta_deg:.3f}".replace(".", "p")
    fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj_max}.npz")
    
    print(f"\nCaricamento dati per Theta = {theta_deg}° da {fname}...")
    try:
        data = np.load(fname)
    except FileNotFoundError:
        print(f"Errore: File {fname} non trovato. Interruzione.")
        sys.exit(1)

    # Estraiamo i tempi solo la prima volta (sono uguali per tutti)
    if times is None:
        times = data['times']
        n_times = len(times)

    # Estraiamo le matrici nella base ECCITONICA (più veloce ed equivalente)
    rho_redfield_exc = data['rho_redfield_exc']
    psi_traj_exc = data['psi_traj']
    
    print(f"Calcolo della Trace Distance per varie grandezze dell'ensemble (N)...")
    for N_sub in N_list:
        # 1. Selezioniamo il sottoinsieme di traiettorie
        psi_sub = psi_traj_exc[:, :, :N_sub]
        
        # 2. Ricostruiamo la matrice di densità media su N_sub traiettorie
        # Usiamo einsum: somma su k (traiettorie) e divide per N_sub
        rho_sub_avg = np.einsum('itk, jtk -> tij', psi_sub, np.conjugate(psi_sub)) / N_sub
        
        # 3. Calcoliamo la Trace Distance nel tempo
        td_time_sub = np.zeros(n_times)
        for t in range(n_times):
            td_time_sub[t] = trace_distance_njit(rho_redfield_exc[t], rho_sub_avg[t])
        
        # Salviamo i risultati
        td_time_dict[theta_deg][N_sub] = td_time_sub
        mean_td_dict[theta_deg].append(np.mean(td_time_sub))
        max_td_dict[theta_deg].append(np.max(td_time_sub))
        
        print(f"  Completato N = {N_sub:5d} (Mean TD: {np.mean(td_time_sub):.4f})")

print("\nElaborazione dati completata. Inizio generazione plot...")

# ==========================================
# CONFIGURAZIONE PLOT GENERALE
# ==========================================
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--'
})

def save_fig(fig, filename):
    path_png = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Salvato: {path_png}")
    plt.close(fig)

# ==========================================
# PLOT 1: TRACE DISTANCE NEL TEMPO (UN GRAFICO PER OGNI N)
# ==========================================
for N_sub in N_list:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for theta_deg in angles_deg:
        ax.plot(times, td_time_dict[theta_deg][N_sub], 
                color=colors[theta_deg], linewidth=2.0, alpha=0.8,
                label=labels[theta_deg])
        
    ax.set_title(f'Trace Distance Over Time ($N = {N_sub}$ Trajectories)')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Trace Distance (vs Redfield Exact)')
    ax.set_yscale('log')
    
    # Fissiamo un range Y comune se possibile, per rendere le immagini confrontabili
    # ax.set_ylim(1e-3, 1.0) # Scommentare per forzare limiti identici su tutti i plot
    
    ax.legend(loc='best')
    save_fig(fig, f'Compare_TD_Time_N{N_sub}')

# ==========================================
# PLOT 2: CONVERGENZA AL CLT - VALORE MEDIO
# ==========================================
fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
N_smooth = np.logspace(np.log10(N_list[0]), np.log10(N_list[-1]), 200)

for theta_deg in angles_deg:
    y_data = np.array(mean_td_dict[theta_deg])
    
    # Fit della curva
    popt, _ = curve_fit(clt_fit, N_list, y_data)
    a_fit = popt[0]
    
    # Scatter plot dei dati reali
    ax_mean.loglog(N_list, y_data, 'o', color=colors[theta_deg], markersize=8, 
                   label=f'{labels[theta_deg]} Data')
    # Plot della curva di fit
    ax_mean.loglog(N_smooth, clt_fit(N_smooth, a_fit), '--', color=colors[theta_deg], 
                   linewidth=2, label=f'Fit $1/\\sqrt{{N}}$ (a={a_fit:.3f})')

ax_mean.set_title('Convergence of Mean Trace Distance')
ax_mean.set_xlabel('Number of Trajectories ($N$)')
ax_mean.set_ylabel('Time-Averaged Trace Distance')
ax_mean.legend()
save_fig(fig_mean, 'Compare_Convergence_MeanTD')

# ==========================================
# PLOT 3: CONVERGENZA AL CLT - VALORE MASSIMO
# ==========================================
fig_max, ax_max = plt.subplots(figsize=(8, 6))

for theta_deg in angles_deg:
    y_data = np.array(max_td_dict[theta_deg])
    
    # Fit della curva
    popt, _ = curve_fit(clt_fit, N_list, y_data)
    a_fit = popt[0]
    
    # Scatter plot dei dati reali
    ax_max.loglog(N_list, y_data, 's', color=colors[theta_deg], markersize=8, 
                  label=f'{labels[theta_deg]} Data')
    # Plot della curva di fit
    ax_max.loglog(N_smooth, clt_fit(N_smooth, a_fit), '--', color=colors[theta_deg], 
                  linewidth=2, label=f'Fit $1/\\sqrt{{N}}$ (a={a_fit:.3f})')

ax_max.set_title('Convergence of Maximum Trace Distance')
ax_max.set_xlabel('Number of Trajectories ($N$)')
ax_max.set_ylabel('Maximum Trace Distance')
ax_max.legend()
save_fig(fig_max, 'Compare_Convergence_MaxTD')

print("--- Script eseguito con successo ---")