#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig, get_angle_color, NO_JUMP_COLOR

# Apply global thesis style settings
set_thesis_style()

# ==========================
# Input Parsing from Bash
# ==========================
# Read the angle passed from the bash script (e.g., python plot_script.py 180)
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 180.0  # Default fallback if run manually without arguments

theta_rad = np.radians(theta_deg)

# Colore fisso associato a questo angolo (vedi mappa ANGLE_COLORS in
# plot_style.py): tinge le curve legate ai DATI stocastici (Avg_traj,
# Single_Traj, No-Jump). Le curve teoriche di riferimento (Lindblad,
# Anc_trace) restano invece in nero/grigio in ogni plot, per essere
# riconoscibili indipendentemente dall'angolo.
theta_color = get_angle_color(theta_deg)

# ==========================
# Setup and Data Loading
# ==========================

# --- Time Step and Trajectory String Formatting ---
# dt = 0.01
dt = 0.0012
N_traj = 10000
# SAVE_STRIDE = 10
SAVE_STRIDE = 50  # deve corrispondere a SAVE_STRIDE usato in CM_generic_rho_only.py

dt_str = f"{dt:.6f}".replace(".", "p")

# NOTA: lo script di calcolo (CM_generic_rho_only.py) ora salva il file usando
# phi_deg (gradi), non phi_rad come nella versione precedente. Il nome file e'
# quindi costruito con theta_deg, non theta_rad.
phi_str = f"{theta_deg:.4f}".replace(".", "p")

# --- Results Directory and Output Setup ---
results_dir = "../../Results/Data/Complete_rho/"

# Create a specific subfolder for the current angle
Output_dir = f"../../Results/Plot/Populations/small_dt/{phi_str}"
os.makedirs(Output_dir, exist_ok=True)

# Nome file aggiornato: estensione .h5 e suffisso _stride{SAVE_STRIDE}
fname = os.path.join(
    results_dir,
    f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}_stride{SAVE_STRIDE}.h5"
)

try:
    f_h5 = h5py.File(fname, "r")
    print(f"Data extraction completed successfully for Theta = {theta_deg}°")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# --------------------------------------------------------------
# 'times' e' l'asse temporale sottocampionato (n_saved punti): tutti i
# dataset "corti" nel file (rho_traj, rho_trace, rho_list_lindblad,
# rho_traj_isolated, total_jumps) sono allineati su questo stesso asse.
#
# 'jump_records' resta invece a piena risoluzione temporale (n_times punti,
# uno per ogni traiettoria) ed e' l'unico modo affidabile per sapere se una
# data traiettoria ha subito almeno un salto quantistico: usarlo al posto di
# una soglia euristica sulla popolazione sottocampionata evita falsi
# negativi (salti avvenuti tra due punti salvati e quindi invisibili nella
# derivata di pop_11 sottocampionata).
# --------------------------------------------------------------
times = f_h5['times'][:]
total_jumps = f_h5['total_jumps'][:]
jump_records = f_h5['jump_records'][:]  # shape (n_times, N_traj), a piena risoluzione

# -----------------------------------------------
# Extract from rho_traj (3, 3, n_saved, N_traj)
# Nota: dtype complex64 nel file, upcast a complex128 non necessario per il plotting.
# -----------------------------------------------
rho_all = f_h5['rho_traj'][:]

# Populations
pop_00 = np.real(rho_all[0, 0, :, :])
pop_11 = np.real(rho_all[1, 1, :, :])
pop_22 = np.real(rho_all[2, 2, :, :])

# Coherences
coh_01 = rho_all[0, 1, :, :]
coh_12 = rho_all[1, 2, :, :]
coh_02 = rho_all[0, 2, :, :]

# Averages over all trajectories
avg_pop_00 = pop_00.mean(axis=1)
avg_pop_11 = pop_11.mean(axis=1)
avg_pop_22 = pop_22.mean(axis=1)
avg_coh_01 = coh_01.mean(axis=1)
avg_coh_12 = coh_12.mean(axis=1)
avg_coh_02 = coh_02.mean(axis=1)

# -----------------------------------------------
# Extract baseline: rho_trace (3, 3, n_saved)
# -----------------------------------------------
rho_trace = f_h5['rho_trace'][:]
pops_trace_00 = np.real(rho_trace[0, 0, :])
pops_trace_11 = np.real(rho_trace[1, 1, :])
pops_trace_22 = np.real(rho_trace[2, 2, :])

# ----------------------------------------------------
# Extract Lindblad: rho_list_lindblad (n_saved, 3, 3)
# ----------------------------------------------------
rho_lind = f_h5['rho_list_lindblad'][:]
lindblad_00 = np.real(rho_lind[:, 0, 0])
lindblad_11 = np.real(rho_lind[:, 1, 1])
lindblad_22 = np.real(rho_lind[:, 2, 2])
lindblad_12 = rho_lind[:, 1, 2]
lindblad_01 = rho_lind[:, 0, 1]
lindblad_02 = rho_lind[:, 0, 2]

# -----------------------------------------------
# Extract isolated system: rho_traj_isolated (3, 3, n_saved)
# -----------------------------------------------
rho_iso = f_h5['rho_traj_isolated'][:]
pop_traj_isolated_00 = np.real(rho_iso[0, 0, :])
pop_traj_isolated_11 = np.real(rho_iso[1, 1, :])
pop_traj_isolated_22 = np.real(rho_iso[2, 2, :])

# File HDF5 non piu' necessario oltre questo punto: tutti i dataset sono stati
# gia' caricati in memoria come array NumPy.
f_h5.close()


# =========================================================
# Find trajectories that experienced a Quantum Jump
# =========================================================
# Criterio corretto: usa jump_records (a piena risoluzione, un flag preciso
# per ogni singolo step di ogni traiettoria) invece di una soglia euristica
# sulla derivata di pop_11 sottocampionata. Quest'ultima puo' mancare salti
# avvenuti tra due punti salvati (falsi negativi) o essere innescata da
# fluttuazioni che non sono veri salti (falsi positivi).
jump_mask = jump_records.sum(axis=0) > 0   # True se la traiettoria ha subito >=1 salto
jump_indices = np.where(jump_mask)[0]

print(f"Total trajectories evaluated: {pop_11.shape[1]}")
print(f"Found {len(jump_indices)} trajectories with jumps.")

if len(jump_indices) > 0:
    sample_idx = jump_indices[0]
else:
    sample_idx = 0  # Fallback
print(f"Selected sample_idx for plotting: {sample_idx}")


# ==========================================
# Plot 0: Plotting the Total Jump Counts
# ==========================================
fig_jumps, ax_jumps = plt.subplots(figsize=(10, 5))
ax_jumps.plot(times, total_jumps, color=theta_color, alpha=0.9, linewidth=2.0,
              label=f'Jumps per step (Total: {np.sum(total_jumps)})')

# ax_jumps.set_title(f"Total Jumps Over Time (Theta={theta_deg}°, dt={dt})", fontsize=14)
ax_jumps.set_xlabel("Time (fs)")
ax_jumps.set_ylabel("Number of Jumps")
ax_jumps.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)

save_fig(fig_jumps, f'Total_Jumps_Theta_{phi_str}', Output_dir)


# ====================================
# Plot 1: Convergence Avg vs Trace vs Lindblad
# ====================================
populations = [
    {'lindblad': lindblad_00, 'trace': pops_trace_00, 'avg': avg_pop_00, 'label': r'|0\rangle'},
    {'lindblad': lindblad_11, 'trace': pops_trace_11, 'avg': avg_pop_11, 'label': r'|1\rangle'},
    {'lindblad': lindblad_22, 'trace': pops_trace_22, 'avg': avg_pop_22, 'label': r'|2\rangle'},
]

# 3 subplot in verticale (una sotto l'altra) invece che affiancati
fig01, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=False)

for ax, pop in zip(axes, populations):
    lbl = pop['label']
    # Lindblad e trace restano curve teoriche di riferimento -> nero/grigio
    ax.plot(times, pop['lindblad'], label=r'Lindblad', linewidth=2, linestyle='--', color='black')
    ax.plot(times, pop['trace'],    label=r'AS_trace', linewidth=2, linestyle=':', color='gray')
    # avg_traj e' un dato stocastico legato a questo angolo -> colore fisso
    ax.plot(times, pop['avg'], label=r'Avg_traj',  linewidth=2, alpha=0.8, color=theta_color)

    ax.set_xlim(0, 80)

    # ax.set_title(f'Population {lbl}', fontsize=14)
    ax.set_ylabel(fr'Population ${lbl}$')

    formatter = ticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

# Solo l'ultimo subplot (in basso) mostra la label dell'asse x
axes[-1].set_xlabel('Time (fs)')

# fig01.suptitle(f'Angle {theta_deg}° — Lindblad vs Trace vs Avg Traj | dt={dt}, N_traj={N_traj}', fontsize=15)
save_fig(fig01, f'Comparison_3pop_Theta_{phi_str}', Output_dir) 


# ================================================
# Plot 2: Comparison trajectories Collisional vs Lindblad
# ================================================
plot_data_single = [
    {'single': pop_00[:, sample_idx], 'lindblad': lindblad_00, 'label': r'|0\rangle'},
    {'single': pop_11[:, sample_idx], 'lindblad': lindblad_11, 'label': r'|1\rangle'},
    {'single': pop_22[:, sample_idx], 'lindblad': lindblad_22, 'label': r'|2\rangle'}
]

# 3 subplot in verticale (una sotto l'altra) invece che affiancati
fig02, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=False)

for ax, data_s in zip(axes, plot_data_single):
    lbl = data_s['label']
    # Single traj e' un dato stocastico legato a questo angolo -> colore fisso
    ax.plot(times, data_s['single'], label=r'Single Traj', linewidth=2, alpha=0.9, color=theta_color)
    ax.plot(times, data_s['lindblad'], label=r'Lindblad', linewidth=2, linestyle=':', color='black')

    # ax.set_title(f'Population {lbl}', fontsize=14)
    ax.set_ylabel(fr'Population ${lbl}$')
    
    ax.set_xlim(0, 80)

    formatter = ticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False) 
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

# Solo l'ultimo subplot (in basso) mostra la label dell'asse x
axes[-1].set_xlabel('Time (fs)')

# fig02.suptitle(f'Angle {theta_deg}° - Single Trajectory vs Lindblad (Sample: {sample_idx})', fontsize=15)
save_fig(fig02, f'Single_Traj_vs_Lindblad_Theta_{phi_str}', Output_dir)


# ======================================================
# Plot 3: Many Single Trajectories vs Average vs Lindblad
# ======================================================
num_samples = 50 

plot_data_many = [
    {'samples': pop_00[:, :num_samples], 'lindblad': lindblad_00, 'avg': avg_pop_00, 'jump': pop_00[:, sample_idx], 'label': r'|0\rangle'},
    {'samples': pop_11[:, :num_samples], 'lindblad': lindblad_11, 'avg': avg_pop_11, 'jump': pop_11[:, sample_idx], 'label': r'|1\rangle'},
    {'samples': pop_22[:, :num_samples], 'lindblad': lindblad_22, 'avg': avg_pop_22, 'jump': pop_22[:, sample_idx], 'label': r'|2\rangle'}
]

fig03, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=False)

for ax, data_m in zip(axes, plot_data_many):
    lbl = data_m['label']

    for i in range(num_samples):
        ax.plot(times, data_m['samples'][:, i], color='gray', alpha=0.40, linewidth=0.5,
                label='Single Traj' if i == 0 else "")

    if len(jump_indices) > 0:
        ax.plot(times, data_m['jump'], color=theta_color, alpha=0.70, linewidth=1, label="")

    ax.plot(times, data_m['lindblad'], label='Lindblad', linewidth=2, linestyle='--', color='black')
    # avg_traj e' il dato stocastico principale legato a questo angolo -> colore fisso
    ax.plot(times, data_m['avg'], label='Avg Traj', linewidth=2, color=theta_color, alpha=0.9)

    ax.set_xlim(0, 80)

    # ax.set_title(f'Population {lbl}', fontsize=14)
    ax.set_ylabel(fr'Population ${lbl}$')

    formatter = ticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False) 
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

# Solo l'ultimo subplot (in basso) mostra la label dell'asse x
axes[-1].set_xlabel('Time (fs)')

# fig03.suptitle(f'Angle {theta_deg}° - Many Trajectories vs Average | dt={dt}', fontsize=15)
save_fig(fig03, f'Many_Traj_vs_Average_Theta_{phi_str}', Output_dir)


# =========================================================
# Plot 4: Plotting Real and Imaginary Coherences
# =========================================================
coherence_data = [
    ('01', lindblad_01, avg_coh_01),
    ('12', lindblad_12, avg_coh_12),
    ('02', lindblad_02, avg_coh_02)
]

fig04, axes = plt.subplots(3, 2, figsize=(16, 15))

for row_idx, (label, lind_data, avg_data) in enumerate(coherence_data):
    # Real Part
    ax_real = axes[row_idx, 0]
    ax_real.plot(times, np.real(lind_data), label=f'Lindblad', linewidth=2, linestyle='--', color='black')
    ax_real.plot(times, np.real(avg_data), label=f'Avg Traj', linewidth=2, color=theta_color, alpha=0.9)
    # ax_real.set_title(f'Real Part of Coherence $\\rho_{{{label}}}$', fontsize=14)
    ax_real.set_ylabel(fr'Re($\rho_{{{label}}}$)')

    # Imaginary Part
    ax_imag = axes[row_idx, 1]
    ax_imag.plot(times, np.imag(lind_data), label=f'Lindblad', linewidth=2, linestyle='--', color='black')
    ax_imag.plot(times, np.imag(avg_data), label=f'Avg Traj', linewidth=2, color=theta_color, alpha=0.9)
    # ax_imag.set_title(f'Imaginary Part of Coherence $\\rho_{{{label}}}$', fontsize=14)
    ax_imag.set_ylabel(fr'Im($\rho_{{{label}}}$)')

    ax.set_xlim(0, 80)

for ax in axes.flat:
    ax.set_xlabel('Time (fs)')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
    formatter = ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(formatter)

# fig04.suptitle(f'Angle {theta_deg}° - Lindblad vs Average Trajectory Coherences', fontsize=16, y=0.98)
save_fig(fig04, f'Coherences_Theta_{phi_str}', Output_dir)


# =========================================================
# Plot 5 & 6: No-Jump Post-Selected Subensemble
# =========================================================
# L'analisi No-Jump ha senso fisico solo nel regime "puro Quantum Jump"
# (theta = 0): e' li' che la distinzione tra traiettorie con e senza salti
# e' concettualmente significativa. Per gli altri angoli (State Diffusion e
# intermedi) il blocco viene saltato.
if theta_deg == 0:
    N_traj_total = pop_00.shape[1]
    all_indices = np.arange(N_traj_total)
    no_jump_indices = np.setdiff1d(all_indices, jump_indices)

    print(f"No-jump trajectories: {len(no_jump_indices)} / {N_traj_total}")

    if len(no_jump_indices) > 0:
        # --- Plot Single No-Jump Trajectory vs Lindblad ---
        # Traiettoria campione presa dal sottoinsieme SENZA salti (diverso da
        # 'sample_idx', che invece e' scelto tra le traiettorie CON salti per
        # il Plot 2 standard). Stesso layout a 3 subplot verticali degli altri
        # plot "single trajectory", colore NO_JUMP_COLOR per coerenza con gli
        # altri plot di questo blocco.
        sample_idx_nj = no_jump_indices[0]
        print(f"Selected sample_idx (no-jump) for plotting: {sample_idx_nj}")

        plot_data_single_nj = [
            {'single': pop_00[:, sample_idx_nj], 'lindblad': lindblad_00, 'label': r'|0\rangle'},
            {'single': pop_11[:, sample_idx_nj], 'lindblad': lindblad_11, 'label': r'|1\rangle'},
            {'single': pop_22[:, sample_idx_nj], 'lindblad': lindblad_22, 'label': r'|2\rangle'}
        ]

        fig_single_nj, axes_single_nj = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=False)

        for ax, data_s in zip(axes_single_nj, plot_data_single_nj):
            lbl = data_s['label']
            ax.plot(times, data_s['single'], label='Single Traj (No-Jump)', linewidth=2, alpha=0.9, color=NO_JUMP_COLOR)
            ax.plot(times, data_s['lindblad'], label='Lindblad', linewidth=2, linestyle=':', color='black')

            ax.set_ylabel(fr'Population ${lbl}$')

            formatter = ticker.ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
            ax.set_xlim(0, 80)
        axes_single_nj[-1].set_xlabel('Time (fs)')

        save_fig(fig_single_nj, f'NO_JUMPS_Single_Traj_vs_Lindblad_Theta_{phi_str}', Output_dir)

        # Populations (Averaging only over the 'no_jump_indices')
        avg_pop_00_nj = pop_00[:, no_jump_indices].mean(axis=1)
        avg_pop_11_nj = pop_11[:, no_jump_indices].mean(axis=1)
        avg_pop_22_nj = pop_22[:, no_jump_indices].mean(axis=1)

        # Coherences
        avg_coh_01_nj = coh_01[:, no_jump_indices].mean(axis=1)
        avg_coh_12_nj = coh_12[:, no_jump_indices].mean(axis=1)
        avg_coh_02_nj = coh_02[:, no_jump_indices].mean(axis=1)

        # --- Plot Populations (No-Jump) ---
        fig_pop, axes_pop = plt.subplots(1, 3, figsize=(18, 5))
        pop_data_nj = [
            {'lindblad': lindblad_00, 'full_avg': avg_pop_00, 'no_jump': avg_pop_00_nj, 'label': r'|0\rangle'},
            {'lindblad': lindblad_11, 'full_avg': avg_pop_11, 'no_jump': avg_pop_11_nj, 'label': r'|1\rangle'},
            {'lindblad': lindblad_22, 'full_avg': avg_pop_22, 'no_jump': avg_pop_22_nj, 'label': r'|2\rangle'}
        ]

        for ax, data_nj in zip(axes_pop, pop_data_nj):
            lbl = data_nj['label']
            ax.plot(times, data_nj['lindblad'], label='Lindblad', linewidth=2, linestyle='--', color='black')
            ax.plot(times, data_nj['full_avg'], label='Standard Avg', linewidth=2, color='gray', alpha=0.6)
            # Colore dedicato NO_JUMP_COLOR (viola), distinto sia dal rosso
            # standard di theta=0 sia dagli altri colori della mappa angoli.
            ax.plot(times, data_nj['no_jump'], label='No-Jump Evolution', linewidth=2.5, color=NO_JUMP_COLOR, alpha=0.95)

            # ax.set_title(f'Population {lbl}', fontsize=14)
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel(fr'Population ${lbl}$')
            ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
            ax.set_xlim(0, 80)
        # fig_pop.suptitle(f'Angle {theta_deg}° Populations: Post-Selected Subensemble (No Jumps)', fontsize=16)
        save_fig(fig_pop, f'NO_JUMPS_Populations_Theta_{phi_str}', Output_dir)

        # --- Plot Coherences (No-Jump) ---
        fig_coh, axes_coh = plt.subplots(3, 2, figsize=(16, 15))
        coh_data_nj = [
            ('01', lindblad_01, avg_coh_01, avg_coh_01_nj),
            ('12', lindblad_12, avg_coh_12, avg_coh_12_nj),
            ('02', lindblad_02, avg_coh_02, avg_coh_02_nj)
        ]

        for row_idx, (label, lind_data, full_avg, no_jump_avg) in enumerate(coh_data_nj):
            # Real Part
            ax_real = axes_coh[row_idx, 0]
            ax_real.plot(times, np.real(lind_data), label='Lindblad', linestyle='--', color='black')
            ax_real.plot(times, np.real(full_avg), label='Standard Avg', color='gray', alpha=0.6)
            ax_real.plot(times, np.real(no_jump_avg), label='No-Jump', linewidth=2.5, color=NO_JUMP_COLOR, alpha=0.95)
            # ax_real.set_title(f'Real Part $\\rho_{{{label}}}$')
            ax_real.set_ylabel(fr'Re($\rho_{{{label}}}$)')

            # Imaginary Part
            ax_imag = axes_coh[row_idx, 1]
            ax_imag.plot(times, np.imag(lind_data), label='Lindblad', linestyle='--', color='black')
            ax_imag.plot(times, np.imag(full_avg), label='Standard Avg', color='gray', alpha=0.6)
            ax_imag.plot(times, np.imag(no_jump_avg), label='No-Jump', linewidth=2.5, color=NO_JUMP_COLOR, alpha=0.95)
            # ax_imag.set_title(f'Imaginary Part $\\rho_{{{label}}}$')
            ax_imag.set_ylabel(fr'Im($\rho_{{{label}}}$)')
            ax_imag.set_xlim(0, 80)
        for ax in axes_coh.flat:
            ax.set_xlabel('Time (fs)')
            ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

        # fig_coh.suptitle(f'Angle {theta_deg}° Coherences: Post-Selected Subensemble (No Jumps)', fontsize=16, y=0.98)
        save_fig(fig_coh, f'NO_JUMPS_Coherences_Theta_{phi_str}', Output_dir)
    else:
        print("Nessuna traiettoria senza salti trovata: plot No-Jump saltati.")
else:
    print(f"Analisi No-Jump saltata: eseguita solo per theta = 0 (qui theta = {theta_deg}°).")

print("All plots generated and saved successfully.")