#!/usr/bin/env python
# coding: utf-8

"""
Plot comparativi State Diffusion (theta=90, colonna sinistra) vs Quantum Jump
(theta=0, colonna destra), in griglia 3x2 (3 righe = popolazioni |0>,|1>,|2>;
2 colonne = SD, QJ).

Genera le versioni "affiancate" di:
  - Comparison_3pop   (Lindblad vs Anc_trace vs Avg_traj)
  - Single_Traj_vs_Lindblad
  - Many_Traj_vs_Average

Stile, colori e convenzioni sono identici a Plot_generic_hdf5.py:
  - theta=90 (SD) -> blu, theta=0 (QJ) -> rosso (da ANGLE_COLORS in plot_style.py)
  - Lindblad / Anc_trace restano nero/grigio (curve teoriche di riferimento)
  - fontsize, palette, salvataggio PDF -> plot_style.py (set_thesis_style, save_fig)

Assi condivisi: stessa riga -> stesso asse y; stessa colonna -> stesso asse x.
Asse x tagliato a [0, 80] su tutti i pannelli (xlim).

Uso: python Plot_QJ_vs_SD_hdf5.py
(non prende argomenti da bash: gli angoli sono fissi, 90 a sinistra e 0 a destra)
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from plot_style import set_thesis_style, save_fig, get_angle_color

set_thesis_style()

# ==========================
# Parametri fissi di questo confronto
# ==========================
THETA_LEFT = 90.0   # State Diffusion -> colonna sinistra
THETA_RIGHT = 0.0   # Quantum Jump    -> colonna destra

COLOR_LEFT = get_angle_color(THETA_LEFT)   # blu
COLOR_RIGHT = get_angle_color(THETA_RIGHT)  # rosso

# Taglio comune dell'asse x per tutti i pannelli di tutti e tre i plot
XLIM = (0, 80)

# --- Time Step and Trajectory String Formatting (deve combaciare con quanto
# usato in CM_generic_rho_only.py e in Plot_generic_hdf5.py) ---
dt = 0.0012
N_traj = 10000
SAVE_STRIDE = 50

dt_str = f"{dt:.6f}".replace(".", "p")

# --- Results Directory and Output Setup ---
results_dir = "../../Results/Data/Complete_rho/"
Output_dir = "../../Results/Plot/Populations/small_dt/QJ_vs_SD"
os.makedirs(Output_dir, exist_ok=True)


def _make_fname(theta_deg):
    phi_str = f"{theta_deg:.4f}".replace(".", "p")
    return os.path.join(
        results_dir,
        f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}_stride{SAVE_STRIDE}.h5"
    ), phi_str


def load_angle_data(theta_deg):
    """
    Carica tutti i dataset necessari per un dato angolo dal file HDF5
    corrispondente. Stessa logica di caricamento di Plot_generic_hdf5.py,
    incapsulata qui per poter caricare comodamente sia SD che QJ.

    Returns: dict con tutti gli array necessari ai tre plot comparativi.
    """
    fname, phi_str = _make_fname(theta_deg)

    try:
        f_h5 = h5py.File(fname, "r")
        print(f"Data extraction completed successfully for Theta = {theta_deg}°")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {fname} not found. Ensure the simulation for theta={theta_deg}° has completed."
        )

    times = f_h5['times'][:]
    jump_records = f_h5['jump_records'][:]  # (n_times, N_traj), piena risoluzione

    rho_all = f_h5['rho_traj'][:]  # (3, 3, n_saved, N_traj)
    pop_00 = np.real(rho_all[0, 0, :, :])
    pop_11 = np.real(rho_all[1, 1, :, :])
    pop_22 = np.real(rho_all[2, 2, :, :])

    avg_pop_00 = pop_00.mean(axis=1)
    avg_pop_11 = pop_11.mean(axis=1)
    avg_pop_22 = pop_22.mean(axis=1)

    rho_trace = f_h5['rho_trace'][:]
    pops_trace_00 = np.real(rho_trace[0, 0, :])
    pops_trace_11 = np.real(rho_trace[1, 1, :])
    pops_trace_22 = np.real(rho_trace[2, 2, :])

    rho_lind = f_h5['rho_list_lindblad'][:]
    lindblad_00 = np.real(rho_lind[:, 0, 0])
    lindblad_11 = np.real(rho_lind[:, 1, 1])
    lindblad_22 = np.real(rho_lind[:, 2, 2])

    f_h5.close()

    # Criterio esatto per i salti: jump_records a piena risoluzione temporale
    # (vedi Plot_generic_hdf5.py per la motivazione completa).
    jump_mask = jump_records.sum(axis=0) > 0
    jump_indices = np.where(jump_mask)[0]

    return {
        'theta_deg': theta_deg,
        'times': times,
        'jump_indices': jump_indices,
        'pop_00': pop_00, 'pop_11': pop_11, 'pop_22': pop_22,
        'avg_pop_00': avg_pop_00, 'avg_pop_11': avg_pop_11, 'avg_pop_22': avg_pop_22,
        'pops_trace_00': pops_trace_00, 'pops_trace_11': pops_trace_11, 'pops_trace_22': pops_trace_22,
        'lindblad_00': lindblad_00, 'lindblad_11': lindblad_11, 'lindblad_22': lindblad_22,
    }


# ==========================
# Caricamento dati per entrambi gli angoli
# ==========================
data_left = load_angle_data(THETA_LEFT)    # State Diffusion (theta=90)
data_right = load_angle_data(THETA_RIGHT)  # Quantum Jump (theta=0)

print(f"[SD, theta={THETA_LEFT}°] Trajectories with jumps: {len(data_left['jump_indices'])}")
print(f"[QJ, theta={THETA_RIGHT}°] Trajectories with jumps: {len(data_right['jump_indices'])}")

# --------------------------------------------------------------
# sample_idx: scelto UNA SOLA VOLTA sul dataset Quantum Jump (theta=0),
# come prima traiettoria che ha subito un salto in quel dataset. Lo stesso
# indice numerico viene poi riusato per indicizzare l'array State Diffusion
# (theta=90), anche se quella specifica traiettoria SD potrebbe non aver
# avuto un salto in quell'indice: e' una scelta di allineamento voluta
# (stessa "colonna" del Monte Carlo, seed corrispondente), non una ricerca
# indipendente del salto in SD.
# --------------------------------------------------------------
qj_jump_indices = data_right['jump_indices']
if len(qj_jump_indices) > 0:
    sample_idx = qj_jump_indices[0]
else:
    sample_idx = 0  # fallback se nessuna traiettoria QJ ha avuto salti
print(f"Selected sample_idx (from QJ, theta=0): {sample_idx}")


# ================================================================
# Helper per la griglia 3x2 con assi condivisi (riga -> stesso y,
# colonna -> stesso x) e xlim comune.
# ================================================================
def make_grid_3x2():
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex='col', sharey='row')
    for row in range(3):
        for col in range(2):
            axes[row, col].set_xlim(*XLIM)
    return fig, axes


population_labels = [r'|0\rangle', r'|1\rangle', r'|2\rangle']


# ====================================================================
# Plot A: Comparison_3pop  (Lindblad vs Anc_trace vs Avg_traj), SD | QJ
# ====================================================================
fig_a, axes_a = make_grid_3x2()

datasets_a = [data_left, data_right]
colors_a = [COLOR_LEFT, COLOR_RIGHT]

pop_keys = ['00', '11', '22']

for col, (data, color) in enumerate(zip(datasets_a, colors_a)):
    times = data['times']
    for row, key in enumerate(pop_keys):
        ax = axes_a[row, col]
        lindblad = data[f'lindblad_{key}']
        trace = data[f'pops_trace_{key}']
        avg = data[f'avg_pop_{key}']

        ax.plot(times, lindblad, label='Lindblad', linewidth=2, linestyle='--', color='black')
        ax.plot(times, trace, label='AS_trace', linewidth=2, linestyle=':', color='gray')
        ax.plot(times, avg, label='Avg_traj', linewidth=2, alpha=0.8, color=color)

        formatter = ticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(title=fr"$\theta = {data['theta_deg']:.0f}^\circ$", loc='best')

# Label y solo sulla colonna sinistra (asse condiviso per riga)
for row in range(3):
    axes_a[row, 0].set_ylabel(fr'Population ${population_labels[row]}$')

# Label x solo sull'ultima riga (asse condiviso per colonna)
for col in range(2):
    axes_a[-1, col].set_xlabel('Time (fs)')

# Intestazioni di colonna per chiarire SD (sinistra) vs QJ (destra)
axes_a[0, 0].set_title('State Diffusion', fontsize=20)
axes_a[0, 1].set_title('Quantum Jump', fontsize=20)

save_fig(fig_a, 'Comparison_3pop_SD_vs_QJ', Output_dir)


# # ====================================================================
# # Plot B: Single_Traj_vs_Lindblad, SD | QJ (stesso sample_idx da QJ)
# # ====================================================================
# fig_b, axes_b = make_grid_3x2()

# for col, (data, color) in enumerate(zip(datasets_a, colors_a)):
#     times = data['times']
#     for row, key in enumerate(pop_keys):
#         ax = axes_b[row, col]
#         single = data[f'pop_{key}'][:, sample_idx]
#         lindblad = data[f'lindblad_{key}']

#         ax.plot(times, single, label='Single Traj', linewidth=2, alpha=0.9, color=color)
#         ax.plot(times, lindblad, label='Lindblad', linewidth=2, linestyle=':', color='black')

#         formatter = ticker.ScalarFormatter(useOffset=False)
#         formatter.set_scientific(False)
#         ax.yaxis.set_major_formatter(formatter)
#         ax.legend(title=fr"$\theta = {data['theta_deg']:.0f}^\circ$", loc='best')

# for row in range(3):
#     axes_b[row, 0].set_ylabel(fr'Population ${population_labels[row]}$')

# for col in range(2):
#     axes_b[-1, col].set_xlabel('Time (fs)')

# # axes_b[0, 0].set_title('State Diffusion', fontsize=18)
# # axes_b[0, 1].set_title('Quantum Jump', fontsize=18)

# save_fig(fig_b, 'Single_Traj_vs_Lindblad_SD_vs_QJ', Output_dir)


# # ====================================================================
# # Plot C: Many_Traj_vs_Average, SD | QJ
# # ====================================================================
# num_samples = 50

# fig_c, axes_c = make_grid_3x2()

# for col, (data, color) in enumerate(zip(datasets_a, colors_a)):
#     times = data['times']
#     jump_indices = data['jump_indices']
#     for row, key in enumerate(pop_keys):
#         ax = axes_c[row, col]
#         samples = data[f'pop_{key}'][:, :num_samples]
#         lindblad = data[f'lindblad_{key}']
#         avg = data[f'avg_pop_{key}']
#         jump_traj = data[f'pop_{key}'][:, sample_idx]

#         for i in range(num_samples):
#             ax.plot(times, samples[:, i], color='gray', alpha=0.40, linewidth=0.5,
#                      label='Single Traj' if i == 0 else "")

#         if len(jump_indices) > 0:
#             ax.plot(times, jump_traj, color=color, alpha=0.70, linewidth=1, label="")

#         ax.plot(times, lindblad, label='Lindblad', linewidth=2, linestyle='--', color='black')
#         ax.plot(times, avg, label='Avg Traj', linewidth=2, color=color, alpha=0.9)

#         formatter = ticker.ScalarFormatter(useOffset=False)
#         formatter.set_scientific(False)
#         ax.yaxis.set_major_formatter(formatter)
#         ax.legend(title=fr"$\theta = {data['theta_deg']:.0f}^\circ$", loc='best')

# for row in range(3):
#     axes_c[row, 0].set_ylabel(fr'Population ${population_labels[row]}$')

# for col in range(2):
#     axes_c[-1, col].set_xlabel('Time (fs)')

# #  axes_c[0, 0].set_title('State Diffusion', fontsize=18)
# # axes_c[0, 1].set_title('Quantum Jump', fontsize=18)

# save_fig(fig_c, 'Many_Traj_vs_Average_SD_vs_QJ', Output_dir)

print("All comparison plots (SD vs QJ) generated and saved successfully.")