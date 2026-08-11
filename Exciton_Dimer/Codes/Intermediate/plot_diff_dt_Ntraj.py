#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Plot per un SINGOLO angolo theta:
#   - Plot A: popolazione del sito iniziale vs tempo,
#             confronto tra diversi N_traj (dt fisso)
#   - Plot B: popolazione del sito iniziale vs tempo,
#             confronto tra diversi dt (N_traj fisso)
#   - Plot C: convergenza Lindblad / traccia ancilla / media
#             traiettorie WF per una singola coppia (dt, N_traj)
#
# Legge i file .npz prodotti da run_single_theta_sweep.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_style import set_thesis_style, save_fig

set_thesis_style()

# ============================================================
#                     PARAMETRI DI CONTROLLO
# ============================================================
# Devono coincidere con quelli usati in run_single_theta_sweep.py

THETA_DEG = 0.0

# Sito la cui popolazione viene plottata: 0 -> |10>, 1 -> |01>
# (Nel codice originale psi_sys_initial ha eccitazione sul sito 2,
#  quindi il sito "di partenza" e' l'indice 1 -> |01>. Cambia se serve.)
site_index = 1

DT_FIXED = 0.01                             # <-- dt fissato per lo sweep su N_traj
N_TRAJ_SWEEP = [100, 1000, 10000]           # <-- CAMBIA QUI i valori di N_traj da confrontare

# ---- Sweep 2: dt variabile, N_traj fisso ----
N_TRAJ_FIXED = 1000                        # <-- N_traj fissato per lo sweep su dt
DT_SWEEP = [0.01, 0.02, 0.05, 0.1]

# ---- Plot C: singola coppia (dt, N_traj) per convergenza ----
DT_CONV = 0.01                              # <-- CAMBIA QUI se serve
N_TRAJ_CONV = 10000                         # <-- CAMBIA QUI se serve

# ---- Plot D: singole traiettorie vs sistema isolato ----
DT_SINGLE = 0.01                            # <-- CAMBIA QUI se serve
N_TRAJ_SINGLE = 1000                       # <-- deve essere un file gia' calcolato
N_TRAJ_TO_PLOT = 10                         # <-- quante traiettorie individuali mostrare (<= N_SINGLE_TRAJ_SAVE)

MODE_LABEL = "single_theta_sweep"
Input_dir = f"../../Results/Data/Complete_rho/{MODE_LABEL}/"

Output_dir = os.path.join("../Results/Plot/Populations/Comparison", MODE_LABEL, str(THETA_DEG))
os.makedirs(Output_dir, exist_ok=True)

theta_rad = np.radians(THETA_DEG)
theta_str = f"{theta_rad:.6f}".replace(".", "p")

label_site = r"$|10\rangle$" if site_index == 0 else r"$|01\rangle$"


# ============================================================
#                     FUNZIONI DI SUPPORTO
# ============================================================
def _make_fname_npz(Input_dir, theta_str, dt, N_traj):
    dt_str = f"{dt:.6f}".replace(".", "p")
    return os.path.join(Input_dir, f"result_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")


def load_population(theta_str, dt, N_traj, site_index):
    """
    Carica il file .npz e ritorna (times, avg_pop) per il sito scelto,
    mediando su TUTTE le traiettorie presenti nel file (N_traj colonne).
    """
    filepath = _make_fname_npz(Input_dir, theta_str, dt, N_traj)

    if not os.path.exists(filepath):
        print(f"ATTENZIONE: file non trovato -> {filepath}")
        return None, None

    data = np.load(filepath)
    times = data['times']

    raw_pop = data['pop_00'] if site_index == 0 else data['pop_11']

    # Media su tutte le traiettorie disponibili nel file
    n_avail = raw_pop.shape[1]
    avg_pop = np.mean(raw_pop[:, :n_avail], axis=1)

    return times, avg_pop


def load_lindblad(theta_str, dt, N_traj, site_index):
    """
    Carica la baseline analitica di Lindblad da un file .npz.
    Indice diagonale 2 -> |10>, indice diagonale 1 -> |01>
    (stessa convenzione usata nello script di riferimento).
    """
    filepath = _make_fname_npz(Input_dir, theta_str, dt, N_traj)

    if not os.path.exists(filepath):
        print(f"ATTENZIONE: file non trovato -> {filepath}")
        return None, None

    data = np.load(filepath)
    times = data['times']
    rho_lindblad = data['rho_list_lindblad']

    diag_idx = 2 if site_index == 0 else 1
    pop_lindblad = np.real(rho_lindblad[:, diag_idx, diag_idx])

    return times, pop_lindblad


def load_trace_ancilla(theta_str, dt, N_traj, site_index):
    """
    Carica la baseline della traccia sull'ancilla ('pops_trace') da un file .npz.
    """
    filepath = _make_fname_npz(Input_dir, theta_str, dt, N_traj)

    if not os.path.exists(filepath):
        print(f"ATTENZIONE: file non trovato -> {filepath}")
        return None, None

    data = np.load(filepath)
    times = data['times']
    pops_trace = data['pops_trace']

    pop_trace = pops_trace[site_index, :]

    return times, pop_trace


def load_single_trajs_and_isolated(theta_str, dt, N_traj, site_index):
    """
    Carica un sottoinsieme di singole traiettorie quantum-jump (colonne)
    insieme alla curva del sistema isolato, dallo stesso file .npz.
    """
    filepath = _make_fname_npz(Input_dir, theta_str, dt, N_traj)

    if not os.path.exists(filepath):
        print(f"ATTENZIONE: file non trovato -> {filepath}")
        return None, None, None

    data = np.load(filepath)
    times = data['times']

    single_trajs = data['single_trajs_10'] if site_index == 0 else data['single_trajs_01']
    pop_isolated = data['pop_traj_isolated'][site_index, :]

    return times, single_trajs, pop_isolated


# ============================================================
#         PLOT A: confronto al variare di N_traj (dt fisso)
# ============================================================
plt.close('all')
figA, axA = plt.subplots()

for N_traj in N_TRAJ_SWEEP:
    times, avg_pop = load_population(theta_str, DT_FIXED, N_traj, site_index)
    if times is None:
        continue
    axA.plot(times, avg_pop, label=fr'$N_{{traj}}={N_traj}$')

# Baseline Lindblad (non dipende da N_traj, presa dal primo file disponibile)
times_lind, pop_lind = load_lindblad(theta_str, DT_FIXED, N_TRAJ_SWEEP[0], site_index)
if times_lind is not None:
    axA.plot(times_lind, pop_lind, label='Lindblad', color='black', linestyle='--')

axA.set_xlim(60, 80)  # Zoom su un intervallo di tempo specifico
axA.set_ylim(0.4, 0.6)    # Limiti dell'asse y per chiarezza
axA.set_xlabel(r'Time [1/V]')
axA.set_ylabel(r'Population $|1\rangle$')
axA.legend(title=fr"$dt = {DT_FIXED}$", loc='upper right')

# save_fig(figA, f"Ntraj_sweep_Theta_{theta_str}_dt{str(DT_FIXED).replace('.', 'p')}", Output_dir)
save_fig(figA, f"ZOOM_Ntraj_sweep_Theta_{theta_str}_dt{str(DT_FIXED).replace('.', 'p')}", Output_dir)


# # ============================================================
# #         PLOT B: confronto al variare di dt (N_traj fisso)
# # ============================================================
# plt.close('all')
# figB, axB = plt.subplots()

# for dt in DT_SWEEP:
#     times, avg_pop = load_population(theta_str, dt, N_TRAJ_FIXED, site_index)
#     if times is None:
#         continue
#     axB.plot(times, avg_pop, label=fr'$dt={dt}$', linewidth=1.2)

# # Baseline Lindblad (non dipende da dt, presa dal primo file disponibile)
# times_lind, pop_lind = load_lindblad(theta_str, DT_SWEEP[0], N_TRAJ_FIXED, site_index)
# if times_lind is not None:
#     axB.plot(times_lind, pop_lind, label='Lindblad', color='black', linestyle='--', linewidth=1.2)

# # axB.set_xlim(55, 65)  # Zoom su un intervallo di tempo specifico
# # axB.set_ylim(0.4, 0.6)    # Limiti dell'asse y per chiarezza
# axB.set_xlabel(r'Time [1/V]')
# axB.set_ylabel(r'Population $|1\rangle$')
# axB.legend(title=fr"$N_{{traj}} = {N_TRAJ_FIXED}$", title_fontsize=11)

# save_fig(figB, f"dt_sweep_Theta_{theta_str}_Ntraj{N_TRAJ_FIXED}", Output_dir)
# # save_fig(figB, f"ZOOM_dt_sweep_Theta_{theta_str}_Ntraj{N_TRAJ_FIXED}", Output_dir)


# # ============================================================
# #         PLOT C: convergenza Lindblad / traccia / media WF
# #                 per una singola coppia (dt, N_traj)
# # ============================================================
# plt.close('all')
# figC, axC = plt.subplots()

# times_conv, avg_pop_conv = load_population(theta_str, DT_CONV, N_TRAJ_CONV, site_index)
# times_lind_c, pop_lind_c = load_lindblad(theta_str, DT_CONV, N_TRAJ_CONV, site_index)
# times_trace_c, pop_trace_c = load_trace_ancilla(theta_str, DT_CONV, N_TRAJ_CONV, site_index)

# if pop_lind_c is not None:
#     axC.plot(times_lind_c, pop_lind_c, label='Lindblad', color='black', linestyle='--', linewidth=2.0)

# if pop_trace_c is not None:
#     axC.plot(times_trace_c, pop_trace_c, label='Ancilla trace', color='green', linewidth=1.2)

# if avg_pop_conv is not None:
#     axC.plot(times_conv, avg_pop_conv, label='Avg trajectories', color='#0072B2', alpha=0.7, linewidth=2.0)

# axC.set_xlim(55, 65)  # Zoom su un intervallo di tempo specifico
# axC.set_ylim(0.45, 0.55)    # Limiti dell'asse y per chiarezza
# axC.set_xlabel(r'Time [1/V]')
# axC.set_ylabel(r'Population $|1\rangle$')
# axC.legend(title=fr"$dt = {DT_CONV}$, $N_{{traj}} = {N_TRAJ_CONV}$", title_fontsize=16)

# # save_fig(figC, f"Convergence_Theta_{theta_str}_dt{str(DT_CONV).replace('.', 'p')}_Ntraj{N_TRAJ_CONV}", Output_dir)
# save_fig(figC, f"ZOOM_Convergence_Theta_{theta_str}_dt{str(DT_CONV).replace('.', 'p')}_Ntraj{N_TRAJ_CONV}", Output_dir)


# # ============================================================
# #     PLOT D: singole traiettorie quantum-jump vs sistema isolato
# # ============================================================
# plt.close('all')
# figD, axD = plt.subplots()

# times_d, single_trajs_d, pop_isolated_d = load_single_trajs_and_isolated(
#     theta_str, DT_SINGLE, N_TRAJ_SINGLE, site_index)

# if single_trajs_d is not None:
#     n_to_plot = min(N_TRAJ_TO_PLOT, single_trajs_d.shape[1])

#     for k in range(n_to_plot):
#         label = 'Single trajectories' if k == 0 else None
#         axD.plot(times_d, single_trajs_d[:, k], color='gray', alpha=0.35, linewidth=0.8, label=label)

#     axD.plot(times_d, pop_isolated_d, label='Isolated system', color='black', linewidth=1.4)

# axD.set_xlabel(r'Time [1/V]')
# axD.set_ylabel(r'Population $|1\rangle$')
# axD.legend(title=fr"$dt = {DT_SINGLE}$", title_fontsize=11)

# save_fig(figD, f"SingleTraj_vs_Isolated_Theta_{theta_str}_dt{str(DT_SINGLE).replace('.', 'p')}", Output_dir)

# print("Plot completati e salvati in:", Output_dir)