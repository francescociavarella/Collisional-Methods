#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import njit

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig, get_angle_color, get_site_colors, get_angle_gradient

# Apply global thesis style settings
set_thesis_style()

# =================================
# NUMBA OPTIMIZED METRIC FUNCTIONS
# =================================

@njit
def fidelity_generic_njit(rho, sigma):
    """Numba-compatible generalized fidelity."""
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho, 0.0)

    diag_matrix = np.diag(np.sqrt(evals_rho)).astype(np.complex128)

    sqrt_rho = evecs_rho @ diag_matrix @ evecs_rho.conj().T
    inner_matrix = sqrt_rho @ sigma.astype(np.complex128) @ sqrt_rho
    inner_matrix = 0.5 * (inner_matrix + inner_matrix.conj().T)

    evals_inner = np.linalg.eigvalsh(inner_matrix)
    evals_inner = np.maximum(evals_inner, 0.0)

    fidelity = np.sum(np.sqrt(evals_inner))**2
    return min(1.0, fidelity)

@njit
def trace_distance_generic_njit(rho, sigma):
    """Numba-compatible generalized trace distance."""
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)

    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)

# ==========================
# Input Parsing from Bash
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 0.0  # Default fallback if run manually

# Colore fisso associato a questo angolo (vedi mappa ANGLE_COLORS in
# plot_style.py): tinge le curve legate ai DATI stocastici (Avg trajectories,
# Single trajectory). Le curve teoriche di riferimento (Redfield, Ancilla
# trace, Isolated system) restano invece nei loro colori fissi attuali
# (nero/verde/rosso di riferimento), per essere riconoscibili
# indipendentemente dall'angolo. Il Plot 7 fa eccezione: li' ogni eccitone
# ha gia' un colore diverso (palette tab10) per distinguerli tra loro nello
# stesso grafico, quindi non usa get_angle_color.
theta_color = get_angle_color(theta_deg)

# --- Must match the values used in the main simulation script ---
dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/Plot/Eigenstate/{theta_str}"
os.makedirs(Output_dir, exist_ok=True)

fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Theta = {theta_deg} deg")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

times = data['times']
# Conversione a picosecondi: 'times' nel file e' in femtosecondi, si divide
# per 1000 per ottenere i ps. Usato in TUTTI i plot del file al posto di
# 'times' grezzo, con label 'Time (ps)' invece di 'Time (fs)'.
times_ps = times / 1000.0
dt_val = float(data['dt'])
N_site = int(data['N_site'])
eigenergies = data['eigenergies']
psi0_exc = data['psi0_exc']

total_jumps = data['total_jumps']
jump_counts = data['jump_counts']            # (n_times, n_traj)

# Extract matrices directly in the EXCITON basis
psi_traj_exc = data['psi_traj']                  # (N_site, n_times, n_traj)
rho_redfield_exc = data['rho_redfield_exc']       # (n_times, N_site, N_site)
rho_trace_coll_exc = data['rho_trace_coll_exc']  # (n_times, N_site, N_site)
rho_traj_avg_exc = data['rho_traj_avg_exc']        # (n_times, N_site, N_site)

n_times, n_traj = jump_counts.shape

# ==========================
# Exciton-basis single-trajectory populations
# ==========================
pop_traj_exc = np.abs(psi_traj_exc) ** 2                          # (N_site, n_times, n_traj)

# ==========================
# Isolated system (no collisions): in the eigenstate basis,
# populations are strictly constant over time!
# ==========================
pop_iso_exc_constant = np.abs(psi0_exc) ** 2                      # (N_site,)
pop_iso_exc = np.tile(pop_iso_exc_constant, (n_times, 1))         # (n_times, N_site)

# ==========================
# Redfield / collisional / MC-avg populations (exciton basis)
# ==========================
pop_redfield_exc = np.real(np.diagonal(rho_redfield_exc, axis1=1, axis2=2))
pop_trace_coll_exc = np.real(np.diagonal(rho_trace_coll_exc, axis1=1, axis2=2))
pop_traj_avg_exc = np.real(np.diagonal(rho_traj_avg_exc, axis1=1, axis2=2))

# ==========================
# Identify trajectories that experienced at least one jump
# ==========================
n_jumps_per_traj = jump_counts.sum(axis=0)   # (n_traj,)
jump_indices = np.where(n_jumps_per_traj > 0)[0]
print(f"Total trajectories: {n_traj}")
print(f"Trajectories with at least one jump: {len(jump_indices)}")
sample_idx = jump_indices[0] if len(jump_indices) > 0 else 0
print(f"Selected sample_idx for single-trajectory plots: {sample_idx}")

EXC_LABELS = [f"Exciton {i+1}" for i in range(N_site)]
# Etichette in notazione a ket per l'asse y (Population $|i\rangle$), stessa
# convenzione gia' usata nel resto del progetto per i plot a singolo
# autostato/sito per pannello. EXC_LABELS resta invariata per le legende
# (dove "Exciton i" e' piu' leggibile del solo ket).
EXC_KET_LABELS = [rf'|{i+1}\rangle' for i in range(N_site)]


def make_axes_grid_7panel(fig):
    """
    Costruisce una griglia di assi per N_site=7 pannelli: 4 righe totali,
    le prime 3 righe con 2 pannelli affiancati (6 siti), la quarta riga
    con un solo pannello centrato (7° sito).

    Layout: GridSpec 4 righe x 4 colonne.
      - Righe 0,1,2: pannello sinistro su colonne [0:2], destro su [2:4].
      - Riga 3: pannello unico centrato su colonne [1:3].

    Returns: np.array di 7 Axes, in ordine (sito 1..7).
    """
    gs = fig.add_gridspec(4, 4)
    axes = []
    for row in range(3):
        axes.append(fig.add_subplot(gs[row, 0:2]))
        axes.append(fig.add_subplot(gs[row, 2:4]))
    axes.append(fig.add_subplot(gs[3, 1:3]))
    return np.array(axes)


def add_shared_legend(fig, source_ax, theta_deg, top_margin=0.90, legend_y=None):
    """
    Aggiunge un'UNICA legenda condivisa per l'intera figura, orizzontale su
    una sola riga, in una fascia RISERVATA sopra i pannelli (invece di una
    legenda dentro ogni singolo pannello, che con piu' pannelli risulta
    ripetuta e ingombrante).

    fig.tight_layout(rect=(0, 0, 1, top_margin)) restringe esplicitamente
    l'area occupata dagli assi, lasciando una fascia libera sopra (da
    top_margin a 1.0) in cui la legenda viene poi posizionata: questo
    evita qualunque sovrapposizione con i tick label o le curve dei
    pannelli superiori, indipendentemente da quanto la griglia sia fitta.

    Le voci vengono lette da 'source_ax' (un Axes qualsiasi gia' popolato
    con le curve, es. il primo pannello), assumendo che tutti i pannelli
    della figura condividano lo stesso set di curve/etichette.

    Parameters:
    - fig         : Figure a cui aggiungere la legenda condivisa.
    - source_ax   : Axes da cui leggere handles/labels (es. axes[0]).
    - theta_deg   : float, usato come titolo della legenda.
    - top_margin  : float, bordo superiore riservato agli assi (0-1). Figure
                    piu' "fitte" (griglie a piu' righe) richiedono margini
                    piu' generosi; figure corte (una sola riga di pannelli)
                    ne richiedono meno in proporzione.
    - legend_y    : float opzionale, coordinata verticale della legenda; se
                    None viene centrata automaticamente nella fascia libera.
    """
    handles, labels = source_ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, top_margin))
    if legend_y is None:
        legend_y = top_margin + (1.0 - top_margin) * 0.5 + 0.02
    fig.legend(handles, labels, title=fr"$\theta = {theta_deg}^\circ$",
               loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=len(handles), frameon=False)


# # ==========================================
# # Plot 0: Total jump counts over time
# # ==========================================
# fig0, ax0 = plt.subplots(figsize=(10, 5))
# ax0.plot(times_ps, total_jumps, color=theta_color, alpha=0.9, linewidth=2.0,
#          label=f'Jumps per step (Total: {np.sum(total_jumps)})')
# ax0.set_xlabel("Time (ps)")
# ax0.set_ylabel("Number of Jumps")
# ax0.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right')
# save_fig(fig0, f'Total_Jumps_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 1: Populations - Redfield vs Collisional (trace) vs Avg trajectories
# # ==========================================
# fig1 = plt.figure(figsize=(14, 16))
# axes1 = make_axes_grid_7panel(fig1)

# for i in range(N_site):
#     ax = axes1[i]
#     ax.plot(times_ps, pop_redfield_exc[:, i], label='Redfield', linewidth=2, linestyle='--', color='black')
#     ax.plot(times_ps, pop_trace_coll_exc[:, i], label='Ancilla trace', linewidth=2, linestyle=':', color='green')
#     # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
#     ax.plot(times_ps, pop_traj_avg_exc[:, i], label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
#     ax.set_xlabel('Time (ps)')
#     ax.set_ylabel(fr'Population ${EXC_KET_LABELS[i]}$')

# # Legenda condivisa dell'intera figura, orizzontale su una sola riga sopra
# # i 7 pannelli (invece di ripeterla identica 7 volte, una per pannello).
# add_shared_legend(fig1, axes1[0], theta_deg)

# save_fig(fig1, f'Comparison_Eigen_Populations_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 2: Single trajectory vs isolated system (All Excitons)
# # ==========================================
# fig2 = plt.figure(figsize=(14, 16))
# axes2 = make_axes_grid_7panel(fig2)

# for i in range(N_site):
#     ax = axes2[i]
#     # Single trajectory e' un dato stocastico legato a questo angolo -> colore fisso
#     ax.plot(times_ps, pop_traj_exc[i, :, sample_idx], label='Single trajectory', linewidth=2, color=theta_color, alpha=0.9)
#     # NOTA: dinamica del sistema isolato rimossa da questo plot su richiesta.
#     ax.plot(times_ps, pop_redfield_exc[:, i], label='Redfield', linewidth=1.5, linestyle='--', color='black', alpha=0.8)
#     ax.set_xlabel('Time (ps)')
#     ax.set_ylabel(fr'Population ${EXC_KET_LABELS[i]}$')
#     # Asse y forzato a [-0.005, 0.020] per i pannelli Exciton 1, 2, 4, 5, 7
#     # (indici array 0, 1, 3, 4, 6), SOLO per theta = 0. Per gli altri angoli
#     # l'asse y resta automatico (scala determinata dai dati).
#     if theta_deg == 0 and i in (0, 1, 3, 4, 6):
#         ax.set_ylim(-0.005, 0.020)

# # Legenda condivisa dell'intera figura, orizzontale su una sola riga sopra
# # i 7 pannelli.
# add_shared_legend(fig2, axes2[0], theta_deg)

# save_fig(fig2, f'Single_EigenTraj_vs_Isolated_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 3: Many single trajectories (light) + avg + Redfield (All Excitons)
# # ==========================================
num_samples = min(100, n_traj)

# fig3 = plt.figure(figsize=(14, 16))
# axes3 = make_axes_grid_7panel(fig3)

# for i in range(N_site):
#     ax = axes3[i]
#     for k in range(num_samples):
#         ax.plot(times_ps, pop_traj_exc[i, :, k], color='gray', alpha=0.12, linewidth=0.5,
#                  label='Single trajectories' if k == 0 else "")
#     ax.plot(times_ps, pop_redfield_exc[:, i], label='Redfield', linewidth=2.2, linestyle='--', color='black')
#     # Avg trajectories e' il dato stocastico principale legato a questo angolo -> colore fisso
#     ax.plot(times_ps, pop_traj_avg_exc[:, i], label='Avg trajectories', linewidth=2.2, color=theta_color, alpha=0.9)
#     ax.set_xlabel('Time (ps)')
#     ax.set_ylabel(fr'Population ${EXC_KET_LABELS[i]}$')

# # Legenda condivisa dell'intera figura, orizzontale su una sola riga sopra
# # i 7 pannelli.
# add_shared_legend(fig3, axes3[0], theta_deg)

# save_fig(fig3, f'Many_EigenTraj_vs_Average_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 4: Coherences (real & imaginary), selected exciton pairs
# # ==========================================
# pairs_to_plot = [(0, 1), (1, 2), (0, 2)]

# fig4, axes4 = plt.subplots(len(pairs_to_plot), 2, figsize=(14, 5 * len(pairs_to_plot)))

# for row, (i, j) in enumerate(pairs_to_plot):
#     coh_redfield = rho_redfield_exc[:, i, j]
#     coh_trace_coll = rho_trace_coll_exc[:, i, j]
#     coh_avg = rho_traj_avg_exc[:, i, j]

#     ax_re = axes4[row, 0]
#     ax_re.plot(times_ps, np.real(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
#     ax_re.plot(times_ps, np.real(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
#     # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
#     ax_re.plot(times_ps, np.real(coh_avg), label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
#     ax_re.set_ylabel(fr'Re($\rho_{{{i+1}{j+1}}}$)')

#     ax_im = axes4[row, 1]
#     ax_im.plot(times_ps, np.imag(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
#     ax_im.plot(times_ps, np.imag(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
#     ax_im.plot(times_ps, np.imag(coh_avg), label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
#     ax_im.set_ylabel(fr'Im($\rho_{{{i+1}{j+1}}}$)')

# for ax in axes4.flat:
#     ax.set_xlabel('Time (ps)')
#     ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

# save_fig(fig4, f'Eigen_Coherences_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Bonus Plot 5: Purity Tr[rho^2] over time -- model consistency check
# # ==========================================
# def purity(rho_traj):
#     """Calculates the purity of a density matrix over time."""
#     return np.real(np.einsum('tij,tji->t', rho_traj, rho_traj))

# purity_redfield = purity(rho_redfield_exc)
# purity_trace_coll = purity(rho_trace_coll_exc)
# purity_traj = purity(rho_traj_avg_exc)

# fig5, ax5 = plt.subplots(figsize=(9, 5))
# ax5.plot(times_ps, purity_redfield, label='Redfield', linewidth=2, linestyle='--', color='black')
# ax5.plot(times_ps, purity_trace_coll, label='Ancilla trace', linewidth=2, linestyle=':', color='green')
# # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
# ax5.plot(times_ps, purity_traj, label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
# ax5.axhline(1.0 / N_site, color='gray', linestyle='-.', linewidth=1, label=f'Maximally mixed (1/{N_site})')
# ax5.set_xlabel('Time (ps)')
# ax5.set_ylabel('Purity Tr[$\\rho^2$]')
# ax5.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
# save_fig(fig5, f'Eigen_Purity_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Bonus Plot 6: Histogram of jumps per trajectory
# # ==========================================
# fig6, ax6 = plt.subplots(figsize=(8, 5))
# # Istogramma dei salti: dato stocastico legato a questo angolo -> colore fisso
# # label aggiunta per evitare il warning "No artists with labels found" quando
# # legend() viene chiamata senza che hist() abbia un parametro label esplicito.
# ax6.hist(n_jumps_per_traj, bins=min(50, int(n_jumps_per_traj.max()) + 1), color=theta_color, alpha=0.85,
#          label=f'$N_{{\\mathrm{{traj}}}} = {n_traj}$')
# ax6.set_xlabel('Number of jumps (over full trajectory)')
# ax6.set_ylabel('Number of trajectories')
# ax6.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
# save_fig(fig6, f'Jumps_Histogram_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 7: All populations in a single plot (Redfield vs Avg Trajectories)
# # ==========================================
# # NOTA: qui il colore-per-angolo non si applica. Ogni eccitone ha gia' un
# # colore diverso (palette tab10) per essere distinto dagli altri eccitoni
# # nello stesso grafico: la palette resta quindi invariata rispetto
# # all'originale.
# fig7, ax7 = plt.subplots(figsize=(10, 6))

# colors = plt.cm.tab10(np.linspace(0, 1, N_site))

# for i in range(N_site):
#     ax7.plot(times_ps, pop_redfield_exc[:, i], color=colors[i], linestyle='--', linewidth=2)
#     ax7.plot(times_ps, pop_traj_avg_exc[:, i], color=colors[i], linestyle='-', linewidth=2, alpha=0.7)

# ax7.set_xlabel('Time (ps)')
# ax7.set_ylabel('Population')

# custom_handles = [
#     Line2D([0], [0], color=colors[i], lw=2, label=f'Exciton {i+1}')
#     for i in range(N_site)
# ]

# ax7.legend(handles=custom_handles, title=fr"$\theta = {theta_deg}^\circ$",
#            loc='upper right', ncol=2, title_fontsize=11)
# save_fig(fig7, f'All_Eigen_Populations_Together_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 8: Trace distance (Redfield vs Avg Trajectories)
# # ==========================================
# td_time = np.zeros(n_times)

# for t in range(n_times):
#     td_time[t] = trace_distance_generic_njit(rho_redfield_exc[t], rho_traj_avg_exc[t])

# fig8, ax8 = plt.subplots(figsize=(8, 5))
# # Trace distance rispetto a Redfield: dato stocastico legato a questo angolo -> colore fisso
# ax8.plot(times_ps, td_time, color=theta_color, linewidth=2, label='Trace Distance')
# ax8.set_xlabel('Time (ps)')
# ax8.set_ylabel('Trace Distance')
# ax8.set_yscale('log')
# ax8.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
# save_fig(fig8, f'Eigen_Trace_Distance_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 9: Selected excitons (3, 6, 7) - Avg trajectories vs Redfield
# # ==========================================
# # 3 pannelli affiancati, uno per Exciton 3, 6, 7 (indici array 2, 5, 6).
# # Ogni pannello ha un ylim dedicato, richiesto per evidenziare la dinamica
# # specifica di quel sito senza farla schiacciare dalla scala automatica.
# selected_excitons = [
#     {'idx': 2, 'label': 'Exciton 3', 'ylim': (0.45, 0.8)},
#     {'idx': 5, 'label': 'Exciton 6', 'ylim': (0.19, 0.24)},
#     {'idx': 6, 'label': 'Exciton 7', 'ylim': (0.0, 0.023)},
# ]

# fig9, axes9 = plt.subplots(1, 3, figsize=(18, 5))

# for ax, exc in zip(axes9, selected_excitons):
#     i = exc['idx']
#     ax.plot(times_ps, pop_redfield_exc[:, i], label='Redfield', linewidth=2, linestyle='--', color='black')
#     # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
#     ax.plot(times_ps, pop_traj_avg_exc[:, i], label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
#     ax.set_xlabel('Time (ps)')
#     ax.set_ylabel(fr'Population $|{i+1}\rangle$')
#     ax.set_ylim(*exc['ylim'])

# # Legenda condivisa dell'intera figura, orizzontale su una sola riga sopra
# # i 3 pannelli.
# add_shared_legend(fig9, axes9[0], theta_deg, top_margin=0.80)

# save_fig(fig9, f'Selected_Excitons_Theta_{theta_str}', Output_dir)


# # ==========================================
# # Plot 10: Variance over time of selected eigenstates (across trajectories)
# # ==========================================
# # Var[pop_i](t) = <pop_i(t)^2>_traj - (<pop_i(t)>_traj)^2, calcolata
# # sull'intero insieme di traiettorie (asse n_traj), un valore per ogni
# # autostato e ogni istante temporale.
# var_pop_exc = np.var(pop_traj_exc, axis=2, ddof=0)  # (N_site, n_times)

# # Solo Exciton 3, 6, 7 (indici array 2, 5, 6): con sole 3 curve la sfumatura
# # per angolo (chiaro -> scuro) resta ben distinguibile, a differenza del
# # caso con tutti e 7 gli autostati dove le tonalita' vicine si confondevano.
# selected_variance_indices = [2, 5, 6]

# # Sfumatura di colori legata all'angolo corrente, campionata su sole 3
# # tonalita' (una per autostato selezionato): dal piu' chiaro (Exciton 3) al
# # piu' scuro (Exciton 7), nella stessa famiglia cromatica del colore base
# # di questo theta (es. arancione->rosso scuro per theta=0, azzurro->blu
# # scuro per theta=90).
# variance_gradient = get_angle_gradient(theta_deg, len(selected_variance_indices))

# fig10, ax10 = plt.subplots(figsize=(10, 6))

# for color, i in zip(variance_gradient, selected_variance_indices):
#     ax10.plot(times_ps, var_pop_exc[i], color=color, linewidth=2,
#                label=EXC_LABELS[i])

# ax10.set_xlabel('Time (ps)')
# ax10.set_ylabel('Variance of Population')
# # Asse y fissato allo stesso range per tutti gli angoli, cosi' la scala
# # resta direttamente confrontabile tra un theta e l'altro (angoli diversi
# # raggiungono valori di varianza molto diversi, quindi la scala automatica
# # renderebbe i plot non comparabili tra loro).
# ax10.set_ylim(-0.05, 0.26)

# # Legenda condivisa (qui la figura ha un solo pannello, ma la spostiamo
# # comunque sopra l'asse per coerenza con gli altri plot e per non coprire
# # le curve, specialmente vicino al bordo superiore del range fissato).
# add_shared_legend(fig10, ax10, theta_deg, top_margin=0.85)

# save_fig(fig10, f'Eigen_Population_Variance_Theta_{theta_str}', Output_dir)

# ==========================================
# Plot 11: Many single trajectories (light) + avg + Redfield (ONLY Exciton 3)
# ==========================================
fig11, ax11 = plt.subplots(figsize=(10, 6))

# Exciton 3 corresponds to index 2 in 0-indexed arrays
target_idx = 2  

# Plot a subset of single trajectories as light background lines
for k in range(num_samples):
    ax11.plot(times_ps, pop_traj_exc[target_idx, :, k], color='gray', alpha=0.12, linewidth=0.5,
              label='Single trajectories' if k == 0 else "")

# Plot theoretical Redfield baseline
ax11.plot(times_ps, pop_redfield_exc[:, target_idx], label='Secular Redfield', 
          linewidth=2.2, linestyle='--', color='black')

# Plot the stochastic average for this angle using the specific theta color
ax11.plot(times_ps, pop_traj_avg_exc[:, target_idx], label='Avg trajectories', 
          linewidth=2.2, color=theta_color, alpha=0.9)

# Set labels using the existing ket notation list
ax11.set_xlabel('Time (ps)')
ax11.set_ylabel(fr'Population ${EXC_KET_LABELS[target_idx]}$')

# Add legend directly to the single axis (no need for the shared legend function here)
ax11.legend(loc='lower left')

# Save the figure in the specific output directory
save_fig(fig11, f'Many_EigenTraj_vs_Average_Exciton3_Theta_{theta_str}', Output_dir)


print("All eigenstate basis plots generated and saved successfully.")