#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from numba import njit

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig, get_angle_color

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
# (nero/verde), per essere riconoscibili indipendentemente dall'angolo. Il
# Plot 7 fa eccezione: li' ogni sito ha gia' un colore diverso (palette
# tab10) per distinguerli tra loro nello stesso grafico, quindi non usa
# get_angle_color.
theta_color = get_angle_color(theta_deg)

# --- Must match the values used in the main simulation script ---
dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/Plot/Populations/{theta_str}"
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
eigenvectors = data['eigenvectors']
psi0_exc = data['psi0_exc']

total_jumps = data['total_jumps']
jump_counts = data['jump_counts']            # (n_times, n_traj)

psi_traj_exc = data['psi_traj']                  # (N_site, n_times, n_traj), exciton basis, complex64
pop_traj_avg = data['pop_traj_mean']        # (n_times, N_site)
pop_traj_stderr = data['pop_traj_stderr']

rho_redfield_site = data['rho_redfield_site']       # (n_times, N_site, N_site)
rho_trace_coll_site = data['rho_trace_coll_site']  # (n_times, N_site, N_site)
rho_traj_avg_site = data['rho_traj_avg_site']        # (n_times, N_site, N_site)

n_times, n_traj = jump_counts.shape

# ==========================
# Site-basis single-trajectory populations
# ==========================
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)   # (N_site, n_times, n_traj)
pop_traj_site = np.abs(psi_traj_site) ** 2                          # (N_site, n_times, n_traj)

# ==========================
# Isolated system (no collisions) recomputation
# ==========================
phase = np.exp(-1j * np.outer(times, eigenergies))          # (n_times, N)
psi_iso_exc = phase * psi0_exc[None, :]                      # (n_times, N)
psi_iso_site = psi_iso_exc @ eigenvectors.T                  # (n_times, N_site)
pop_iso_site = np.abs(psi_iso_site) ** 2                     # (n_times, N_site)

# ==========================
# Redfield / collisional / MC-avg populations (site basis)
# ==========================
pop_redfield_site = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))
pop_trace_coll_site = np.real(np.diagonal(rho_trace_coll_site, axis1=1, axis2=2))
pop_traj_avg_site = np.real(np.diagonal(rho_traj_avg_site, axis1=1, axis2=2))

# ==========================
# Identify trajectories that experienced at least one jump
# ==========================
n_jumps_per_traj = jump_counts.sum(axis=0)   # (n_traj,)
jump_indices = np.where(n_jumps_per_traj > 0)[0]
print(f"Total trajectories: {n_traj}")
print(f"Trajectories with at least one jump: {len(jump_indices)}")
sample_idx = jump_indices[0] if len(jump_indices) > 0 else 0
print(f"Selected sample_idx for single-trajectory plots: {sample_idx}")

SITE_LABELS = [f"Site {i+1}" for i in range(N_site)]


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


# ==========================================
# Plot 0: Total jump counts over time
# ==========================================
fig0, ax0 = plt.subplots(figsize=(10, 5))
ax0.plot(times_ps, total_jumps, color=theta_color, alpha=0.9, linewidth=2.0,
         label=f'Jumps per step (Total: {np.sum(total_jumps)})')
ax0.set_xlabel("Time (ps)")
ax0.set_ylabel("Number of Jumps")
ax0.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right')
save_fig(fig0, f'Total_Jumps_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 1: Populations - Redfield vs Collisional vs Avg (Centered Grid Layout)
# ==========================================
fig1 = plt.figure(figsize=(14, 16))
axes1 = make_axes_grid_7panel(fig1)

for i in range(N_site):
    ax = axes1[i]
    ax.plot(times_ps, pop_redfield_site[:, i], label='Redfield', linewidth=2, linestyle='--', color='black')
    ax.plot(times_ps, pop_trace_coll_site[:, i], label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
    ax.plot(times_ps, pop_traj_avg_site[:, i], label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(f'Population {SITE_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

save_fig(fig1, f'Comparison_Populations_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 2: Single trajectory vs isolated system for all sites (Centered Grid Layout)
# ==========================================
fig2 = plt.figure(figsize=(14, 16))
axes2 = make_axes_grid_7panel(fig2)

for i in range(N_site):
    ax = axes2[i]
    # Single trajectory e' un dato stocastico legato a questo angolo -> colore fisso
    ax.plot(times_ps, pop_traj_site[i, :, sample_idx], label='Single trajectory', linewidth=2, color=theta_color, alpha=0.9)
    # NOTA: dinamica del sistema isolato rimossa da questo plot su richiesta.
    ax.plot(times_ps, pop_redfield_site[:, i], label='Redfield', linewidth=1.5, linestyle='--', color='black', alpha=0.8)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(f'Population {SITE_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
    # Asse y forzato a [-0.005, 0.020] per i pannelli Site 1, 2, 4, 5, 7
    # (indici array 0, 1, 3, 4, 6), SOLO per theta = 0. Per gli altri angoli
    # l'asse y resta automatico (scala determinata dai dati).
    if theta_deg == 0 and i in (0, 1, 3, 4, 6):
        ax.set_ylim(-0.005, 0.020)

save_fig(fig2, f'Single_Traj_vs_Isolated_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 3: Many single trajectories + avg + Redfield for all sites (Centered Grid Layout)
# ==========================================
num_samples = min(100, n_traj)

fig3 = plt.figure(figsize=(14, 16))
axes3 = make_axes_grid_7panel(fig3)

for i in range(N_site):
    ax = axes3[i]
    for k in range(num_samples):
        ax.plot(times_ps, pop_traj_site[i, :, k], color='gray', alpha=0.12, linewidth=0.5,
                 label='Single trajectories' if k == 0 else "")
    ax.plot(times_ps, pop_redfield_site[:, i], label='Redfield', linewidth=2.2, linestyle='--', color='black')
    # Avg trajectories e' il dato stocastico principale legato a questo angolo -> colore fisso
    ax.plot(times_ps, pop_traj_avg_site[:, i], label='Avg trajectories', linewidth=2.2, color=theta_color, alpha=0.9)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(f'Population {SITE_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

save_fig(fig3, f'Many_Traj_vs_Average_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 4: Coherences (real & imaginary), selected site pairs
# ==========================================
pairs_to_plot = [(0, 1), (1, 2), (0, 2)]

fig4, axes4 = plt.subplots(len(pairs_to_plot), 2, figsize=(14, 5 * len(pairs_to_plot)))

for row, (i, j) in enumerate(pairs_to_plot):
    coh_redfield = rho_redfield_site[:, i, j]
    coh_trace_coll = rho_trace_coll_site[:, i, j]
    coh_avg = rho_traj_avg_site[:, i, j]

    ax_re = axes4[row, 0]
    ax_re.plot(times_ps, np.real(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_re.plot(times_ps, np.real(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    # Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
    ax_re.plot(times_ps, np.real(coh_avg), label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
    ax_re.set_ylabel(fr'Re($\rho_{{{i+1}{j+1}}}$)')

    ax_im = axes4[row, 1]
    ax_im.plot(times_ps, np.imag(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_im.plot(times_ps, np.imag(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax_im.plot(times_ps, np.imag(coh_avg), label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
    ax_im.set_ylabel(fr'Im($\rho_{{{i+1}{j+1}}}$)')

for ax in axes4.flat:
    ax.set_xlabel('Time (ps)')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')

save_fig(fig4, f'Coherences_Theta_{theta_str}', Output_dir)


# ==========================================
# Bonus Plot 5: Purity Tr[rho^2] over time
# ==========================================
def purity(rho_traj):
    return np.real(np.einsum('tij,tji->t', rho_traj, rho_traj))

purity_redfield = purity(rho_redfield_site)
purity_trace_coll = purity(rho_trace_coll_site)
purity_traj = purity(rho_traj_avg_site)

fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.plot(times_ps, purity_redfield, label='Redfield', linewidth=2, linestyle='--', color='black')
ax5.plot(times_ps, purity_trace_coll, label='Ancilla trace', linewidth=2, linestyle=':', color='green')
# Avg trajectories e' un dato stocastico legato a questo angolo -> colore fisso
ax5.plot(times_ps, purity_traj, label='Avg trajectories', linewidth=2, color=theta_color, alpha=0.9)
ax5.axhline(1.0 / N_site, color='gray', linestyle='-.', linewidth=1, label=f'Maximally mixed (1/{N_site})')
ax5.set_xlabel('Time (ps)')
ax5.set_ylabel('Purity Tr[$\\rho^2$]')
ax5.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
save_fig(fig5, f'Purity_Theta_{theta_str}', Output_dir)


# ==========================================
# Bonus Plot 6: Histogram of jumps per trajectory
# ==========================================
fig6, ax6 = plt.subplots(figsize=(8, 5))
# Istogramma dei salti: dato stocastico legato a questo angolo -> colore fisso.
# label aggiunta per evitare il warning "No artists with labels found" quando
# legend() viene chiamata senza che hist() abbia un parametro label esplicito.
ax6.hist(n_jumps_per_traj, bins=min(50, int(n_jumps_per_traj.max()) + 1), color=theta_color, alpha=0.85,
         label=f'$N_{{\\mathrm{{traj}}}} = {n_traj}$')
ax6.set_xlabel('Number of jumps (over full trajectory)')
ax6.set_ylabel('Number of trajectories')
ax6.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
save_fig(fig6, f'Jumps_Histogram_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 7: All populations in a single plot (Redfield vs Avg Trajectories)
# ==========================================
# NOTA: qui il colore-per-angolo non si applica. Ogni sito ha gia' un
# colore diverso (palette tab10) per essere distinto dagli altri siti
# nello stesso grafico: la palette resta quindi invariata rispetto
# all'originale.
fig7, ax7 = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, N_site))

for i in range(N_site):
    ax7.plot(times_ps, pop_redfield_site[:, i], color=colors[i], linestyle='--', linewidth=2)
    ax7.plot(times_ps, pop_traj_avg_site[:, i], color=colors[i], linestyle='-', linewidth=2, alpha=0.7)

ax7.set_xlabel('Time (ps)')
ax7.set_ylabel('Population')

custom_handles = [
    Line2D([0], [0], color=colors[i], lw=2, label=f'Site {i+1}')
    for i in range(N_site)
]

ax7.legend(handles=custom_handles, title=fr"$\theta = {theta_deg}^\circ$",
           loc='upper right', ncol=2)
save_fig(fig7, f'All_Populations_Together_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 8: Trace distance (Redfield vs Avg Trajectories)
# ==========================================
td_time = np.zeros(n_times)

for t in range(n_times):
    td_time[t] = trace_distance_generic_njit(rho_redfield_site[t], rho_traj_avg_site[t])

fig8, ax8 = plt.subplots(figsize=(8, 5))
# Trace distance rispetto a Redfield: dato stocastico legato a questo angolo -> colore fisso
ax8.plot(times_ps, td_time, color=theta_color, linewidth=2, label='Trace Distance')
ax8.set_xlabel('Time (ps)')
ax8.set_ylabel('Trace Distance')
ax8.set_yscale('log')
ax8.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best')
save_fig(fig8, f'Trace_Distance_Theta_{theta_str}', Output_dir)

print("All site population plots generated and saved successfully.")