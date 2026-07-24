#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================
# Input Parsing from Bash
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 0.0  # Default fallback if run manually

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
# psi_traj is in the exciton basis -> transform to site basis
# ==========================
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)   # (N_site, n_times, n_traj)
pop_traj_site = np.abs(psi_traj_site) ** 2                          # (N_site, n_times, n_traj)

# ==========================
# Isolated system (no collisions): recomputed on the fly from
# eigenergies, eigenvectors, psi0_exc, times -- no need to store it upfront
# ==========================
phase = np.exp(-1j * np.outer(times, eigenergies))          # (n_times, N)
psi_iso_exc = phase * psi0_exc[None, :]                      # (n_times, N)
psi_iso_site = psi_iso_exc @ eigenvectors.T                  # (n_times, N_site)
pop_iso_site = np.abs(psi_iso_site) ** 2                     # (n_times, N_site)

# ==========================
# Redfield / collisional / MC-avg populations (site basis)
# ==========================
pop_redfield_site = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))       # (n_times, N_site)
pop_trace_coll_site = np.real(np.diagonal(rho_trace_coll_site, axis1=1, axis2=2))  # (n_times, N_site)
pop_traj_avg_site = np.real(np.diagonal(rho_traj_avg_site, axis1=1, axis2=2))              # should ~ pop_traj_avg

# ==========================
# Identify trajectories that experienced at least one jump (theta=0 only meaningful)
# Uses the exact jump record, not a population-threshold heuristic
# ==========================
n_jumps_per_traj = jump_counts.sum(axis=0)   # (n_traj,)
jump_indices = np.where(n_jumps_per_traj > 0)[0]
print(f"Total trajectories: {n_traj}")
print(f"Trajectories with at least one jump: {len(jump_indices)}")
sample_idx = jump_indices[0] if len(jump_indices) > 0 else 0
print(f"Selected sample_idx for single-trajectory plots: {sample_idx}")

# ===========================
# General plot setup
# ===========================
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':',
    'figure.autolayout': True
})

def save_fig(fig, filename):
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)

SITE_LABELS = [f"Site {i+1}" for i in range(N_site)]

# ==========================================
# Plot 0: Total jump counts over time
# ==========================================
fig0, ax0 = plt.subplots(figsize=(10, 5))
ax0.plot(times, total_jumps, color='purple', alpha=0.8, linewidth=1.5,
         label=f'Jumps per step (Total: {np.sum(total_jumps)})')
ax0.set_title(f"Total Jumps Over Time (Theta={theta_deg} deg, dt={dt_val})")
ax0.set_xlabel("Time (fs)")
ax0.set_ylabel("Number of Jumps")
ax0.legend(loc='upper right')
save_fig(fig0, f'Total_Jumps_Theta_{theta_str}')


# ==========================================
# Plot 1: Populations - Redfield vs Collisional (trace) vs Avg trajectories
# One subplot per site (grid layout)
# ==========================================
ncols = 4
nrows = int(np.ceil(N_site / ncols))
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
axes1 = np.atleast_1d(axes1).flatten()

for i in range(N_site):
    ax = axes1[i]
    ax.plot(times, pop_redfield_site[:, i], label='Redfield', linewidth=2, linestyle='--', color='black')
    ax.plot(times, pop_trace_coll_site[:, i], label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax.plot(times, pop_traj_avg_site[:, i], label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax.set_title(SITE_LABELS[i])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Population')
    ax.legend(loc='best')

for j in range(N_site, len(axes1)):
    axes1[j].axis('off')

fig1.suptitle(f'Theta={theta_deg} deg - Redfield vs Ancilla-trace vs Avg-trajectories | dt={dt_val}, N_traj={n_traj}')
save_fig(fig1, f'Comparison_Populations_Theta_{theta_str}')


# ==========================================
# Plot 2: Single trajectory (with jump) vs isolated system, sites 1,2,3
# ==========================================
sites_to_plot = [0, 1, 2]   # sites 1, 2, 3

fig2, axes2 = plt.subplots(1, len(sites_to_plot), figsize=(6 * len(sites_to_plot), 5))
for ax, i in zip(np.atleast_1d(axes2), sites_to_plot):
    ax.plot(times, pop_traj_site[i, :, sample_idx], label='Single trajectory', linewidth=1.8, color='blue', alpha=0.85)
    ax.plot(times, pop_iso_site[:, i], label='Isolated system (no collisions)', linewidth=2, linestyle=':', color='red')
    ax.plot(times, pop_redfield_site[:, i], label='Redfield', linewidth=1.5, linestyle='--', color='black', alpha=0.8)
    ax.set_title(SITE_LABELS[i])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Population')
    ax.legend(loc='best')

fig2.suptitle(f'Theta={theta_deg} deg - Single Trajectory (idx={sample_idx}) vs Isolated System')
save_fig(fig2, f'Single_Traj_vs_Isolated_Theta_{theta_str}')


# ==========================================
# Plot 3: Many single trajectories (light) + avg + Redfield, sites 1,2,3
# ==========================================
num_samples = min(100, n_traj)

fig3, axes3 = plt.subplots(1, len(sites_to_plot), figsize=(6 * len(sites_to_plot), 5))
for ax, i in zip(np.atleast_1d(axes3), sites_to_plot):
    for k in range(num_samples):
        ax.plot(times, pop_traj_site[i, :, k], color='gray', alpha=0.12, linewidth=0.5,
                 label='Single trajectories' if k == 0 else "")
    ax.plot(times, pop_redfield_site[:, i], label='Redfield', linewidth=2.2, linestyle='--', color='black')
    ax.plot(times, pop_traj_avg_site[:, i], label='Avg trajectories', linewidth=2.2, color='blue', alpha=0.9)
    ax.set_title(SITE_LABELS[i])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Population')
    ax.legend(loc='best')

fig3.suptitle(f'Theta={theta_deg} deg - {num_samples} Trajectories vs Average vs Redfield')
save_fig(fig3, f'Many_Traj_vs_Average_Theta_{theta_str}')


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
    ax_re.plot(times, np.real(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_re.plot(times, np.real(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax_re.plot(times, np.real(coh_avg), label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax_re.set_title(f'Re[$\\rho_{{{i+1}{j+1}}}$]')

    ax_im = axes4[row, 1]
    ax_im.plot(times, np.imag(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_im.plot(times, np.imag(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax_im.plot(times, np.imag(coh_avg), label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax_im.set_title(f'Im[$\\rho_{{{i+1}{j+1}}}$]')

for ax in axes4.flat:
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Value')
    ax.legend(loc='best')

fig4.suptitle(f'Theta={theta_deg} deg - Coherences: Redfield vs Ancilla-trace vs Avg-trajectories', y=0.995)
save_fig(fig4, f'Coherences_Theta_{theta_str}')


# ==========================================
# Bonus Plot 5: Purity Tr[rho^2] over time -- model consistency check
# ==========================================
def purity(rho_traj):
    return np.real(np.einsum('tij,tji->t', rho_traj, rho_traj))

purity_redfield = purity(rho_redfield_site)
purity_trace_coll = purity(rho_trace_coll_site)
purity_traj = purity(rho_traj_avg_site)

fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.plot(times, purity_redfield, label='Redfield', linewidth=2, linestyle='--', color='black')
ax5.plot(times, purity_trace_coll, label='Ancilla trace', linewidth=2, linestyle=':', color='green')
ax5.plot(times, purity_traj, label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
ax5.axhline(1.0 / N_site, color='gray', linestyle='-.', linewidth=1, label=f'Maximally mixed (1/{N_site})')
ax5.set_title(f'Theta={theta_deg} deg - Purity Tr[$\\rho^2$] Over Time')
ax5.set_xlabel('Time (fs)')
ax5.set_ylabel('Purity')
ax5.legend(loc='best')
save_fig(fig5, f'Purity_Theta_{theta_str}')


# ==========================================
# Bonus Plot 6: Histogram of jumps per trajectory (meaningful mainly at theta=0)
# ==========================================
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.hist(n_jumps_per_traj, bins=min(50, int(n_jumps_per_traj.max()) + 1), color='purple', alpha=0.75)
ax6.set_title(f'Theta={theta_deg} deg - Distribution of Total Jumps per Trajectory')
ax6.set_xlabel('Number of jumps (over full trajectory)')
ax6.set_ylabel('Number of trajectories')
save_fig(fig6, f'Jumps_Histogram_Theta_{theta_str}')

print("All plots generated and saved successfully.")