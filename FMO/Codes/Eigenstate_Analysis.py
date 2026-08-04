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
from plot_style import set_thesis_style, save_fig

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

# ==========================================
# Plot 0: Total jump counts over time
# ==========================================
fig0, ax0 = plt.subplots(figsize=(10, 5))
ax0.plot(times, total_jumps, color='purple', alpha=0.8, linewidth=1.5,
         label=f'Jumps per step (Total: {np.sum(total_jumps)})')
ax0.set_xlabel("Time (fs)")
ax0.set_ylabel("Number of Jumps")
ax0.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)
save_fig(fig0, f'Total_Jumps_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 1: Populations - Redfield vs Collisional (trace) vs Avg trajectories
# ==========================================
fig1 = plt.figure(figsize=(20, 8))
gs1 = fig1.add_gridspec(2, 8)
axes1 = []
for i in range(4):
    axes1.append(fig1.add_subplot(gs1[0, i*2:(i+1)*2]))
for c_start, c_end in [(1, 3), (3, 5), (5, 7)]:
    axes1.append(fig1.add_subplot(gs1[1, c_start:c_end]))
axes1 = np.array(axes1)

for i in range(N_site):
    ax = axes1[i]
    ax.plot(times, pop_redfield_exc[:, i], label='Redfield', linewidth=2, linestyle='--', color='black')
    ax.plot(times, pop_trace_coll_exc[:, i], label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax.plot(times, pop_traj_avg_exc[:, i], label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(f'Population {EXC_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)

save_fig(fig1, f'Comparison_Eigen_Populations_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 2: Single trajectory vs isolated system (All Excitons)
# ==========================================
fig2 = plt.figure(figsize=(20, 8))
gs2 = fig2.add_gridspec(2, 8)
axes2 = []
for i in range(4):
    axes2.append(fig2.add_subplot(gs2[0, i*2:(i+1)*2]))
for c_start, c_end in [(1, 3), (3, 5), (5, 7)]:
    axes2.append(fig2.add_subplot(gs2[1, c_start:c_end]))
axes2 = np.array(axes2)

for i in range(N_site):
    ax = axes2[i]
    ax.plot(times, pop_traj_exc[i, :, sample_idx], label='Single trajectory', linewidth=1.8, color='blue', alpha=0.85)
    ax.plot(times, pop_iso_exc[:, i], label='Isolated system', linewidth=2, linestyle=':', color='red')
    ax.plot(times, pop_redfield_exc[:, i], label='Redfield', linewidth=1.5, linestyle='--', color='black', alpha=0.8)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(f'Population {EXC_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)

save_fig(fig2, f'Single_EigenTraj_vs_Isolated_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 3: Many single trajectories (light) + avg + Redfield (All Excitons)
# ==========================================
num_samples = min(100, n_traj)

fig3 = plt.figure(figsize=(20, 8))
gs3 = fig3.add_gridspec(2, 8)
axes3 = []
for i in range(4):
    axes3.append(fig3.add_subplot(gs3[0, i*2:(i+1)*2]))
for c_start, c_end in [(1, 3), (3, 5), (5, 7)]:
    axes3.append(fig3.add_subplot(gs3[1, c_start:c_end]))
axes3 = np.array(axes3)

for i in range(N_site):
    ax = axes3[i]
    for k in range(num_samples):
        ax.plot(times, pop_traj_exc[i, :, k], color='gray', alpha=0.12, linewidth=0.5,
                 label='Single trajectories' if k == 0 else "")
    ax.plot(times, pop_redfield_exc[:, i], label='Redfield', linewidth=2.2, linestyle='--', color='black')
    ax.plot(times, pop_traj_avg_exc[:, i], label='Avg trajectories', linewidth=2.2, color='blue', alpha=0.9)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(f'Population {EXC_LABELS[i]}')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)

save_fig(fig3, f'Many_EigenTraj_vs_Average_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 4: Coherences (real & imaginary), selected exciton pairs
# ==========================================
pairs_to_plot = [(0, 1), (1, 2), (0, 2)]

fig4, axes4 = plt.subplots(len(pairs_to_plot), 2, figsize=(14, 5 * len(pairs_to_plot)))

for row, (i, j) in enumerate(pairs_to_plot):
    coh_redfield = rho_redfield_exc[:, i, j]
    coh_trace_coll = rho_trace_coll_exc[:, i, j]
    coh_avg = rho_traj_avg_exc[:, i, j]

    ax_re = axes4[row, 0]
    ax_re.plot(times, np.real(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_re.plot(times, np.real(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax_re.plot(times, np.real(coh_avg), label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax_re.set_ylabel(fr'Re($\rho_{{{i+1}{j+1}}}$)')

    ax_im = axes4[row, 1]
    ax_im.plot(times, np.imag(coh_redfield), label='Redfield', linewidth=2, linestyle='--', color='black')
    ax_im.plot(times, np.imag(coh_trace_coll), label='Ancilla trace', linewidth=2, linestyle=':', color='green')
    ax_im.plot(times, np.imag(coh_avg), label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
    ax_im.set_ylabel(fr'Im($\rho_{{{i+1}{j+1}}}$)')

for ax in axes4.flat:
    ax.set_xlabel('Time (fs)')
    ax.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)

save_fig(fig4, f'Eigen_Coherences_Theta_{theta_str}', Output_dir)


# ==========================================
# Bonus Plot 5: Purity Tr[rho^2] over time -- model consistency check
# ==========================================
def purity(rho_traj):
    """Calculates the purity of a density matrix over time."""
    return np.real(np.einsum('tij,tji->t', rho_traj, rho_traj))

purity_redfield = purity(rho_redfield_exc)
purity_trace_coll = purity(rho_trace_coll_exc)
purity_traj = purity(rho_traj_avg_exc)

fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.plot(times, purity_redfield, label='Redfield', linewidth=2, linestyle='--', color='black')
ax5.plot(times, purity_trace_coll, label='Ancilla trace', linewidth=2, linestyle=':', color='green')
ax5.plot(times, purity_traj, label='Avg trajectories', linewidth=2, color='blue', alpha=0.7)
ax5.axhline(1.0 / N_site, color='gray', linestyle='-.', linewidth=1, label=f'Maximally mixed (1/{N_site})')
ax5.set_xlabel('Time (fs)')
ax5.set_ylabel('Purity Tr[$\\rho^2$]')
ax5.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
save_fig(fig5, f'Eigen_Purity_Theta_{theta_str}', Output_dir)


# ==========================================
# Bonus Plot 6: Histogram of jumps per trajectory
# ==========================================
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.hist(n_jumps_per_traj, bins=min(50, int(n_jumps_per_traj.max()) + 1), color='purple', alpha=0.75)
ax6.set_xlabel('Number of jumps (over full trajectory)')
ax6.set_ylabel('Number of trajectories')
ax6.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
save_fig(fig6, f'Jumps_Histogram_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 7: All populations in a single plot (Redfield vs Avg Trajectories)
# ==========================================
fig7, ax7 = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, N_site))

for i in range(N_site):
    ax7.plot(times, pop_redfield_exc[:, i], color=colors[i], linestyle='--', linewidth=2)
    ax7.plot(times, pop_traj_avg_exc[:, i], color=colors[i], linestyle='-', linewidth=2, alpha=0.7)

ax7.set_xlabel('Time (fs)')
ax7.set_ylabel('Population')

custom_handles = [
    Line2D([0], [0], color=colors[i], lw=2, label=f'Exciton {i+1}')
    for i in range(N_site)
]

ax7.legend(handles=custom_handles, title=fr"$\theta = {theta_deg}^\circ$", 
           loc='upper right', ncol=2, title_fontsize=11)
save_fig(fig7, f'All_Eigen_Populations_Together_Theta_{theta_str}', Output_dir)


# ==========================================
# Plot 8: Trace distance (Redfield vs Avg Trajectories)
# ==========================================
td_time = np.zeros(n_times)

for t in range(n_times):
    td_time[t] = trace_distance_generic_njit(rho_redfield_exc[t], rho_traj_avg_exc[t])

fig8, ax8 = plt.subplots(figsize=(8, 5))
ax8.plot(times, td_time, color='red', linewidth=2, label='Trace Distance')
ax8.set_xlabel('Time (fs)')
ax8.set_ylabel('Trace Distance')
ax8.set_yscale('log') 
ax8.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
save_fig(fig8, f'Eigen_Trace_Distance_Theta_{theta_str}', Output_dir)

print("All eigenstate basis plots generated and saved successfully.")