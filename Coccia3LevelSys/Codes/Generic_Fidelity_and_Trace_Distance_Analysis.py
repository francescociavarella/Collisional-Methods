#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange

# =====================================================================
# NUMBA OPTIMIZED METRIC FUNCTIONS
# =====================================================================

@njit
def fidelity_generic_njit(rho, sigma):
    """Numba-compatible generalized fidelity."""
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho, 0.0) 
    
    # Explicitly cast the float64 diagonal matrix to complex128
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

@njit(parallel=True)
def compute_metrics_all_trajectories(rho_all, rho_lind):
    """
    Computes fidelity and trace distance for ALL individual trajectories.
    Uses generic 3x3 density matrices and parallel processing (prange).
    """
    n_times = rho_all.shape[2]
    N_traj = rho_all.shape[3]
    
    fidelity_matrix = np.zeros((n_times, N_traj))
    trace_dist_matrix = np.zeros((n_times, N_traj))
    
    for t in range(n_times):
        lindblad_t = rho_lind[t]  
        for n in prange(N_traj):
            rho_traj_t = np.zeros((3, 3), dtype=np.complex128)
            for i in range(3):
                for j in range(3):
                    rho_traj_t[i, j] = rho_all[i, j, t, n]
            
            fidelity_matrix[t, n] = fidelity_generic_njit(rho_traj_t, lindblad_t)
            trace_dist_matrix[t, n] = trace_distance_generic_njit(rho_traj_t, lindblad_t)
            
    return fidelity_matrix, trace_dist_matrix

def fidelity_generic(rho, sigma):
    """Standard Python wrapper for generic fidelity calculation."""
    return fidelity_generic_njit(rho, sigma)

def trace_distance_generic(rho, sigma):
    """Standard Python wrapper for generic trace distance calculation."""
    return trace_distance_generic_njit(rho, sigma)

# ==================
# Angles Definition
# ==================

# Parse arguments from Bash script
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1]) 
else:
    phi_deg = 0.0  # Default to QJ limit for testing

phi_rad = np.radians(phi_deg)

# Simulation parameters used for file naming
dt = 0.01
N_traj = 10000
dt_str = f"{dt:.6f}".replace(".", "p")
phi_str = f"{phi_rad:.4f}".replace(".", "p")

# --- Results Directory and Output Setup ---
data_dir = "../Results/Data/Complete_rho/"
metrics_dir = "../Results/Data/Metrics/"
Output_dir = f"../Results/Plot/Metrics/{int(phi_deg)}"

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(Output_dir, exist_ok=True)

# Load Simulation Data
fname = os.path.join(data_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Angle = {phi_deg} degrees.")
except FileNotFoundError:
    print(f"Error: File {fname} not found.")
    sys.exit(1)

times = data['times']
rho_all = data['rho_tot_all']
rho_lind = data['rho_list_lindblad']

# ======================================
# Metrics Calculation / Loading
# ======================================

metrics_filename = f"{phi_rad:.4f}_F_TD_Metrics_dt{dt_str}_Ntraj{N_traj}.npz".replace(".", "p")
save_path = os.path.join(metrics_dir, metrics_filename)

if os.path.exists(save_path):
    print("Loading pre-computed metrics data...")
    loaded_data = np.load(save_path)
    fid_matrix = loaded_data['fid_matrix']
    td_matrix = loaded_data['td_matrix']
else:
    print("Computing metrics for all trajectories. This may take a moment...")
    fid_matrix, td_matrix = compute_metrics_all_trajectories(rho_all, rho_lind)
    np.savez_compressed(save_path, fid_matrix=fid_matrix, td_matrix=td_matrix)
    print(f"Metrics saved to {save_path}")

# ===========================
# General Setup for Plotting
# ===========================
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.figsize': (10, 5), 'axes.grid': True, 'grid.alpha': 0.3,
    'grid.linestyle': ':', 'figure.autolayout': True, 
    'axes.formatter.useoffset': False 
})

def save_fig(fig, filename):
    """Saves the figure cleanly in the dynamically created output directory"""
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)

# =====================================================================
# PLOT 1: Global Convergence (Averaged States)
# =====================================================================
plt.close('all')

N_max = rho_all.shape[3]
N_list = sorted(list(set([100, 1000, min(10000, N_max), N_max])))

fig01, axes = plt.subplots(1, 2, figsize=(16, 5))
n_times = len(times)

for N in N_list:
    # Average the density matrix over the first N trajectories
    rho_avg_N = rho_all[:, :, :, :N].mean(axis=3)
    
    fidelity_list = np.zeros(n_times)
    td_list = np.zeros(n_times)

    for t in range(n_times):
        fidelity_list[t] = fidelity_generic(rho_lind[t], rho_avg_N[:, :, t])
        td_list[t] = trace_distance_generic(rho_lind[t], rho_avg_N[:, :, t])

    axes[0].plot(times, td_list, label=f'N = {N}', alpha=0.9, linewidth=1.5)
    axes[1].plot(times, fidelity_list, label=f'N = {N}', alpha=0.9, linewidth=1.5)

# Formatting Trace Distance
axes[0].set_xlabel("Time", fontsize=12)
axes[0].set_ylabel(r"Trace Distance $\mathcal{T}(\langle\rho\rangle_N, \rho_L)$", fontsize=12)
axes[0].set_title("Trace Distance Convergence vs Lindblad", fontsize=14)
axes[0].legend(loc='best')
axes[0].set_ylim(bottom=0)

# Formatting Fidelity
axes[1].set_xlabel("Time", fontsize=12)
axes[1].set_ylabel(r"Fidelity $\mathcal{F}(\langle\rho\rangle_N, \rho_L)$", fontsize=12)
axes[1].set_title("Fidelity Convergence vs Lindblad", fontsize=14)
axes[1].legend(loc='best')

fig01.suptitle(f'[Angle: {phi_deg}°] Ensemble Size Convergence | dt={dt}', fontsize=16, y=1.02)
filename_avg = f"Angle_{int(phi_deg)}_Global_Convergence_Metrics_dt{dt_str}"
save_fig(fig01, filename_avg)

# =========================================================
# PLOT 2: Single Trajectory Comparison (Jump vs No-Jump)
# =========================================================
plt.close('all')

# Extract the real population of state |1> for all trajectories across all time
pop_11_all = np.real(rho_all[1, 1, :, :])

# Find the maximum population in state |1> achieved by each trajectory
max_pop_1_per_traj = np.max(pop_11_all, axis=0)

# Identify trajectories based on whether they jumped (high pop in |1>) or not
jump_candidates = np.where(max_pop_1_per_traj > 0.8)[0]
no_jump_candidates = np.where(max_pop_1_per_traj < 0.1)[0]

# Fallback indices if criteria are not met (e.g. in SD limit where pure jumps don't occur)
idx_jump = jump_candidates[0] if len(jump_candidates) > 0 else 0
idx_no_jump = no_jump_candidates[0] if len(no_jump_candidates) > 0 else 1

# Extract metrics for these specific trajectories
td_jump = td_matrix[:, idx_jump]
fid_jump = fid_matrix[:, idx_jump]

td_no_jump = td_matrix[:, idx_no_jump]
fid_no_jump = fid_matrix[:, idx_no_jump]

fig03, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot Trace Distance
axes[0].plot(times, td_jump, label=f'Jumped Traj (idx={idx_jump})', color='crimson', linewidth=1.5, alpha=0.9)
axes[0].plot(times, td_no_jump, label=f'No-Jump Traj (idx={idx_no_jump})', color='darkorange', linewidth=1.5, alpha=0.9, linestyle='--')
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"Trace Distance $\mathcal{T}(\rho_{traj}^{(i)}, \rho_L)$")
axes[0].set_title("Single Trajectory Trace Distance vs Lindblad")
axes[0].legend(loc='best')
axes[0].set_ylim(bottom=0)

# Plot Fidelity
axes[1].plot(times, fid_jump, label=f'Jumped Traj (idx={idx_jump})', color='dodgerblue', linewidth=1.5, alpha=0.9)
axes[1].plot(times, fid_no_jump, label=f'No-Jump Traj (idx={idx_no_jump})', color='forestgreen', linewidth=1.5, alpha=0.9, linestyle='--')
axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"Fidelity $\mathcal{F}(\rho_{traj}^{(i)}, \rho_L)$")
axes[1].set_title("Single Trajectory Fidelity vs Lindblad")
axes[1].legend(loc='best')

fig03.suptitle(f'[Angle: {phi_deg}°] Single Trajectory Metrics (Jump vs No-Jump) | dt={dt}', fontsize=16, y=1.02)
filename_single = f"Angle_{int(phi_deg)}_Single_Traj_Comparison_dt{dt_str}"
save_fig(fig03, filename_single)

print("All metric plots generated successfully.")