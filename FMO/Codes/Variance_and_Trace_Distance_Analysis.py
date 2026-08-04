#!/usr/bin/env python
# coding: utf-8
"""
Statistical analysis for FMO quantum trajectories (Variance, M1, Fano, Heatmaps).
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to save files
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit
from scipy.stats import poisson, norm

# Import custom thesis style and saving function
from plot_style import set_thesis_style, save_fig

# Apply global thesis style settings
set_thesis_style()

# Suppress the specific scipy warning about catastrophic cancellation at t=0
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

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
    theta_deg = 90.0  # Default fallback if run manually

# --- Must match the values used in the main simulation script ---
dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/Plot/Statistical_Analysis/{theta_str}"
os.makedirs(Output_dir, exist_ok=True)

fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Theta = {theta_deg} deg")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# Data Extraction
times = data['times']
dt_val = float(data['dt'])
N_site = int(data['N_site'])
eigenergies = data['eigenergies']
eigenvectors = data['eigenvectors']
psi0_exc = data['psi0_exc']

psi_traj_exc = data['psi_traj']         # (N_site, n_times, n_traj), exciton basis, complex64
jump_counts = data['jump_counts']       # (n_times, n_traj) - M1 APPLICATIONS COUNT

# Load density matrices for Trace Distance check
if 'rho_redfield_site' in data and 'rho_traj_avg_site' in data:
    rho_redfield_site = data['rho_redfield_site']
    rho_traj_avg_site = data['rho_traj_avg_site']
else:
    print("Warning: Density matrices (site basis) not found.")
    rho_redfield_site = None
    rho_traj_avg_site = None

n_times = len(times)
n_traj = psi_traj_exc.shape[2]

# ==========================
# Single-trajectory populations (Site & Exciton basis)
# ==========================
psi_traj_site = np.einsum('ia,atk->itk', eigenvectors, psi_traj_exc)   # (N_site, n_times, n_traj)
pop_traj_site = np.abs(psi_traj_site) ** 2                             # (N_site, n_times, n_traj)
pop_traj_exc = np.abs(psi_traj_exc) ** 2                               # (N_site, n_times, n_traj)

# ==========================
# STATISTICAL ANALYSIS: VARIANCE
# ==========================
print("Computing Variance over time...")
var_pop_site_time = np.var(pop_traj_site, axis=2)
var_pop_exc_time = np.var(pop_traj_exc, axis=2)

colors = plt.cm.viridis(np.linspace(0, 1, N_site))

# ==========================================
# PLOT 1a: Variance over Time (Site Basis)
# ==========================================
fig1a, ax1a = plt.subplots(figsize=(10, 6))

for i in range(N_site):
    ax1a.plot(times, var_pop_site_time[i, :], color=colors[i], linewidth=2, label=f'Site {i+1}')

ax1a.set_xlabel('Time (fs)')
ax1a.set_ylabel('Variance')
ax1a.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', ncol=2, title_fontsize=11)

save_fig(fig1a, f'Variance_Time_Sites_Theta_{theta_str}', Output_dir)

# ==========================================
# PLOT 1b: Variance over Time (Exciton Basis)
# ==========================================
fig1b, ax1b = plt.subplots(figsize=(10, 6))

for i in range(N_site):
    ax1b.plot(times, var_pop_exc_time[i, :], color=colors[i], linewidth=2, label=f'Exciton {i+1}')

ax1b.set_xlabel('Time (fs)')
ax1b.set_ylabel('Variance')
ax1b.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', ncol=2, title_fontsize=11)

save_fig(fig1b, f'Variance_Time_Excitons_Theta_{theta_str}', Output_dir)


# ==========================================
# PLOT 2: Trace distance (Redfield vs Avg Trajectories)
# ==========================================
if rho_redfield_site is not None and rho_traj_avg_site is not None:
    td_time = np.zeros(n_times)
    for t in range(n_times):
        td_time[t] = trace_distance_generic_njit(rho_redfield_site[t], rho_traj_avg_site[t])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(times, td_time, color='red', linewidth=2, label='Trace Distance')
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Trace Distance')
    ax2.set_yscale('log') 
    ax2.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
    
    save_fig(fig2, f'Trace_Distance_Theta_{theta_str}', Output_dir)


# ==========================================
# PLOT 3: STATISTICAL DISTRIBUTION OF MEASUREMENT OUTCOMES (M1)
# ==========================================
print("Computing universal statistical distribution of M1 counts...")

n_jumps_total = jump_counts.sum(axis=0)  
mean_jumps = np.mean(n_jumps_total)
var_jumps = np.var(n_jumps_total)

fig4, ax4 = plt.subplots(figsize=(8, 5))

if theta_deg == 0.0:
    max_jumps = int(np.max(n_jumps_total))
    bins = np.arange(-0.5, max_jumps + 1.5, 1) 
    
    ax4.hist(n_jumps_total, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Simulated Jumps (M1)')
    
    k_values = np.arange(0, max_jumps + 1)
    poisson_pmf = poisson.pmf(k_values, mu=mean_jumps)
    
    ax4.plot(k_values, poisson_pmf, 'ro--', markersize=6, linewidth=2, label=f'Poisson Fit ($\\lambda$ = {mean_jumps:.2f})')
    
    if max_jumps < 20:
        ax4.set_xticks(k_values) 

else:
    bins_c = np.linspace(np.min(n_jumps_total), np.max(n_jumps_total), 50)
    
    ax4.hist(n_jumps_total, bins=bins_c, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Simulated Omodyne Clicks (M1)')
    
    mu_gauss, std_gauss = norm.fit(n_jumps_total)
    x_gauss = np.linspace(np.min(n_jumps_total)*0.9, np.max(n_jumps_total)*1.1, 200)
    pdf_gauss = norm.pdf(x_gauss, mu_gauss, std_gauss)
    
    ax4.plot(x_gauss, pdf_gauss, 'g--', linewidth=2.5, label=f'Gaussian Fit\n($\\mu$={mu_gauss:.1f}, $\\sigma$={std_gauss:.1f})')

ax4.set_xlabel('Total Number of $M_1$ Applications')
ax4.set_ylabel('Probability Density')
ax4.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)
ax4.grid(True, alpha=0.3)

save_fig(fig4, f'M1_Counts_Distribution_Theta_{theta_str}', Output_dir)


# ==========================================
# PLOT 4: FANO FACTOR EVOLUTION (Var/Mean) OVER TIME
# ==========================================
print("Computing Fano Factor evolution over time...")

cumulative_jumps = np.cumsum(jump_counts, axis=0) 
mean_t = np.mean(cumulative_jumps, axis=1)
var_t = np.var(cumulative_jumps, axis=1)

fano_t = np.zeros_like(mean_t)
mask = mean_t > 0
fano_t[mask] = var_t[mask] / mean_t[mask]

fano_t[~mask] = 1.0 if theta_deg == 0.0 else 0.5  

fig5, ax5 = plt.subplots(figsize=(8, 5))

ax5.plot(times[mask], fano_t[mask], color='purple', linewidth=2, label='Simulated $\\text{Var}(N)/\\langle N \\rangle$')

if theta_deg == 0.0:
    ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Poisson Theoretical Limit (1.0)')
else:
    ax5.axhline(0.5, color='green', linestyle='--', linewidth=2, label='Binomial Theoretical Limit (0.5)')

ax5.set_xlabel('Time (fs)')
ax5.set_ylabel('Variance / Mean')

if len(fano_t[mask]) > 0:
    y_max = max(1.2, np.max(fano_t[mask])*1.1)
    y_min = min(0.3, np.min(fano_t[mask])*0.9)
    ax5.set_ylim(y_min, y_max)

ax5.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='best', title_fontsize=11)
ax5.grid(True, alpha=0.3)

save_fig(fig5, f'Statistical_Index_Evolution_Theta_{theta_str}', Output_dir)


# ==========================================
# PLOT 5: DENSITY HEATMAP (ONE PLOT PER SITE)
# ==========================================
print("Computing independent Density Heatmaps for each site...")

n_pop_bins = 100
pop_bins = np.linspace(0.0, 1.0, n_pop_bins + 1)

dt_plot = times[1] - times[0]
time_bins = np.append(times, times[-1] + dt_plot)
X_times = np.repeat(times, n_traj)

has_redfield = False
if rho_redfield_site is not None:
    pop_redfield = np.real(np.diagonal(rho_redfield_site, axis1=1, axis2=2))
    has_redfield = True

for i in range(N_site):
    figD, axD = plt.subplots(figsize=(8, 5))
    
    Y_pops = pop_traj_site[i, :, :].flatten()
    
    h, xedges, yedges, im = axD.hist2d(X_times, Y_pops, bins=[time_bins, pop_bins], 
                                      cmap='Blues', norm=LogNorm(), density=False)
    
    if has_redfield:
        axD.plot(times, pop_redfield[:, i], color='red', linewidth=2.5, linestyle='--', 
                 label='Redfield Exact (Mean Path)')
    
    axD.set_xlabel('Time (fs)')
    axD.set_ylabel(f'Population (Site {i+1})')
    axD.set_ylim(0, 1)
    
    cbar = figD.colorbar(im, ax=axD, pad=0.02)
    cbar.set_label('Number of Trajectories')
    
    if has_redfield:
        axD.legend(title=fr"$\theta = {theta_deg}^\circ$", loc='upper right', title_fontsize=11)
        
    save_fig(figD, f'Population_Heatmap_Site_{i+1}_Theta_{theta_str}', Output_dir)

print("Statistical analysis and image saving successfully completed!")