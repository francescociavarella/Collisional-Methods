#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg') # Backend for rendering without a display
import matplotlib.pyplot as plt
import matplotlib.colors as colors # Required for LogNorm

# ==================
# Angles Definition
# ==================

# Parse arguments from Bash script
if len(sys.argv) > 1:
    phi_deg = float(sys.argv[1]) 
    bash_mode = sys.argv[2] if len(sys.argv) > 2 else "unknown" 
else:
    phi_deg = 90.0 
    bash_mode = "local_test"

phi_rad = np.radians(phi_deg)

# Simulation parameters used for file naming
dt = 0.01
N_traj = 10000
dt_str = f"{dt:.6f}".replace(".", "p")
phi_str = f"{phi_rad:.4f}".replace(".", "p")

# --- Results Directory and Output Setup ---
results_dir = "../Results/Data/Complete_rho/"

# Dynamically load the correct data file based on the angle
fname = os.path.join(results_dir, f"result_phi{phi_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Angle = {phi_deg}°")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# Create a specific subfolder for the current angle (e.g., Plot/Populations/180)
Output_dir = f"../Results/Plot/Populations/{int(phi_deg)}"
os.makedirs(Output_dir, exist_ok=True)


# ===========================
# General Setup for Plotting
# ===========================

# Global Style Settings (Matplotlib rcParams)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'figure.autolayout': True, 
    'axes.formatter.useoffset': False 
})

def save_fig(fig, filename):
    """Saves the figure cleanly in the dynamically created output directory"""
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)


# ================
# DATA EXTRACTION
# ================

# Extract time array
times = data['times']

# Extract total jumps if available
total_jumps = data['total_jumps'] if 'total_jumps' in data else None

# Extract full state population data from rho_tot_all (3, 3, n_times, N_traj)
rho_all = data['rho_tot_all']
pop_00 = np.real(rho_all[0, 0, :, :])
pop_11 = np.real(rho_all[1, 1, :, :])
pop_22 = np.real(rho_all[2, 2, :, :])

# Extract Lindblad data for reference if needed
rho_lind = data['rho_list_lindblad']
lindblad_00 = np.real(rho_lind[:, 0, 0])
lindblad_11 = np.real(rho_lind[:, 1, 1])
lindblad_22 = np.real(rho_lind[:, 2, 2])

print("Data and individual populations loaded successfully!")


# ===============================
# POPULATIONS VARIANCE OVER TIME
# ===============================

plt.close('all')

# Calculate the Variance across all trajectories (axis=1)
var_00 = np.var(pop_00, axis=1)
var_11 = np.var(pop_11, axis=1)
var_22 = np.var(pop_22, axis=1)

# Create the figure
fig_var, ax = plt.subplots(figsize=(10, 5))

# Plot the variance for each level
ax.plot(times, var_00, label=r'Variance $|0\rangle$', color='royalblue', linewidth=2)
ax.plot(times, var_11, label=r'Variance $|1\rangle$', color='forestgreen', linewidth=2)
ax.plot(times, var_22, label=r'Variance $|2\rangle$', color='darkorange', linewidth=2)

# Formatting
ax.set_xlabel('Time')
ax.set_ylabel(r'Variance $\sigma^2$')
ax.set_title(f'Population Variance over Time (Angle: {phi_deg}°)')
ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best')

# Global figure settings
fig_var.suptitle(f'Statistical Spread of Populations | dt={dt}, N_traj={N_traj}', fontsize=14)

# Save the figure
filename_var = f"Angle_{int(phi_deg)}_Populations_Variance_dt{dt_str}"
save_fig(fig_var, filename_var)


# =================================================================
# HEATMAP DISTRIBUTION OF POPULATIONS (LOG COLORMAPPING)
# =================================================================

# n_traj = pop_00.shape[1]
# n_times = len(times)

# # Heatmap Bin Parameters
# n_bins = 150 
# bins_array = np.linspace(0.0, 1.0, n_bins + 1)

# # Initialize heatmap matrices
# heatmap_00 = np.zeros((n_bins, n_times))
# heatmap_11 = np.zeros((n_bins, n_times))
# heatmap_22 = np.zeros((n_bins, n_times))

# # Compute the histogram for each time step
# for t in range(n_times):
#     counts_00, _ = np.histogram(pop_00[t, :], bins=bins_array)
#     heatmap_00[:, t] = counts_00
    
#     counts_11, _ = np.histogram(pop_11[t, :], bins=bins_array)
#     heatmap_11[:, t] = counts_11
    
#     counts_22, _ = np.histogram(pop_22[t, :], bins=bins_array)
#     heatmap_22[:, t] = counts_22

# # IMPORTANT: Keep masking zeros. LogNorm hates exact zeros.
# hm_masked_00 = np.ma.masked_where(heatmap_00 == 0, heatmap_00)
# hm_masked_11 = np.ma.masked_where(heatmap_11 == 0, heatmap_11)
# hm_masked_22 = np.ma.masked_where(heatmap_22 == 0, heatmap_22)

# # Create the Figure (1 Row, 3 Columns)
# fig_hm, axes = plt.subplots(1, 3, figsize=(20, 5))

# # Configuration for the loop
# heatmap_configs = [
#     (0, hm_masked_00, r'Population $|0\rangle$', 'Blues'),
#     (1, hm_masked_11, r'Population $|1\rangle$', 'Greens'),
#     (2, hm_masked_22, r'Population $|2\rangle$', 'Oranges')
# ]

# for idx, hm_masked, title, cmap in heatmap_configs:
#     ax = axes[idx]
    
#     # Render Heatmap with LogNorm
#     im = ax.imshow(
#         hm_masked,
#         aspect='auto',
#         origin='lower',
#         extent=[times[0], times[-1], 0.0, 1.0], 
#         cmap=cmap,
#         interpolation='nearest',
#         norm=colors.LogNorm(vmin=1, vmax=n_traj) 
#     )
    
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Population')
#     ax.set_title(title)
    
#     # Add colorbar 
#     cbar = fig_hm.colorbar(im, ax=ax)
#     cbar.set_label('Trajectory Count ($\log_{10}$)')

# # Global Formatting
# fig_hm.suptitle(f'[Angle: {phi_deg}°] Population Distributions over Time | dt={dt}', fontsize=16, y=1.05)

# # Save the figure
# filename_hm_pop = f"Angle_{int(phi_deg)}_Populations_LogHeatmap_dt{dt_str}"
# save_fig(fig_hm, filename_hm_pop)

# print(f"All plots for Angle = {phi_deg}° generated and saved.")