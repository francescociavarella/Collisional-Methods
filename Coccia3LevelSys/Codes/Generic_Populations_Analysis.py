#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from qutip import *
from scipy.linalg import sqrtm, eigvalsh
from scipy.stats import linregress
from numba import njit, prange
import pickle
import os
import time

get_ipython().run_line_magic('matplotlib', 'ipympl')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import Image, display, Math


# In[ ]:


# ====================================
# Physical & Simulation Parameters
# ====================================

# Time parameters
dt = 0.01
tf = 100.0
time_steps = int(tf / dt)    


# In[ ]:


# ============================================================
# GLOBAL CONFIGURATION :    
#   'QJ' -> Quantum Jump
#   'SD' -> State Diffusion
# ============================================================

MODE = 'QJ'   # Switch this to 'SD' or 'QJ'

# Optional: define parameters if you want to use them in titles/filenames later
dt = 0.01
N_traj = 10000
dt_str = f"{dt:.6f}".replace(".", "p")

# Configuration mapping based on MODE
_cfg = {
    'QJ': {
        'Input_file': f"../Results/Data/Complete_rho/result_mode_QJ_dt{dt_str}_Ntraj{N_traj}.npz",
        'Output_dir': "../Results/Plot/Populations/QJ" 
    },
    'SD': {
        'Input_file': f"../Results/Data/Complete_rho/result_mode_SD_dt{dt_str}_Ntraj{N_traj}.npz",
        'Output_dir': "../Results/Plot/Populations/SD"
    },
}

# Apply the selected configuration
cfg = _cfg[MODE]
Input_file = cfg['Input_file']
Output_dir = cfg['Output_dir']

# Create the output directory if it doesn't exist
os.makedirs(Output_dir, exist_ok=True)

print(f"--- Configuration Setup ---")
print(f"Current mode : {MODE}")
print(f"Input Data   : {Input_file}")
print(f"Output Plots : {Output_dir}")
print(f"---------------------------\n")


# In[ ]:


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
    'figure.autolayout': True, # Equivalent to plt.tight_layout()
    
    # Disable the automatic global offset on axes (e.g., +1e0)
    'axes.formatter.useoffset': False 
})

# Automatic Figure Saving Function
def save_fig(fig, filename):
    """
    Saves the figure in both PNG or PDF
    """
    path_png = os.path.join(Output_dir, f"{filename}.png")
    # path_pdf = os.path.join(Output_dir, f"{filename}.pdf")  # save in pdf
    
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    # fig.savefig(path_pdf, bbox_inches='tight') # save in pdf
    print(f"Figure saved in: {Output_dir}/{filename}")


#  ### Data Extraction

# In[ ]:


# ================
# DATA EXTRACTION
# ================

# 1. Load the file using numpy
data = np.load(Input_file)

# 2. Extract arrays
times = data['times']

if 'total_jumps' in data:
    total_jumps = data['total_jumps']
else:
    total_jumps = None

# -----------------------------------------------
# Extract from rho_tot_all (3, 3, n_times, N_traj)
# -----------------------------------------------
rho_all = data['rho_tot_all']

# Extract Individual Populations (Real part of diagonal elements)
# Shape will be (n_times, N_traj)
pop_00_all = np.real(rho_all[0, 0, :, :])
pop_11_all = np.real(rho_all[1, 1, :, :])
pop_22_all = np.real(rho_all[2, 2, :, :])

# ----------------------------------------------------
# Extract Lindblad: rho_list_lindblad (n_times, 3, 3)
# ----------------------------------------------------
rho_lind = data['rho_list_lindblad']

# Extract Lindblad Populations (Real part)
# Shape will be (n_times,)
lind_00 = np.real(rho_lind[:, 0, 0])
lind_11 = np.real(rho_lind[:, 1, 1])
lind_22 = np.real(rho_lind[:, 2, 2])

print("Data and individual populations loaded successfully!")


# ## Plot

# In[ ]:


# =================================================================
# POPULATIONS VARIANCE OVER TIME
# =================================================================

plt.close('all')

# Calculate the Variance across all trajectories (axis=1)
var_00 = np.var(pop_00_all, axis=1)
var_11 = np.var(pop_11_all, axis=1)
var_22 = np.var(pop_22_all, axis=1)

# Create the figure
fig_var, ax = plt.subplots(figsize=(10, 5))

# Plot the variance for each level
ax.plot(times, var_00, label=r'Variance $|0\rangle$', color='royalblue', linewidth=2)
ax.plot(times, var_11, label=r'Variance $|1\rangle$', color='forestgreen', linewidth=2)
ax.plot(times, var_22, label=r'Variance $|2\rangle$', color='darkorange', linewidth=2)

# Formatting
ax.set_xlabel('Time')
ax.set_ylabel('Variance $\sigma^2$')
ax.set_title(f'Population Variance over Time ({MODE})')
ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best')

# Global figure settings
fig_var.suptitle(f'Statistical Spread of Populations | dt={dt}, N_traj={N_traj}', fontsize=14)
plt.tight_layout()

# Save the figure
filename_var = f"{MODE}_Populations_Pure_Variance_dt{dt_str}"
save_fig(fig_var, filename_var)

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 

# =================================================================
# HEATMAP DISTRIBUTION OF POPULATIONS (LOG COLORMAPPING)
# =================================================================

plt.close('all')

n_traj = pop_00_all.shape[1]
n_times = len(times)

# Heatmap Bin Parameters
n_bins = 150 
bins_array = np.linspace(0.0, 1.0, n_bins + 1)

# Initialize heatmap matrices
heatmap_00 = np.zeros((n_bins, n_times))
heatmap_11 = np.zeros((n_bins, n_times))
heatmap_22 = np.zeros((n_bins, n_times))

# Compute the histogram for each time step
for t in range(n_times):
    counts_00, _ = np.histogram(pop_00_all[t, :], bins=bins_array)
    heatmap_00[:, t] = counts_00
    
    counts_11, _ = np.histogram(pop_11_all[t, :], bins=bins_array)
    heatmap_11[:, t] = counts_11
    
    counts_22, _ = np.histogram(pop_22_all[t, :], bins=bins_array)
    heatmap_22[:, t] = counts_22

# IMPORTANT: Keep masking zeros. LogNorm hates exact zeros.
hm_masked_00 = np.ma.masked_where(heatmap_00 == 0, heatmap_00)
hm_masked_11 = np.ma.masked_where(heatmap_11 == 0, heatmap_11)
hm_masked_22 = np.ma.masked_where(heatmap_22 == 0, heatmap_22)

# Create the Figure (1 Row, 3 Columns)
fig_hm, axes = plt.subplots(1, 3, figsize=(20, 5))

# Configuration for the loop
heatmap_configs = [
    (0, hm_masked_00, r'Population $|0\rangle$', 'Blues'),
    (1, hm_masked_11, r'Population $|1\rangle$', 'Greens'),
    (2, hm_masked_22, r'Population $|2\rangle$', 'Oranges')
]

for idx, hm_masked, title, cmap in heatmap_configs:
    ax = axes[idx]
    
    # --- FIXED: Removed separate vmin/vmax, kept only inside LogNorm ---
    im = ax.imshow(
        hm_masked,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0.0, 1.0], 
        cmap=cmap,
        interpolation='nearest',
        norm=colors.LogNorm(vmin=1, vmax=n_traj) 
    )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title(title)
    
    # Add colorbar - it will automatically show log labels (10^0, 10^1, etc.)
    cbar = fig_hm.colorbar(im, ax=ax)
    cbar.set_label('Trajectory Count ($\log_{10}$)')

# Global Formatting
fig_hm.suptitle(f'[{MODE}] Population Distributions over Time (Log Counts) | dt={dt}', fontsize=16, y=1.05)
plt.tight_layout()

# Save the figure dynamically
filename_hm_pop = f"{MODE}_Populations_LogHeatmap_dt{dt_str}"
save_fig(fig_hm, filename_hm_pop)

plt.show()


# In[ ]:




