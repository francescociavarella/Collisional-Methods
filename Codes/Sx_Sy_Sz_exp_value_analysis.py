#!/usr/bin/env python
# coding: utf-8

# Notebook to calculate the distrubution in time, the Variance and the fitting of the average over different number of N_trajectoreis, all for one single angle theta

# In[1]:


import numpy as np
from qutip import *
from scipy.linalg import sqrtm, eigvalsh
from scipy.stats import linregress
from numba import njit, prange
import pickle
import os
import gc


# In[2]:


get_ipython().run_line_magic('matplotlib', 'ipympl')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import Image, display, Math


# In[3]:


# ==================================================
# NUMBA OPTIMIZED LOOP FOR PAULI EXPECTATION VALUES
# ==================================================
@njit(parallel=True)
def compute_pauli_expectations_all_trajectories(pop_10, coh_1001, coh_0110, pop_01):
    """
    Computes the expectation values of sigma_x, sigma_y, and sigma_z 
    for ALL individual trajectories.
    
    Inputs:
        2D NumPy arrays: (time_steps, N_traj) containing the density matrix elements.
        
    Returns:
        sigma_x_matrix, sigma_y_matrix, sigma_z_matrix: 2D arrays (time_steps, N_traj)
    """
    time_steps = pop_10.shape[0]
    N_traj = pop_10.shape[1]
    
    # Pre-allocate output 2D arrays (time_steps, N_traj)
    sigma_x_matrix = np.zeros((time_steps, N_traj))
    sigma_y_matrix = np.zeros((time_steps, N_traj))
    sigma_z_matrix = np.zeros((time_steps, N_traj))
    
    # Outer loop over time
    for t in range(time_steps):
        # Parallel loop over all trajectories (using all CPU cores)
        for n in prange(N_traj):
            
            # <sigma_z> = rho_00 - rho_11
            sigma_z_matrix[t, n] = pop_10[t, n] - pop_01[t, n]
            
            # <sigma_x> = rho_01 + rho_10
            # .real is used to ensure the output is strictly a real float
            sigma_x_matrix[t, n] = (coh_0110[t, n] + coh_1001[t, n]).real
            
            # <sigma_y> = i * (rho_01 - rho_10)
            # 1j is the imaginary unit in Python
            sigma_y_matrix[t, n] = (1j * (coh_0110[t, n] - coh_1001[t, n])).real
            
    return sigma_x_matrix, sigma_y_matrix, sigma_z_matrix


# ## General Setup

# In[4]:


# ====================================
# Physical & Simulation Parameters
# ====================================
# Theta angle in degrees, H_Coll Direction
theta_target_deg = 60.0  # change angle here
theta_rad = np.radians(theta_target_deg)

# Site selector: 0 for |10>, 1 for |01>
site_index = 0

# Time step
dt = 0.01

# Number of trajectories to analyze
N_traj = 10000          


# In[ ]:


# ===========================
# General Setup for Plotting
# ===========================
# --- 1. Output Directory---

if theta_target_deg.is_integer():
    angle_folder = str(int(theta_target_deg))
else:
    angle_folder = str(theta_target_deg)

Output_dir = os.path.join("../Results/Plot/Sxyz_exp_value/Complete", angle_folder)
os.makedirs(Output_dir, exist_ok=True)

# --- 2. Global Style Settings (Matplotlib rcParams) ---
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'figure.autolayout': True # plt.tight_layout()
})

# --- 3. Automatic Figure Saving Function ---
def save_fig(fig, filename):
    """
    Saves the figure in both PNG or PDF
    """
    path_png = os.path.join(Output_dir, f"{filename}.png")
    # path_pdf = os.path.join(Output_dir, f"{filename}.pdf")  # save in pdf
    
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    # fig.savefig(path_pdf, bbox_inches='tight') # save in pdf
    print(f"Figure saved in: {Output_dir}/{filename}")


# In[6]:


# =================
# Input Data Setup
# =================
Input_dir = "../Results/Data/Complete_rho/normal"  # <-- change here if needed

# Format theta and dt for filename 
theta_str = f"{theta_rad:.6f}".replace(".", "p")
dt_str = f"{dt:.6f}".replace(".", "p")

# File name
filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj20000.npz"
filepath = os.path.join(Input_dir, filename)

print(f"Analisi impostata per theta = {theta_target_deg}°")
print(f"File target: {filename}")


#  ### Data Extraction

# In[7]:


if not os.path.exists(filepath):
    print(f"ERRORE: The file {filepath} doesn't exist. Check file name.")
else:
    # Load .npz input containing data
    data = np.load(filepath)
    
    times = data['times']
    n_times = len(times)
    
    print("Matrix extraction in progress")
    
    # =======================
    # 1. Lindblad Extraction
    # =======================
    rho_lindblad_complete = data['rho_list_lindblad']  # 4x4 dimesion
    
    # Populations : Index (2,2) -> |10><10|, Index (1,1)  -> |01><01|    
    pop_lindblad_10 = np.real(rho_lindblad_complete[:, 2, 2])
    pop_lindblad_01 = np.real(rho_lindblad_complete[:, 1, 1])
    
    # Coherences: Index (2,1) -> |10><01|, Index (1, 2) -> |01><10|  ATTENTION : inverted respect to Trajectories, already inverted here
    cohe_lindblad_10_01 = rho_lindblad_complete[:, 1, 2] 
    cohe_lindblad_01_10 = rho_lindblad_complete[:, 2, 1]

    rho_lindblad = np.zeros((n_times, 2, 2), dtype=np.complex128)

    for t in range(n_times):
    
        # Populations : Index (2,2) -> |10><10|, Index (1,1)  -> |01><01| INVERTED respect to Trajectories, already inverted here
        rho_lindblad[t, 0, 0] = rho_lindblad_complete[t, 2, 2]  # |10><10|
        rho_lindblad[t, 1, 1] = rho_lindblad_complete[t, 1, 1]  # |01><01|
        
        # Coherences ATTENTION : inverted respect to Trajectories, already inverted here
        rho_lindblad[t, 0, 1] = rho_lindblad_complete[t, 2, 1]  # |10><01|
        rho_lindblad[t, 1, 0] = rho_lindblad_complete[t, 1, 2]  # |01><10|
        
    # ================================
    # 2.  Raw Trajectories Extraction
    # ================================
    
    pop_traj_10 = data['pop_00']
    pop_traj_01 = data['pop_11']
    
    cohe_traj_10_01 = data['coh_10_01'] 
    cohe_traj_01_10 = data['coh_01_10']

    print("Data extraction completed")


# ## Sx Sy Sz expectation value

# In[8]:


# ==========================================================================
# COMPUTE PAULI EXPECTATION VALUES (ALL TRAJECTORIES)
# ==========================================================================
print("Starting complete calculation for Pauli matrices on all trajectories...")

all_sigma_x, all_sigma_y, all_sigma_z = compute_pauli_expectations_all_trajectories(
    pop_traj_10, cohe_traj_10_01, cohe_traj_01_10, pop_traj_01
)

print(f"Calculation finished. Shape of arrays: {all_sigma_z.shape}")


# In[ ]:


# ==============================================================
# PLOT HEATMAP COMPLETE: PAULI EXPECTATION VALUES (Sx, Sy, Sz)
# ==============================================================
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# Ensure we have the correct dimensions from the previously calculated arrays
# The Numba function returns arrays of shape (time_steps, n_traj)
n_times, n_traj = all_sigma_x.shape

# Heatmap bin parameters (Pauli values range from -1 to 1)
n_bins = 150 
pauli_bins = np.linspace(-1.0, 1.0, n_bins + 1)

# List of the matrices and their corresponding labels for the loop
matrices = [all_sigma_x, all_sigma_y, all_sigma_z]
labels = [r'\sigma_x', r'\sigma_y', r'\sigma_z']
filenames = ['Sigma_X', 'Sigma_Y', 'Sigma_Z']

# Loop over the 3 expectation values to generate 3 separate plots
for sigma_matrix, label, file_suffix in zip(matrices, labels, filenames):
    
    # Initialize an empty heatmap for the current Pauli matrix
    heatmap_complete = np.zeros((n_bins, n_times))
    
    # Compute the histogram for each time step
    for t in range(n_times):
        # Extract all trajectories at time 't'
        counts, _ = np.histogram(sigma_matrix[t, :], bins=pauli_bins)
        heatmap_complete[:, t] = counts

    # Create the figure with a wider figsize, just like the Fidelity Complete plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Mask zero counts to improve visualization (transparent background)
    heatmap_masked_complete = np.ma.masked_where(heatmap_complete == 0, heatmap_complete) 

    # Plot the heatmap
    im = ax.imshow(
        heatmap_masked_complete,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], -1.0, 1.0], # Y-axis goes from -1 to 1
        cmap='viridis',                          # Kept 'viridis' as requested
        interpolation='nearest',
        vmin=1, vmax=np.max(heatmap_complete)*0.8
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of trajectories')

    # Apply labels and dynamic title
    ax.set_xlabel('Time steps')
    ax.set_ylabel(f'Expectation Value $\\langle {label} \\rangle$')
    ax.set_title(f'Distribution of $\\langle {label} \\rangle$ over Time | $\\theta$ = {theta_target_deg}°')

    plt.show()


# In[ ]:


# ==============================================================
# PLOT HEATMAP COMPLETE: PAULI EXPECTATION VALUES (LOG SCALE)
# ==============================================================

plt.close('all')

n_times, n_traj = all_sigma_x.shape
n_bins = 150 
pauli_bins = np.linspace(-1.0, 1.0, n_bins + 1)

matrices = [all_sigma_x, all_sigma_y, all_sigma_z]
labels = [r'\sigma_x', r'\sigma_y', r'\sigma_z']

for sigma_matrix, label in zip(matrices, labels):
    
    heatmap_complete = np.zeros((n_bins, n_times))
    
    for t in range(n_times):
        counts, _ = np.histogram(sigma_matrix[t, :], bins=pauli_bins)
        # Normalize to fraction of trajectories
        heatmap_complete[:, t] = counts / n_traj

    fig, ax = plt.subplots(figsize=(12, 5))

    heatmap_masked = np.ma.masked_where(heatmap_complete == 0, heatmap_complete) 
    vmin_val = 1.0 / n_traj  # Minimum observable fraction

    # Plot using LogNorm and magma to see the fine details!
    im = ax.imshow(
        heatmap_masked,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], -1.0, 1.0], 
        cmap='viridis', 
        norm=LogNorm(vmin=vmin_val, vmax=1.0),
        interpolation='nearest'
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Fraction of trajectories (Log Scale)')

    ax.set_xlabel('Time steps')
    ax.set_ylabel(f'Expectation Value $\\langle {label} \\rangle$')
    ax.set_title(f'Distribution of $\\langle {label} \\rangle$ over Time | $\\theta$ = {theta_target_deg}°')

    # Automatically save the heatmap figure (uncomment when needed)
    filename_heatmap = f"Heatmap_{file_suffix}_Complete_Theta_{theta_str}_dt_{dt_str}"
    save_fig(fig, filename_heatmap)

    plt.show()


# In[ ]:


# ======================================================
# PLOT EXPECTATION VALUES: AVG TRAJECTORIES VS LINDBLAD
# ======================================================

plt.close('all')

# 1. Calculate the exact Lindblad expectation values
# <sigma_z> = rho_00 - rho_11
sz_lindblad = pop_lindblad_10 - pop_lindblad_01

# <sigma_x> = rho_01 + rho_10
sx_lindblad = np.real(cohe_lindblad_01_10 + cohe_lindblad_10_01)

# <sigma_y> = i * (rho_01 - rho_10)
sy_lindblad = np.real(1j * (cohe_lindblad_01_10 - cohe_lindblad_10_01))

# 2. Group data for automated looping
lindblad_expectations = [sx_lindblad, sy_lindblad, sz_lindblad]
trajectory_matrices = [all_sigma_x, all_sigma_y, all_sigma_z]
labels = [r'\sigma_x', r'\sigma_y', r'\sigma_z']
filenames = ['Sigma_X', 'Sigma_Y', 'Sigma_Z']

# List of trajectory counts to average over
N_list = [100, 1000, 10000]

# 3. Loop over the 3 expectation values
for traj_matrix, lind_array, label, file_suffix in zip(trajectory_matrices, lindblad_expectations, labels, filenames):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the average over N trajectories
    for N in N_list:
        # Calculate the mean over the first N trajectories along the columns (axis=1)
        avg_traj = np.mean(traj_matrix[:, :N], axis=1)
        
        # Dynamic styling: make the curve for 10000 trajectories fully opaque and thicker
        alpha_val = 0.6 if N < 10000 else 1.0
        lw = 1.5 if N < 10000 else 2.5
        
        ax.plot(times, avg_traj, label=f'Avg N = {N}', alpha=alpha_val, linewidth=lw)
    
    # Plot the exact Lindblad solution as a dashed black line for comparison
    ax.plot(times, lind_array, label='Lindblad (Exact)', color='black', linestyle='--', linewidth=2.0)
    
    # Apply labels, legend and dynamic title
    ax.set_xlabel("Time steps")
    ax.set_ylabel(f"Expectation Value $\\langle {label} \\rangle$")
    ax.set_title(f"Time Evolution of $\\langle {label} \\rangle$ | $\\theta$ = {theta_target_deg}°")
    ax.legend()
    ax.grid(alpha=0.4, linestyle='--')
    fig.tight_layout()
    
    # Automatically save the figure (uncomment when you want to save)
    filename_avg = f"Avg_{file_suffix}_Evolution_Theta_{theta_target_deg}deg"
    save_fig(fig, filename_avg)
    
    plt.show()


# In[ ]:


# =================================================
# VARIANCE CALCULATION AND PLOT FOR PAULI MATRICES
# =================================================

print("Calculating variances across all trajectories")

# 1. Calculate the variance for each time step across all trajectories (axis=1)
variance_sx = np.var(all_sigma_x, axis=1)
variance_sy = np.var(all_sigma_y, axis=1)
variance_sz = np.var(all_sigma_z, axis=1)

# 2. Plotting the variances together for comparison
plt.close('all')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the three variances
ax.plot(times, variance_sx, label=r'Variance of $\langle \sigma_x \rangle$', color='dodgerblue', linewidth=2)
ax.plot(times, variance_sy, label=r'Variance of $\langle \sigma_y \rangle$', color='mediumseagreen', linewidth=2, alpha=0.8)
ax.plot(times, variance_sz, label=r'Variance of $\langle \sigma_z \rangle$', color='crimson', linewidth=2)

# Set labels and dynamic titles
ax.set_xlabel("Time steps")
ax.set_ylabel("Variance")
ax.set_title(f"Variance of Pauli Expectation Values over Time | $\\theta$ = {theta_target_deg}°")
ax.legend()
ax.grid(alpha=0.4, linestyle='--')

fig.tight_layout()

# Automatically save the figure 
filename_var_pauli = f"Variance_Pauli_Theta_{theta_str}_dt_{dt_str}"
save_fig(fig, filename_var_pauli)

plt.show()


# In[ ]:


# =============================
# SAVE ALL STATISTICAL METRICS 
# =============================

# 1. Define the subdirectory name and create the path
subdir = "metrics"
metrics_dir = os.path.join(Input_dir, subdir)

# 2. Check if the directory exists, if not, create it
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)
    print(f"Created directory: {metrics_dir}")

# 3. Define a UNIQUE output filepath to avoid overwriting Fidelity data
output_filename = f"Metrics_Pauli_Theta_{theta_target_deg}deg.npz"
output_filepath = os.path.join(metrics_dir, output_filename)

# 4. Save everything into a compressed .npz file
np.savez_compressed(
    output_filepath,
    times=times,               
    var_sx=variance_sx,
    var_sy=variance_sy,
    var_sz=variance_sz,
    mean_sx=mean_sx,
    mean_sy=mean_sy,
    mean_sz=mean_sz
)

print(f"Pauli metrics successfully saved in: {output_filepath}")

