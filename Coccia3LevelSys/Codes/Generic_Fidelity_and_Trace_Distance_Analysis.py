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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'ipympl')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import Image, display, Math


# ## Fidelity
# Generic definition : 
# $$ \mathcal{F}\left( \rho, \sigma \right) = \left( Tr \left[ \sqrt{ \sqrt{\rho} \sigma \sqrt{\rho} }\right] \right)^{2} $$ 
# Definition for $ \rho $ Pure State and $ \sigma $ Mixed State : 
# $$ \mathcal{F}\left( \rho, \sigma \right) = \langle \psi_{\sigma} | \sigma | \psi_{\sigma} \rangle $$
# Definition for Pure State : 
# $$ \mathcal{F}\left( \rho, \sigma \right) = |\langle \psi_{\rho} | \psi_{\sigma} \rangle|^{2} $$
# Definition for Qubits : 
# $$ \mathcal{F}\left( \rho, \sigma \right) = Tr\left( \rho \, \sigma \right) + 2 \sqrt{Det\left ( \rho \right) \, Det\left ( \sigma \right)} $$
# ## Trace Distance
# Generic definition : 
# $$ \mathcal{T}\left( \rho, \sigma \right) = \frac{1}{2} Tr \left[ \sqrt{\left( \rho - \sigma \right)^{\dagger} \left( \rho - \sigma  \right)} \right] $$
# ### Relationship : Fuchs-van de Graaf inequality
# $$ 1 - \sqrt{\mathcal{F}\left( \rho, \sigma \right)} \leq \mathcal{T}\left( \rho, \sigma \right) \leq \sqrt{1 - \mathcal{F}\left( \rho, \sigma \right)} $$

# In[ ]:


def fidelity_generic(rho, sigma):
    """
    Calculate the quantum fidelity between two generic density matrices.
    Formula: F(rho, sigma) = ( Tr[ sqrt( sqrt(rho) * sigma * sqrt(rho) ) ] )^2
    
    This version avoids scipy.linalg.sqrtm to prevent RuntimeWarnings, 
    using stable eigenvalue decomposition instead.
    
    Parameters:
        rho (numpy.ndarray): First density matrix (NxN).
        sigma (numpy.ndarray): Second density matrix (NxN).
        
    Returns:
        float: The fidelity between rho and sigma (real number between 0 and 1).
    """
    # 1. Square root of rho using eigenvalue decomposition
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    # Truncate any negative noise to 0.0 before taking the square root
    evals_rho = np.maximum(evals_rho, 0.0) 
    sqrt_rho = evecs_rho @ np.diag(np.sqrt(evals_rho)) @ evecs_rho.conj().T
    
    # 2. Inner matrix: sqrt(rho) * sigma * sqrt(rho)
    inner_matrix = sqrt_rho @ sigma @ sqrt_rho
    
    # Force exact Hermiticity to remove any small imaginary noise
    inner_matrix = 0.5 * (inner_matrix + inner_matrix.conj().T)
    
    # 3. Trace of the square root is the sum of the square roots of the eigenvalues
    evals_inner = eigvalsh(inner_matrix)
    # Again, truncate negative noise to 0.0 before square root
    evals_inner = np.maximum(evals_inner, 0.0)
    
    fidelity = np.sum(np.sqrt(evals_inner))**2
    
    # Ensure numerical errors do not push fidelity slightly above 1.0
    return min(1.0, fidelity)
    


# In[ ]:


@njit
def fidelity_generic_njit(rho, sigma):
    """Numba-compatible generalized fidelity."""
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho, 0.0) 
    
    # FIX: Explicitly cast the float64 diagonal matrix to complex128
    diag_matrix = np.diag(np.sqrt(evals_rho)).astype(np.complex128)
    
    # Now all matrices are complex128, Numba's @ operator will work perfectly
    sqrt_rho = evecs_rho @ diag_matrix @ evecs_rho.conj().T
    
    # Ensure sigma is also treated as complex128 just to be safe
    inner_matrix = sqrt_rho @ sigma.astype(np.complex128) @ sqrt_rho
    inner_matrix = 0.5 * (inner_matrix + inner_matrix.conj().T)
    
    evals_inner = np.linalg.eigvalsh(inner_matrix)
    evals_inner = np.maximum(evals_inner, 0.0)
    
    fidelity = np.sum(np.sqrt(evals_inner))**2
    return min(1.0, fidelity)


# In[ ]:


@njit
def fidelity_qubit(rho, sigma):
    """
    Calculate the exact quantum fidelity between two single-qubit (2x2) density matrices.
    Formula: F(rho, sigma) = Tr(rho * sigma) + 2 * sqrt(Det(rho) * Det(sigma))
    """
    # Trace of the matrix product
    tr_term = np.real(np.trace(rho @ sigma))
    
    # Determinants of the two density matrices
    det_rho = np.real(np.linalg.det(rho))
    det_sigma = np.real(np.linalg.det(sigma))
    
    # FIX NUMERICO: Tronchiamo a 0 gli eventuali valori negativi infinitesimi
    det_rho = max(0.0, det_rho)
    det_sigma = max(0.0, det_sigma)
    
    # Calculate fidelity using the analytical formula for qubits
    fidelity = tr_term + 2.0 * np.sqrt(det_rho * det_sigma)
    
    return fidelity


# In[ ]:


def trace_distance_generic(rho, sigma):
    """
    Calculate the Trace Distance between two generic density matrices.
    Formula: T(rho, sigma) = 1/2 * Tr[ sqrt( (rho - sigma)^dagger * (rho - sigma) ) ]
    
    Parameters:
        rho (numpy.ndarray): First density matrix (NxN).
        sigma (numpy.ndarray): Second density matrix (NxN).
        
    Returns:
        float: The trace distance between rho and sigma (real number between 0 and 1).
    """
    # Difference between the matrices
    diff = rho - sigma
    
    # Force exact Hermiticity to remove numerical noise
    diff = 0.5 * (diff + diff.conj().T)
    
    # Calculate the eigenvalues of the strictly Hermitian matrix 'diff'
    eigenvalues = eigvalsh(diff)
    
    # Trace distance is half the sum of the absolute eigenvalues
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    
    # Ensure it stays within physical bounds
    return min(1.0, t_dist)
    


# In[ ]:


@njit
def trace_distance_generic_njit(rho, sigma):
    """Numba-compatible generalized trace distance."""
    diff = rho - sigma
    diff = 0.5 * (diff + diff.conj().T)
    
    eigenvalues = np.linalg.eigvalsh(diff)
    t_dist = 0.5 * np.sum(np.abs(eigenvalues))
    return min(1.0, t_dist)


# In[ ]:


def trace_distance_qubit(rho, sigma):
    """
    Calculate the exact Trace Distance between two single-qubit (2x2) density matrices.
    For a 2x2 traceless Hermitian matrix (rho - sigma), Det(diff) = -lambda^2 <= 0.
    Therefore, the Trace Distance simplifies to sqrt(-Det(rho - sigma)).
    
    Parameters:
        rho (numpy.ndarray): First density matrix (2x2).
        sigma (numpy.ndarray): Second density matrix (2x2).
        
    Returns:
        float: The trace distance between rho and sigma.
    """
    # Difference between the matrices
    diff = rho - sigma
    
    # Determinant of the difference
    det_diff = np.real(np.linalg.det(diff))
    
    # Since det_diff should be <= 0, -det_diff should be >= 0.
    # We use max(0.0, ...) to truncate any negative noise before applying sqrt.
    val_under_sqrt = max(0.0, -det_diff)
    
    t_dist = np.sqrt(val_under_sqrt)
    
    return min(1.0, t_dist)
    


# In[ ]:


@njit
def fidelity_qubit_single_term(p00, p11, c10, c01, sigma):
    """
    Build the density matrix from its elements and then
    Calculate the exact quantum fidelity between two single-qubit (2x2) density matrices.
    Formula: F(rho, sigma) = Tr(rho * sigma) + 2 * sqrt(Det(rho) * Det(sigma))
    """
    # 1. Density Matrix build up 
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0,0] = p00
    rho[0,1] = c01
    rho[1,0] = c10
    rho[1,1] = p11

    # 2. Trace of the matrix product (Calculated explicitly for 2x2 to avoid Numba issues)
    prod = rho @ sigma
    tr_term = prod[0,0].real + prod[1,1].real
    
    # 3. Determinants of the two 2x2 density matrices (ad - bc)
    det_rho = (rho[0,0] * rho[1,1] - rho[0,1] * rho[1,0]).real
    det_sigma = (sigma[0,0] * sigma[1,1] - sigma[0,1] * sigma[1,0]).real
    
    # 4. Numerical FIX: Truncate to 0 any infinitesimal negative values
    det_rho = max(0.0, det_rho)
    det_sigma = max(0.0, det_sigma)
    
    # 5. Calculate fidelity using the analytical formula for qubits
    fidelity = tr_term + 2.0 * np.sqrt(det_rho * det_sigma)
    
    return fidelity


# In[ ]:


@njit
def trace_distance_qubit_single_term(p00, p11, c10, c01, sigma):
    """
    Build the density matrix from its elements and then
    Calculate the exact Trace Distance between two single-qubit (2x2) density matrices.
    For a 2x2 traceless Hermitian matrix (rho - sigma), Det(diff) = -lambda^2 <= 0.
    Therefore, the Trace Distance simplifies to sqrt(-Det(rho - sigma)).
    
    Parameters:
        rho (numpy.ndarray): First density matrix (2x2).
        sigma (numpy.ndarray): Second density matrix (2x2).
        
    Returns:
        float: The trace distance between rho and sigma.
    """
    # Density Matrix build up 
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0,0] = p00
    rho[0,1] = c01
    rho[1,0] = c10
    rho[1,1] = p11
    
    # Difference between the matrices
    diff = rho - sigma
    
    # Determinant of the difference
    det_diff = (diff[0,0]*diff[1,1] - diff[0,1]*diff[1,0]).real
    
    # Since det_diff should be <= 0, -det_diff should be >= 0.
    # We use max(0.0, ...) to truncate any negative noise before applying sqrt.
    val_under_sqrt = max(0.0, -det_diff)
    
    t_dist = np.sqrt(val_under_sqrt)
    
    return min(1.0, t_dist)


# In[ ]:


# =====================================================================
# NUMBA OPTIMIZED LOOP FOR ALL INDIVIDUAL TRAJECTORIES (3x3 MATRICES)
# =====================================================================

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
            
            # Safely extract the 3x3 density matrix
            rho_traj_t = np.zeros((3, 3), dtype=np.complex128)
            for i in range(3):
                for j in range(3):
                    rho_traj_t[i, j] = rho_all[i, j, t, n]
            
            # Compute and store metrics
            fidelity_matrix[t, n] = fidelity_generic_njit(rho_traj_t, lindblad_t)
            trace_dist_matrix[t, n] = trace_distance_generic_njit(rho_traj_t, lindblad_t)
            
    return fidelity_matrix, trace_dist_matrix


# ## General Setup

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

MODE = 'SD'   # Switch this to 'SD' or 'QJ'

# Optional: define parameters if you want to use them in titles/filenames later
dt = 0.01
N_traj = 10000
dt_str = f"{dt:.6f}".replace(".", "p")

# Configuration mapping based on MODE
_cfg = {
    'QJ': {
        'Input_file': f"../Results/Data/Complete_rho/result_mode_QJ_dt{dt_str}_Ntraj{N_traj}.npz",
        'Output_dir': "../Results/Plot/Fidelity/QJ" 
    },
    'SD': {
        'Input_file': f"../Results/Data/Complete_rho/result_mode_SD_dt{dt_str}_Ntraj{N_traj}.npz",
        'Output_dir': "../Results/Plot/Fidelity/SD"
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


# ============================================================
# DATA EXTRACTION
# ============================================================

# 1. You must load the file first using numpy!
data = np.load(Input_file)

# 2. Extract arrays from the loaded 'data' object, not the string
times = data['times']

# Extract jumps safely (State Diffusion might not have this array saved)
if 'total_jumps' in data:
    total_jumps = data['total_jumps']
else:
    total_jumps = None

# -----------------------------------------------
# Extract from rho_tot_all (3, 3, n_times, N_traj)
# -----------------------------------------------
# We use generic names now. If MODE='QJ', this will be the QJ data automatically.
rho_all = data['rho_tot_all']

# ----------------------------------------------------
# Extract Lindblad: rho_list_lindblad (n_times, 3, 3)
# ----------------------------------------------------
rho_lind = data['rho_list_lindblad']

print("Data loaded successfully!")


# ## Purity

# $ P = Tr[\rho^2]  $

# In[ ]:


# =============================================================
# Purity Check for 3-Level System (Tr[rho^2] = 1.0)
# =============================================================

print(f"Starting Purity Check for {MODE}...")
start_time = time.time()

# We use rho_all from the previous cell: shape (3, 3, n_times, N_traj)
_, _, n_times, N_traj_tot = rho_all.shape
max_deviation = 0.0

# Chunking to manage memory effectively
CHUNK_SIZE = 2000 
n_chunks = int(np.ceil(N_traj_tot / CHUNK_SIZE))

for i in range(n_chunks):
    start_idx = i * CHUNK_SIZE
    end_idx = min((i + 1) * CHUNK_SIZE, N_traj_tot)
    
    # Extract populations for the chunk (Real parts)
    p00 = np.real(rho_all[0, 0, :, start_idx:end_idx])
    p11 = np.real(rho_all[1, 1, :, start_idx:end_idx])
    p22 = np.real(rho_all[2, 2, :, start_idx:end_idx])
    
    # Extract coherences for the chunk (Complex)
    c01 = rho_all[0, 1, :, start_idx:end_idx]
    c12 = rho_all[1, 2, :, start_idx:end_idx]
    c02 = rho_all[0, 2, :, start_idx:end_idx]
    
    # 3-level Purity formula: sum of squared diagonal + 2 * sum of squared off-diagonal magnitudes
    purity_chunk = (p00**2 + p11**2 + p22**2 + 
                    2 * (np.abs(c01)**2 + np.abs(c12)**2 + np.abs(c02)**2))
    
    # Calculate max deviation from ideal pure state (1.0)
    chunk_max_dev = np.max(np.abs(1.0 - purity_chunk))
    
    if chunk_max_dev > max_deviation:
        max_deviation = chunk_max_dev

end_processing_time = time.time() - start_time
print(f"Check completed in {end_processing_time:.4f} seconds.")
print(f"Maximum deviation from ideal purity (1.0): {max_deviation:.4e}")

# Automatic evaluation of the results
if max_deviation < 1e-9:
    print(f"\n✅ Test Passed: All {MODE} trajectories remain perfectly pure.")
else:
    print(f"\n⚠️ Warning: Significant purity deviation detected ({max_deviation:.4e}).")
    print("Check normalization or solver precision.")


# ## Fidelity & Trace Distance calculation

# In[ ]:


# N_traj_current = rho_all.shape[3]

# print(f"Starting parallel computation for {N_traj_current} trajectories...")
# start_time = time.time()

# fid_matrix, td_matrix = compute_metrics_all_trajectories(rho_all, rho_lind)

# end_time = time.time() - start_time
# print(f"Parallel computation completed in {end_time:.4f} seconds.")


# In[ ]:


# # ==================================
# # Save the computed metrics to disk
# # ==================================

# Output_Data_dir = f"../Results/Data/Metrics/"
# os.makedirs(Output_Data_dir, exist_ok=True)  # Ensure the directory exists

# # Define the filename and path using your global configuration variables
# metrics_filename = f"{MODE}_F_TD_Metrics_dt{dt_str}_Ntraj{N_traj}.npz"
# save_path = os.path.join(Output_Data_dir, metrics_filename)

# print("Saving computed metrics to disk... this might take a few seconds.")

# # Save both matrices in a single compressed file
# np.savez_compressed(save_path, fid_matrix=fid_matrix, td_matrix=td_matrix)

# print(f"✅ Data successfully saved to: {save_path}")


# In[ ]:


Output_Data_dir = f"../Results/Data/Metrics/"
metrics_filename = f"{MODE}_F_TD_Metrics_dt{dt_str}_Ntraj{N_traj}.npz"
save_path = os.path.join(Output_Data_dir, metrics_filename)

if os.path.exists(save_path):
    print(f"Found existing metrics file at: {save_path}")
    print("Loading data... please wait.")
    
    # Load the previously saved compressed archive
    loaded_data = np.load(save_path)
    
    # Extract the arrays into your environment variables
    fid_matrix = loaded_data['fid_matrix']
    td_matrix = loaded_data['td_matrix']
    
    print("✅ Data loaded successfully! You are ready to plot.")
else:
    print(f"⚠️ File not found at: {save_path}")
    print("You need to run the Numba computation first.")


# ## Plot 

# In[ ]:


# =====================================================================
# Global Convergence: Fidelity and Trace Distance of the Avg State
# =====================================================================
plt.close('all')

# Ensure we don't try to plot more trajectories than we actually have
N_max = rho_all.shape[3]
N_list = [100, 1000, min(10000, N_max), N_max]

# Remove duplicates in case N_max is 10000
N_list = sorted(list(set(N_list)))

# Create a figure with 2 subplots (1 row, 2 columns)
fig01, axes = plt.subplots(1, 2, figsize=(16, 5))

n_times = len(times)

for N in N_list:
    # 1. Average the FULL density matrix over the first N trajectories
    # rho_all shape: (3, 3, n_times, N_traj) -> mean over axis 3
    rho_avg_N = rho_all[:, :, :, :N].mean(axis=3)
    
    # Pre-allocate lists for the metrics
    fidelity_list = np.zeros(n_times)
    td_list = np.zeros(n_times)

    # 2. Calculate metrics using the standard python functions 
    # (Fast enough for just a few N configurations)
    for t in range(n_times):
        # Using the standard generic functions defined earlier
        fidelity_list[t] = fidelity_generic(rho_lind[t], rho_avg_N[:, :, t])
        td_list[t] = trace_distance_generic(rho_lind[t], rho_avg_N[:, :, t])

    # 3. Plot the current N curve on both subplots
    axes[0].plot(times, td_list, label=f'N = {N}', alpha=0.9, linewidth=1.5)
    axes[1].plot(times, fidelity_list, label=f'N = {N}', alpha=0.9, linewidth=1.5)

# =====================================================================
# Formatting Left Subplot: Trace Distance
# =====================================================================
axes[0].set_xlabel("Time", fontsize=12)
axes[0].set_ylabel(r"Trace Distance $\mathcal{T}(\langle\rho\rangle_N, \rho_L)$", fontsize=12)
axes[0].set_title(f"Trace Distance Convergence vs Lindblad", fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='best')
axes[0].set_ylim(bottom=0)

# Optional: Set y-axis to logarithmic to see the exponential convergence better
# axes[0].set_yscale('log')

# =====================================================================
# Formatting Right Subplot: Fidelity
# =====================================================================
axes[1].set_xlabel("Time", fontsize=12)
axes[1].set_ylabel(r"Fidelity $\mathcal{F}(\langle\rho\rangle_N, \rho_L)$", fontsize=12)
axes[1].set_title(f"Fidelity Convergence vs Lindblad", fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='best')

# Adjust layout to prevent overlap
fig01.suptitle(f'[{MODE}] Ensemble Size Convergence | dt={dt}', fontsize=16, y=1.02)
plt.tight_layout()

# Automatically save the figure dynamically based on the MODE
filename_avg = f"{MODE}_Global_Convergence_Metrics_dt{dt_str}.png"
save_fig(fig01, filename_avg)

plt.show()


# In[ ]:


# =========================================================
# PLOT HEATMAP: FIDELITY & TRACE DISTANCE DISTRIBUTIONS
# =========================================================

plt.close('all')

# 1. Prepare the Data
# Transpose the matrices to get shape (N_traj, n_times)
# Ensure you are using the arrays generated in the previous step
fid_array = fid_matrix.T
td_array = td_matrix.T

n_traj, n_times = fid_array.shape

# 2. Heatmap Bin Parameters
n_bins = 150 
# Both metrics are physically bounded between 0.0 and 1.0
bins_array = np.linspace(0.0, 1.0, n_bins + 1)

heatmap_fid = np.zeros((n_bins, n_times))
heatmap_td = np.zeros((n_bins, n_times))

# Compute the histogram for each time step across all trajectories
for t in range(n_times):
    counts_fid, _ = np.histogram(fid_array[:, t], bins=bins_array)
    heatmap_fid[:, t] = counts_fid
    
    counts_td, _ = np.histogram(td_array[:, t], bins=bins_array)
    heatmap_td[:, t] = counts_td

# Mask zero counts to make the background transparent/white instead of dark
heatmap_masked_fid = np.ma.masked_where(heatmap_fid == 0, heatmap_fid) 
heatmap_masked_td = np.ma.masked_where(heatmap_td == 0, heatmap_td) 

# 3. Create the Figure (1 Row, 2 Columns)
fig02, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left Subplot: Trace Distance Heatmap ---
im_td = axes[0].imshow(
    heatmap_masked_td,
    aspect='auto',
    origin='lower',
    extent=[times[0], times[-1], 0.0, 1.0],  # [x_min, x_max, y_min, y_max]
    cmap='plasma', # Using a different colormap for contrast
    interpolation='nearest',
    vmin=1, 
    vmax=np.max(heatmap_td) * 0.8 # Saturate slightly to highlight lower densities
)
axes[0].set_xlabel('Time')
axes[0].set_ylabel(r'Trace Distance $\mathcal{T}$')
axes[0].set_title(f'Trace Distance Distribution ({MODE})')
cbar_td = fig02.colorbar(im_td, ax=axes[0])
cbar_td.set_label('Number of Trajectories')

# --- Right Subplot: Fidelity Heatmap ---
im_fid = axes[1].imshow(
    heatmap_masked_fid,
    aspect='auto',
    origin='lower',
    extent=[times[0], times[-1], 0.0, 1.0], 
    cmap='viridis',
    interpolation='nearest',
    vmin=1, 
    vmax=np.max(heatmap_fid) * 0.8 
)
axes[1].set_xlabel('Time')
axes[1].set_ylabel(r'Fidelity $\mathcal{F}$')
axes[1].set_title(f'Fidelity Distribution ({MODE})')
cbar_fid = fig02.colorbar(im_fid, ax=axes[1])
cbar_fid.set_label('Number of Trajectories')

# 4. Global Formatting and Saving
fig02.suptitle(f'Single Trajectory Metrics Distribution | dt={dt}, N_traj={n_traj}', fontsize=16)

# Define the base filename without the extension to use with save_fig
filename_heatmap = f"{MODE}_Heatmap_Distributions_dt{dt_str}"
save_fig(fig02, filename_heatmap)

plt.show()


# In[ ]:


# =========================================================
# Metrics for a Single Trajectory vs Lindblad
# =========================================================

plt.close('all')

# 1. Smart Trajectory Selection
# If we are in QJ mode, try to pick a trajectory that actually jumped!
if MODE == "QJ" and 'jump_indices' in locals() and len(jump_indices) > 0:
    sample_idx = jump_indices[0] 
else:
    sample_idx = 978  # Fallback for SD mode or if no jumps occurred

# 2. Extract the metrics for the chosen trajectory
# Using the matrices we computed and loaded earlier
single_traj_fid = fid_matrix[:, sample_idx]
single_traj_td = td_matrix[:, sample_idx]

# 3. Create the Figure (1 Row, 2 Columns)
fig03, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- Left Subplot: Trace Distance ---
axes[0].plot(
    times, 
    single_traj_td, 
    label=f'Single Traj (idx = {sample_idx})', 
    color='crimson', 
    linewidth=1.5, 
    alpha=0.9
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"Trace Distance $\mathcal{T}(\rho_{traj}^{(i)}, \rho_L)$")
axes[0].set_title("Single Trajectory Trace Distance vs Lindblad")
axes[0].legend(loc='best')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_ylim(bottom=0)

# --- Right Subplot: Fidelity ---
axes[1].plot(
    times, 
    single_traj_fid, 
    label=f'Single Traj (idx = {sample_idx})', 
    color='dodgerblue', 
    linewidth=1.5, 
    alpha=0.9
)
axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"Fidelity $\mathcal{F}(\rho_{traj}^{(i)}, \rho_L)$")
axes[1].set_title("Single Trajectory Fidelity vs Lindblad")
axes[1].legend(loc='best')
axes[1].grid(True, linestyle='--', alpha=0.5)

# 4. Global Formatting and Saving
fig03.suptitle(f'[{MODE}] Single Trajectory Metrics | dt={dt}, idx={sample_idx}', fontsize=16, y=1.02)
plt.tight_layout()

# Automatically save the figure
filename_single = f"{MODE}_Single_Traj_Metrics_idx{sample_idx}_dt{dt_str}"
save_fig(fig03, filename_single)

plt.show()


# ## Average Calculations

# In[ ]:


# =====================================================================
# NUMBA HELPER: METRICS FOR AVERAGED STATE
# =====================================================================
@njit
def compute_metrics_avg_state(rho_avg_N, rho_lind):
    """
    Computes Fidelity and Trace Distance over time between the 
    averaged stochastic state and the exact Lindblad state.
    """
    n_times = rho_lind.shape[0]
    fid_arr = np.zeros(n_times)
    td_arr = np.zeros(n_times)
    
    for t in range(n_times):
        # Safely extract the 3x3 density matrix to avoid Numba memory issues
        rho_t = np.zeros((3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                rho_t[i, j] = rho_avg_N[i, j, t]
                
        # Calculate using the previously defined Numba generic functions
        fid_arr[t] = fidelity_generic_njit(rho_t, rho_lind[t])
        td_arr[t] = trace_distance_generic_njit(rho_t, rho_lind[t])
        
    return fid_arr, td_arr

# ===================================================================
# SCALING ANALYSIS: Metrics over time for different N_traj
# ===================================================================

print(f"Starting Scaling Analysis for {MODE}...")
start_time = time.time()

# Target list of trajectories to analyze
target_N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
                 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]

# Ensure we don't ask for more trajectories than we have in rho_all
total_available_traj = rho_all.shape[3]
N_traj_list = [N for N in target_N_list if N <= total_available_traj]

# Initialize lists to store the full time-evolution arrays
fidelity_results_list = []
trace_dist_results_list = []

for N in N_traj_list:
    print(f"Processing N = {N} trajectories...")
    
    # Calculate the mean density matrix over the first N trajectories
    # rho_all shape is (3, 3, n_times, N_traj)
    rho_avg_N = rho_all[:, :, :, :N].mean(axis=3)

    # Compute metrics over time using the ultra-fast Numba function
    fid_time_array, td_time_array = compute_metrics_avg_state(rho_avg_N, rho_lind)
    
    # Append the resulting 1D arrays to the lists
    fidelity_results_list.append(fid_time_array)
    trace_dist_results_list.append(td_time_array)

print(f"Computation fully processed in {time.time() - start_time:.2f} seconds.")

# ===================================================
# MAX & MEAN INFIDELITY & TRACE DISTANCE CALCULATION 
# ===================================================

print("Extracting Infidelity and Trace Distance scalar metrics...")

# Lists to store the calculated scalar metrics for plotting
max_infidelity_results = []
mean_infidelity_results = []
max_trace_distance_results = []
mean_trace_distance_results = []

# Loop through the indices of the lists
for i in range(len(N_traj_list)):
    # Extract the arrays for this specific N
    fid_time_array = fidelity_results_list[i]
    td_time_array = trace_dist_results_list[i]
        
    # Calculate Infidelity: 1 - Fidelity
    infidelity_array = 1.0 - fid_time_array
        
    # Calculate the maximum and temporal mean of the metrics
    max_infidelity_results.append(np.max(infidelity_array))
    mean_infidelity_results.append(np.mean(infidelity_array))
    
    max_trace_distance_results.append(np.max(td_time_array))
    mean_trace_distance_results.append(np.mean(td_time_array))

# Convert lists to NumPy arrays for easier mathematical operations and plotting
max_infidelity_results = np.array(max_infidelity_results)
mean_infidelity_results = np.array(mean_infidelity_results)
max_trace_distance_results = np.array(max_trace_distance_results)
mean_trace_distance_results = np.array(mean_trace_distance_results)

print("Scalar metrics computed successfully!")


# In[ ]:


# ===================================================================
# SCALING ANALYSIS PLOTS: INFIDELITY & TRACE DISTANCE (2x2 GRID)
# ===================================================================

plt.close('all')

# Create a 2x2 grid of subplots for comprehensive analysis
fig11, axes = plt.subplots(2, 2, figsize=(16, 12))

# Convert N_traj_list to a numpy array for vector math operations
N_arr = np.array(N_traj_list)

# -------------------------------------------------------------------
# 1. Reference Slopes for Central Limit Theorem
# Infidelity scales as 1/N, Trace Distance scales as 1/sqrt(N)
# We anchor the theoretical reference lines to the first data point
# -------------------------------------------------------------------
ref_infidelity_mean = mean_infidelity_results[0] * (N_arr[0] / N_arr)
ref_infidelity_max = max_infidelity_results[0] * (N_arr[0] / N_arr)

ref_td_mean = mean_trace_distance_results[0] * np.sqrt(N_arr[0] / N_arr)
ref_td_max = max_trace_distance_results[0] * np.sqrt(N_arr[0] / N_arr)

# -------------------------------------------------------------------
# ROW 1: INFIDELITY (Mean and Max)
# -------------------------------------------------------------------

# Left: Mean Infidelity
axes[0, 0].plot(N_arr, mean_infidelity_results, marker='o', linestyle='-', color='blue', label=f'Mean Infidelity ({MODE})')
axes[0, 0].plot(N_arr, ref_infidelity_mean, linestyle='--', color='black', alpha=0.7, label=r'$\propto 1/N$ (Theory)')
axes[0, 0].set_xlabel(r"$N_{\text{traj}}$")
axes[0, 0].set_ylabel(r"$1 - \langle \mathcal{F} \rangle_t$")
axes[0, 0].set_title("Mean Infidelity vs Trajectories")
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[0, 0].legend()

# Right: Max Infidelity
axes[0, 1].plot(N_arr, max_infidelity_results, marker='s', linestyle='-', color='darkblue', label=f'Max Infidelity ({MODE})')
axes[0, 1].plot(N_arr, ref_infidelity_max, linestyle='--', color='black', alpha=0.7, label=r'$\propto 1/N$ (Theory)')
axes[0, 1].set_xlabel(r"$N_{\text{traj}}$")
axes[0, 1].set_ylabel(r"$\max_t (1 - \mathcal{F})$")
axes[0, 1].set_title("Max Infidelity vs Trajectories")
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, which="both", linestyle='--', alpha=0.5)
axes[0, 1].legend()

# -------------------------------------------------------------------
# ROW 2: TRACE DISTANCE (Mean and Max)
# -------------------------------------------------------------------

# Left: Mean Trace Distance
axes[1, 0].plot(N_arr, mean_trace_distance_results, marker='o', linestyle='-', color='red', label=f'Mean TD ({MODE})')
axes[1, 0].plot(N_arr, ref_td_mean, linestyle='--', color='black', alpha=0.7, label=r'$\propto 1/\sqrt{N}$ (Theory)')
axes[1, 0].set_xlabel(r"$N_{\text{traj}}$")
axes[1, 0].set_ylabel(r"$\langle \mathcal{T} \rangle_t$")
axes[1, 0].set_title("Mean Trace Distance vs Trajectories")
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1, 0].legend()

# Right: Max Trace Distance
axes[1, 1].plot(N_arr, max_trace_distance_results, marker='s', linestyle='-', color='darkred', label=f'Max TD ({MODE})')
axes[1, 1].plot(N_arr, ref_td_max, linestyle='--', color='black', alpha=0.7, label=r'$\propto 1/\sqrt{N}$ (Theory)')
axes[1, 1].set_xlabel(r"$N_{\text{traj}}$")
axes[1, 1].set_ylabel(r"$\max_t (\mathcal{T})$")
axes[1, 1].set_title("Max Trace Distance vs Trajectories")
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1, 1].legend()

# -------------------------------------------------------------------
# GLOBAL FORMATTING AND SAVING
# -------------------------------------------------------------------
fig11.suptitle(f'[{MODE}] Scaling Analysis & Central Limit Theorem Validation | dt={dt}', fontsize=18, y=1.02)
plt.tight_layout()

# Save the unified figure
filename_scaling = f"{MODE}_Scaling_Analysis_Metrics_dt{dt_str}"
save_fig(fig11, filename_scaling)

plt.show()


# In[ ]:


# ===================================================================
# Helper Function for Log-Log Power Law Fit
# ===================================================================
def power_law_fit(x, y):
    """
    Fits the data to a power law y = A * x^B in log-log space.
    Returns the slope (B), intercept (log10(A)), and the fitted y values.
    """
    # Exclude any potential zeros to avoid log10 errors
    valid_idx = y > 0
    x_val = x[valid_idx]
    y_val = y[valid_idx]
    
    log_x = np.log10(x_val)
    log_y = np.log10(y_val)
    
    # Perform a 1st degree polynomial fit (linear fit) on the log-log data
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Generate the fitted curve over the original x points
    fit_y = (10**intercept) * (x**slope)
    
    return slope, intercept, fit_y

# ===================================================================
# SCALING ANALYSIS PLOTS: WITH FITTING (2x2 GRID)
# ===================================================================

plt.close('all')

# Create a 2x2 grid of subplots
fig11, axes = plt.subplots(2, 2, figsize=(16, 12))

# Convert N_traj_list to a numpy array
N_arr = np.array(N_traj_list)

# -------------------------------------------------------------------
# Calculate Fits for all metrics
# -------------------------------------------------------------------
slope_inf_mean, int_inf_mean, fit_inf_mean = power_law_fit(N_arr, mean_infidelity_results)
slope_inf_max, int_inf_max, fit_inf_max = power_law_fit(N_arr, max_infidelity_results)

slope_td_mean, int_td_mean, fit_td_mean = power_law_fit(N_arr, mean_trace_distance_results)
slope_td_max, int_td_max, fit_td_max = power_law_fit(N_arr, max_trace_distance_results)

# -------------------------------------------------------------------
# ROW 1: INFIDELITY (Mean and Max)
# Theoretical slope: -1.0
# -------------------------------------------------------------------

# Left: Mean Infidelity
axes[0, 0].scatter(N_arr, mean_infidelity_results, color='blue', alpha=0.7, label=f'Data ({MODE})')
axes[0, 0].plot(N_arr, fit_inf_mean, linestyle='-', color='blue', 
                label=f'Fit: slope = {slope_inf_mean:.3f}\nInt = {int_inf_mean:.3f}')
axes[0, 0].plot(N_arr, fit_inf_mean[0] * (N_arr[0] / N_arr), linestyle='--', color='black', alpha=0.5, label='Theory (slope = -1.0)')
axes[0, 0].set_xlabel(r"$N_{\text{traj}}$")
axes[0, 0].set_ylabel(r"$1 - \langle \mathcal{F} \rangle_t$")
axes[0, 0].set_title("Mean Infidelity vs Trajectories")
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[0, 0].legend()

# Right: Max Infidelity
axes[0, 1].scatter(N_arr, max_infidelity_results, color='darkblue', alpha=0.7, label=f'Data ({MODE})')
axes[0, 1].plot(N_arr, fit_inf_max, linestyle='-', color='darkblue', 
                label=f'Fit: slope = {slope_inf_max:.3f}\nInt = {int_inf_max:.3f}')
axes[0, 1].plot(N_arr, fit_inf_max[0] * (N_arr[0] / N_arr), linestyle='--', color='black', alpha=0.5, label='Theory (slope = -1.0)')
axes[0, 1].set_xlabel(r"$N_{\text{traj}}$")
axes[0, 1].set_ylabel(r"$\max_t (1 - \mathcal{F})$")
axes[0, 1].set_title("Max Infidelity vs Trajectories")
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, which="both", linestyle='--', alpha=0.5)
axes[0, 1].legend()

# -------------------------------------------------------------------
# ROW 2: TRACE DISTANCE (Mean and Max)
# Theoretical slope: -0.5
# -------------------------------------------------------------------

# Left: Mean Trace Distance
axes[1, 0].scatter(N_arr, mean_trace_distance_results, color='red', alpha=0.7, label=f'Data ({MODE})')
axes[1, 0].plot(N_arr, fit_td_mean, linestyle='-', color='red', 
                label=f'Fit: slope = {slope_td_mean:.3f}\nInt = {int_td_mean:.3f}')
axes[1, 0].plot(N_arr, fit_td_mean[0] * np.sqrt(N_arr[0] / N_arr), linestyle='--', color='black', alpha=0.5, label='Theory (slope = -0.5)')
axes[1, 0].set_xlabel(r"$N_{\text{traj}}$")
axes[1, 0].set_ylabel(r"$\langle \mathcal{T} \rangle_t$")
axes[1, 0].set_title("Mean Trace Distance vs Trajectories")
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1, 0].legend()

# Right: Max Trace Distance
axes[1, 1].scatter(N_arr, max_trace_distance_results, color='darkred', alpha=0.7, label=f'Data ({MODE})')
axes[1, 1].plot(N_arr, fit_td_max, linestyle='-', color='darkred', 
                label=f'Fit: slope = {slope_td_max:.3f}\nInt = {int_td_max:.3f}')
axes[1, 1].plot(N_arr, fit_td_max[0] * np.sqrt(N_arr[0] / N_arr), linestyle='--', color='black', alpha=0.5, label='Theory (slope = -0.5)')
axes[1, 1].set_xlabel(r"$N_{\text{traj}}$")
axes[1, 1].set_ylabel(r"$\max_t (\mathcal{T})$")
axes[1, 1].set_title("Max Trace Distance vs Trajectories")
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, which="both", linestyle='--', alpha=0.5)
axes[1, 1].legend()

# -------------------------------------------------------------------
# GLOBAL FORMATTING AND SAVING
# -------------------------------------------------------------------
fig11.suptitle(f'[{MODE}] Scaling Analysis with Linear Fit | dt={dt}', fontsize=18, y=1.02)
plt.tight_layout()

# Save the unified figure
filename_fit = f"{MODE}_Scaling_Analysis_Fitted_dt{dt_str}"
save_fig(fig11, filename_fit)

plt.show()


# In[ ]:




