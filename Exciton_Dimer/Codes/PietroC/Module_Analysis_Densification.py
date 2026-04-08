#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================
# Jupyter Magic Commands & Core Imports
# ==========================================
# Enable interactive matplotlib plots
#get_ipython().run_line_magic('matplotlib', 'widget')

# Load the autoreload extension to automatically update custom modules
#get_ipython().run_line_magic('load_ext', 'autoreload')
# Set autoreload to update modules every time a cell is executed
#get_ipython().run_line_magic('autoreload', '2')

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting (suitable for cluster environments)
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import sys  # Used to read command line arguments
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import os

# ==========================================
# Custom Modules Import
# ==========================================

from densification import NJIT_syncr_measure_time,  NJIT_bloch_coords, NJIT_vectors_inCartesian_coords
import visualization 
# import nonmarkovian


# ## Data Load

# In[ ]:


# ====================================
# Physical & Simulation Parameters
# ====================================
# Theta angle in degrees, H_Coll Direction
if len(sys.argv) > 1:
    try:
        theta_target_deg = float(sys.argv[1])
    except ValueError:
        print(f"ERROR: Cannot convert '{sys.argv[1]}' to float.")
        sys.exit(1)
else:
    theta_target_deg = 90.0  

theta_rad = np.radians(theta_target_deg)

# Parse Argument 2: Mode (normal or close_to_90)
if len(sys.argv) > 2:
    MODE = sys.argv[2]
else:
    MODE = "normal"

print("\n" + "="*50)
print(f"🚀 STARTING ANALYSIS FOR THETA = {theta_target_deg}° (Mode: {MODE})")
print("="*50 + "\n")

# Site selector: 0 for |10>, 1 for |01>
site_index = 0

# Time step
dt = 0.01

# Number of trajectories to analyze
N_traj_to_plot = 100         

# =================
# Input Data Setup
# =================
# Set the input directory dynamically
if MODE == 'normal':
    Input_dir = "../../Results/Data/Complete_rho/normal"
elif MODE == 'close_to_90':
    Input_dir = "../../Results/Data/Complete_rho/close_90_deg"
else:
    raise ValueError(f"Unknown mode provided: {MODE}")

# Format theta and dt for filename 
theta_str = f"{theta_rad:.6f}".replace(".", "p")
dt_str = f"{dt:.6f}".replace(".", "p")

# File name
filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj20000.npz"
filepath = os.path.join(Input_dir, filename)

print(f"Analisi impostata per theta = {theta_target_deg}°")
print(f"File target: {filename}")

if not os.path.exists(filepath):
    print(f"ERRORE: The file {filepath} doesn't exist. Check file name.")
else:
    # Load .npz input containing data
    data = np.load(filepath)
    
    times= data['times']

    # Define a downsampling factor (e.g., take 1 every 10 time steps)
    time_downsample_factor = 10
    
    # Downsample the time array
    time_stepped = times[::time_downsample_factor]
    N_time = len(time_stepped)
    
    print("Matrix extraction and downsampling in progress")
    
    # ================================
    # Raw Trajectories Extraction
    # ================================
    
    # Extract the full time array
    full_times = data['times']
    
    # Extract and downsample the raw trajectories along the time axis (axis 0)
    # Then transpose (.T) to get the expected (N_traj, N_time) shape
    pop_traj_10 = data['pop_00'][::time_downsample_factor, :].T
    pop_traj_01 = data['pop_11'][::time_downsample_factor, :].T
    
    cohe_traj_10_01 = data['coh_10_01'][::time_downsample_factor, :].T
    cohe_traj_01_10 = data['coh_01_10'][::time_downsample_factor, :].T

    # Extract real and imaginary parts of the coherence
    rho12_re = np.real(cohe_traj_10_01)
    rho12_im = np.imag(cohe_traj_10_01)
    
    # Map populations to density matrix diagonal elements
    rho11 = pop_traj_10
    rho22 = pop_traj_01

    print("Data extraction completed")

# ── Ricostruzione delle matrici densità 2x2 ──────────────────────────────
#
#   rho = [[rho11,        rho12_re + i*rho12_im],
#           [rho12_re - i*rho12_im, rho22       ]]
#
N_traj, N_time = rho11.shape

many_rho = np.zeros((N_traj, N_time, 2, 2), dtype=complex)
many_rho[:, :, 0, 0] =  rho11
many_rho[:, :, 1, 1] =  rho22
many_rho[:, :, 0, 1] =  rho12_re + 1j * rho12_im
many_rho[:, :, 1, 0] =  rho12_re - 1j * rho12_im

print(f"Shape many_rho: {many_rho.shape}")
print(f"Esempio rho[traj=0, t=0]:\n{many_rho[0,0]}")
print(f"Traccia (deve essere ~1): {np.real(np.trace(many_rho[0,0])):.4f}")


# In[ ]:


# ===========================
# General Setup for Plotting
# ===========================
# --- 1. Output Directory---

if theta_target_deg.is_integer():
    angle_folder = str(int(theta_target_deg))
else:
    angle_folder = str(theta_target_deg)

Output_dir = os.path.join("../../Results/Plot/Densification", angle_folder)
os.makedirs(Output_dir, exist_ok=True)

# --- . Automatic Figure Saving Function ---
def save_fig(fig, filename):
    """
    Saves the figure in both PNG or PDF
    """
    path_png = os.path.join(Output_dir, f"{filename}.png")
    # path_pdf = os.path.join(Output_dir, f"{filename}.pdf")  # save in pdf
    
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    # fig.savefig(path_pdf, bbox_inches='tight') # save in pdf
    print(f"Figure saved in: {Output_dir}/{filename}")


# ## Densification Measure

# In[ ]:


# Define how many trajectories to use for the synchronization measure
# 1000 trajectories = ~500,000 pairs (400 times faster than 20,000!)
N_traj_sync = 10000

# Pass only the sliced array to the function
BF_syncr_meas = NJIT_syncr_measure_time(
    many_rho[:N_traj_sync],
    norm     = np.pi / 2,
    minusone = True
)

print(f"Measure calculated on {N_time} time steps, {N_traj_sync} trajectories.")
print(f"Initial value: {BF_syncr_meas[0]:.4f}   Final value: {BF_syncr_meas[-1]:.4f}")


# In[ ]:


# ── Plot densificazione nel tempo ───

plt.close('all')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_stepped, BF_syncr_meas, color='k', lw=2, label=r'$\bar{\alpha}(t) = 1 - \langle\theta\rangle / (\pi/2)$')
ax.set_xlabel('Tempo')
ax.set_ylabel(r'$\bar{\alpha}$')
ax.set_title('Misura di densificazione')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

filename_dens = f"densification_measure_theta{theta_str}_dt{dt_str}_Ntraj{N_traj_sync}"
save_fig(fig, filename_dens)

#plt.show()


# ## SVD Analysis

# In[ ]:


# Define the number of trajectories to use for the SVD analysis
N_traj_svd = 10000  # Change this value to your desired sample size

sing_vals = []
for t in range(N_time):
    # Extract Bloch vectors ONLY for the specified subset of trajectories
    # many_rho[:N_traj_svd] slices the first N_traj_svd elements along axis 0
    vects = NJIT_vectors_inCartesian_coords(many_rho[:N_traj_svd], t_idx=t)  # shape (3, N_traj_svd)
    
    # Perform SVD using compute_uv=False to avoid memory overflow (RAM crash)
    # It only computes and returns the singular values (S)
    S = np.linalg.svd(vects, compute_uv=False)
    sing_vals.append(S)

# Convert the list to a NumPy array for plotting
sing_vals = np.array(sing_vals)  # shape (N_time, 3)

plt.close('all')

# ── Plot SVD Singular Values ─────
fig_svd, ax = plt.subplots(figsize=(8, 4))

labels_sv = ['1st singular value (main spread)',
             '2nd singular value',
             '3rd singular value']

# Loop through the 3 singular values and plot their time evolution
for k in range(3):
    ax.plot(time_stepped, sing_vals[:, k], label=labels_sv[k])

ax.set_xlabel('Time')
ax.set_ylabel('Singular Value')
ax.set_title(f'SVD of Bloch Vectors over Time (Subset: {N_traj_svd} Trajectories)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

filename_svd = f"SVD_singular_values_theta{theta_str}_dt{dt_str}_Ntraj{N_traj_svd}"
save_fig(fig_svd, filename_svd)

#plt.show()


# ## Bloch Sphere Analysis

# In[ ]:


# Reducing the number of trajectories 
N_traj_reduced = 10000  # Change this value to your desired sample size
avg_many_rho = np.mean(many_rho, axis=0)  # shape (N_time, 2, 2)
print(f"Shape avg_many_rho: {avg_many_rho.shape}")


# In[ ]:


#visualization.plot_multiple_bloch_trajectories(many_rho[:N_traj_reduced], quiv_init=True, xylabels=True)
visualization.plot_multiple_bloch_trajectories(avg_many_rho, quiv_init=True, xylabels=True)

fig_avg_traj = plt.gcf()

filename_avg_traj = f"AverageTrajectory_theta{theta_str}_dt{dt_str}"

save_fig(fig_avg_traj, filename_avg_traj)

plt.close(fig_avg_traj)


# In[ ]:


plt.close('all')

fig_multi_bloch, axes = visualization.multi_figure_bloch_plot(n_subplots=4, ncols=2, xylabels=True, figsize=(15,10))

visualization.plot_onebloch_multipletrajectories(ax=axes[0], arrays=[avg_many_rho], alp=1,  quiv_init=True, rot_viev=(0, 0),  xylabels=True)
visualization.plot_onebloch_multipletrajectories(ax=axes[2], arrays=[avg_many_rho], alp=1,  quiv_init=True, rot_viev=(0, 90), xylabels=True, labels=[r'$\xi$'], showlegend=True)

visualization.plot_onebloch_multipletrajectories(ax=axes[1], arrays=many_rho[:N_traj_reduced], rot_viev=(0, 0),  xylabels=True, alp=0.1, colormap_plain=False, quiv_init=False, quiv=False, quiv_alpha=0.35 ) #title='dW') #cyan
visualization.plot_onebloch_multipletrajectories(ax=axes[3], arrays=many_rho[:N_traj_reduced], rot_viev=(0, 90), xylabels=True, alp=0.1, colormap_plain=False, quiv_init=False, quiv=False, quiv_alpha=0.35 ) #title='dW') 

plt.subplots_adjust(wspace=None, hspace=None)

filename_multi_bloch = f"MultiBlochGrid_theta{theta_str}_dt{dt_str}"

# Call your custom saving function
save_fig(fig_multi_bloch, filename_multi_bloch)

# Close the figure to free memory (crucial for cluster loops)
plt.close(fig_multi_bloch)


# ## GIF

# In[ ]:


# ----------------------
# Path to save the GIFs
# ----------------------
path_gif_avg = os.path.join(Output_dir, f"bloch_avg_theta{theta_str}.gif")
path_gif_fading = os.path.join(Output_dir, f"bloch_fading_theta{theta_str}.gif")

# -------------------------------------
# 1. Slice and downsample trajectories
# -------------------------------------
# plot the first 10 trajectories
# take 1 time step every 50 
# array[trajectories, time_steps, rows, cols]
sampled_trajectories = many_rho[:10, ::5, :, :]
sampled_avg_trajectories = np.mean(sampled_trajectories, axis=0)

visualization.generate_bloch_animation(
    array=sampled_avg_trajectories,  # Input: The mean density matrix over time
    filename=path_gif_avg,           # Output: Full path including filename
    color='black',                   # Black color to distinguish the average
    fps=30,                   
    showit=False                     # VERY IMPORTANT: False prevents UI crashes on the cluster
)



# In[ ]:


visualization.MULTI_FadingTrails_generate_bloch_animation(
    rho_list=sampled_trajectories,   # Pass the DOWNSAMPLED array
    filename=path_gif_fading,
    fps=5,
    colormap_use=True,               # Use gradient colors for the trajectories
    alp=0.75,
    trail_len=15,                    # Length of the comet tail (adjust based on new time steps)
    save_every_n=1                   # Set to 1 because we ALREADY downsampled the array!
)


# In[ ]:




