import numpy as np
from numba import njit
import qutip as qt

# --- Matplotlib Headless Backend Setup ---
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting (suitable for cluster environments)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Added for custom legend
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import imageio.v2 as imageio
import os
import sys 

# ==========================================
# Custom Modules Import
# ==========================================
from densification import NJIT_syncr_measure_time, NJIT_bloch_coords, NJIT_vectors_inCartesian_coords
import visualization 

# ====================================
# Argument Parsing & Setup
# ====================================
if len(sys.argv) < 3:
    print("Error: Missing arguments. Usage: python script.py <THETA> <MODE>")
    sys.exit(1)

try:
    theta_target_deg = float(sys.argv[1])
    theta_input_str = sys.argv[1]  # Keep the raw string for naming the GIF (e.g., "90.0")
except ValueError:
    print(f"ERROR: Cannot convert '{sys.argv[1]}' to float.")
    sys.exit(1)

MODE = sys.argv[2]
theta_rad = np.radians(theta_target_deg)

print("\n" + "="*50)
print(f"🚀 STARTING ANALYSIS FOR THETA = {theta_target_deg}° (Mode: {MODE})")
print("="*50 + "\n")

# Physical & Simulation Parameters
site_index = 0
dt = 0.01
N_traj_svd = 10000  # Number of trajectories to analyze with SVD

# =================
# Input Data Setup
# =================
if MODE == 'normal':
    Input_dir = "../../Results/Data/Complete_rho/normal"
elif MODE == 'close_to_90':
    Input_dir = "../../Results/Data/Complete_rho/close_90_deg"
else:
    raise ValueError(f"Unknown mode provided: {MODE}")

# Format theta and dt for .npz filename 
theta_file_str = f"{theta_rad:.6f}".replace(".", "p")
dt_str = f"{dt:.6f}".replace(".", "p")

filename = f"result_theta{theta_file_str}_dt{dt_str}_Ntraj20000.npz"
filepath = os.path.join(Input_dir, filename)

print(f"Target file: {filename}")

if not os.path.exists(filepath):
    print(f"ERROR: The file {filepath} doesn't exist.")
    sys.exit(1)

# ================================
# Data Extraction & Downsampling
# ================================
data = np.load(filepath)
times = data['times']

time_downsample_factor = 1 # Adjust this factor to reduce the number of time steps for SVD (e.g., 10, 100, etc.)
time_stepped = times[::time_downsample_factor]
N_time = len(time_stepped)

print("Matrix extraction and downsampling in progress...")

pop_traj_10 = data['pop_00'][::time_downsample_factor, :].T
pop_traj_01 = data['pop_11'][::time_downsample_factor, :].T
cohe_traj_10_01 = data['coh_10_01'][::time_downsample_factor, :].T

rho12_re = np.real(cohe_traj_10_01)
rho12_im = np.imag(cohe_traj_10_01)
rho11 = pop_traj_10
rho22 = pop_traj_01

# Reconstruct 2x2 density matrices
N_traj, _ = rho11.shape
many_rho = np.zeros((N_traj, N_time, 2, 2), dtype=complex)
many_rho[:, :, 0, 0] = rho11
many_rho[:, :, 1, 1] = rho22
many_rho[:, :, 0, 1] = rho12_re + 1j * rho12_im
many_rho[:, :, 1, 0] = rho12_re - 1j * rho12_im

print("Data extraction completed.")

# ======================
# SVD Analysis Function
# ======================

def plot_svd_components_evolution(time, sing_vals, V_list, theta_str, output_path):
    """
    Plots x, y, z components of the 3 principal axes on the primary Y-axis.
    Plots the Singular Value magnitude as a continuous black line on a secondary Y-axis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axis_names = ['1st Principal Axis (Largest SV)', '2nd Principal Axis', '3rd Principal Axis (Smallest SV)']
    
    # Get the global maximum singular value to scale the secondary axes uniformly
    max_s = np.max(sing_vals) if np.max(sing_vals) > 0 else 1.0

    for i in range(3):
        ax = axes[i]
        
        # 1. Primary Y-axis: Plot the Cartesian components
        ax.plot(time, V_list[:, 0, i], label='x component', color='red', lw=1.5, alpha=0.6)
        ax.plot(time, V_list[:, 1, i], label='y component', color='green', lw=1.5, alpha=0.6)
        ax.plot(time, V_list[:, 2, i], label='z component', color='blue', lw=1.5, alpha=0.6)
        
        ax.set_title(f"{axis_names[i]} Evolution")
        ax.set_ylabel("Component Value")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3)

        # 2. Secondary Y-axis: Plot the Singular Value as a continuous black line
        ax2 = ax.twinx()
        
        # Plot the SV as a simple black line with alpha=0.8
        ax2.plot(time, sing_vals[:, i], color='black', lw=1.5, alpha=0.8, label='Singular Value')
        
        # Set the y-limits dynamically up to the global max SV + 10% padding
        ax2.set_ylim(0, max_s * 1.1)
        ax2.set_ylabel("Singular Value", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Combine legends from both primary and secondary axes on the first subplot
        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            # Combine them in a single legend
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', ncol=4, fontsize='small')

    axes[2].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Analysis] Component plot saved: {output_path}")

# ================================
# Numba Optimized SVD
# ================================
@njit
def fast_svd_evolution(many_rho_subset):
    n_traj = many_rho_subset.shape[0]
    n_time = many_rho_subset.shape[1]
    
    sing_vals_list = np.empty((n_time, 3), dtype=np.float64)
    V_list = np.empty((n_time, 3, 3), dtype=np.float64)
    vects = np.empty((n_traj, 3), dtype=np.float64)

    U_means_list = np.empty((n_time, 3), dtype=np.float64) # for average analysis
    
    for t in range(n_time):
        for i in range(n_traj):
            rho = many_rho_subset[i, t]
            bloch_vec = NJIT_bloch_coords(rho) 
            vects[i, 0] = bloch_vec[0]
            vects[i, 1] = bloch_vec[1]
            vects[i, 2] = bloch_vec[2]
            
        U, S, Vh = np.linalg.svd(vects, full_matrices=False)
        sing_vals_list[t, :] = S

        # Calculate mean of U rows for each principal component
        for j in range(3):
            col_sum = 0.0
            for i in range(n_traj):
                col_sum += U[i, j]
            U_means_list[t, j] = col_sum / n_traj
        
        for r in range(3):
            for c in range(3):
                V_list[t, c, r] = Vh[r, c]
                
    return sing_vals_list, V_list, U_means_list

print("Computing SVD evolution via Numba...")
many_rho_subset = many_rho[:N_traj_svd]
sing_vals_list, V_list, U_means_list = fast_svd_evolution(many_rho_subset)
print("SVD computation finished.")

# ================================
# GIF Generation Function
# ================================
def create_bloch_svd_gif(sing_vals_list, V_list, N_traj, N_time, theta_str, filename="bloch_svd_evolution.gif", frame_step=100, frame_duration=0.2):
    filenames = []
    temp_dir = f"temp_gif_frames_theta_{theta_str}"
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        
    print(f"[Theta {theta_str}] Generating frames (1 every {frame_step} steps)...")
    
    for t_idx in range(0, N_time, frame_step):
        b = qt.Bloch()
        
        S = sing_vals_list[t_idx]
        V = V_list[t_idx]
        S_norm = S / np.sqrt(N_traj)
        
        vec_1 = V[:, 0] * S_norm[0]
        vec_2 = V[:, 1] * S_norm[1]
        vec_3 = V[:, 2] * S_norm[2]
        
        b.add_vectors([vec_1, vec_2, vec_3])
        b.vector_color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        b.render()
        plt.title(f"SVD Axes | Theta: {theta_str}° | Step: {t_idx} / {N_time}")
        
        # --- NEW LEGEND BLOCK ---
        # Create custom legend handles matching the colors of the SVD vectors
        patch1 = mpatches.Patch(color='#1f77b4', label='1st Axis (Max Spread)')
        patch2 = mpatches.Patch(color='#ff7f0e', label='2nd Axis (Intermediate)')
        patch3 = mpatches.Patch(color='#2ca02c', label='3rd Axis (Min Spread)')

        # Add the legend to the current figure. 
        # bbox_to_anchor places it slightly outside the sphere to prevent overlapping.
        plt.legend(handles=[patch1, patch2, patch3], loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize='small')
        # ------------------------
        
        frame_filename = os.path.join(temp_dir, f"frame_{t_idx:05d}.png")
        plt.savefig(frame_filename, bbox_inches='tight', dpi=150)
        filenames.append(frame_filename)
        
        # Safely close QuTiP's figure to avoid memory leaks
        plt.close(b.fig)
        b.clear()

    print(f"[Theta {theta_str}] Stitching frames into a GIF...")
    
    with imageio.get_writer(filename, mode='I', duration=frame_duration) as writer:
        for fname in filenames:
            image = imageio.imread(fname)
            writer.append_data(image)
            
    print(f"[Theta {theta_str}] GIF successfully saved as {filename}!")
    
    for fname in filenames:
        os.remove(fname)
    os.rmdir(temp_dir)

# ===========================================
# Avg Reconstruction Error Analysis Function
# ===========================================

def plot_reconstruction_error(time, many_rho, sing_vals, V_list, U_means, theta_str, output_path):
    """
    Compares the true average Bloch vector with the SVD reconstruction.
    Analyzes the error when using only 1, 2, or 3 components.
    """
    # 1. Calculate True Average Dynamics directly from density matrices
    # N_traj, N_time, 2, 2
    avg_rho = np.mean(many_rho, axis=0)
    true_R = np.zeros((len(time), 3))
    for t in range(len(time)):
        true_R[t] = NJIT_bloch_coords(avg_rho[t])
    
    # 2. Reconstruct using SVD components
    # R_rec = sum_{j=1}^k (U_mean[j] * S[j]) * V[:, j]
    def reconstruct(k_components):
        rec = np.zeros((len(time), 3))
        for j in range(k_components):
            # Weigh the direction V by the singular value and the mean coefficient from U
            coeff = U_means[:, j] * sing_vals[:, j]
            direction = V_list[:, :, j]
            rec += (direction.T * coeff).T 
        return rec

    rec1 = reconstruct(1)
    rec2 = reconstruct(2)
    rec3 = reconstruct(3)

    # 3. Calculate Euclidean Error
    err1 = np.linalg.norm(true_R - rec1, axis=1)
    err2 = np.linalg.norm(true_R - rec2, axis=1)
    err3 = np.linalg.norm(true_R - rec3, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(time, err1, label='Error (1 component)', color='red', ls='--')
    plt.plot(time, err2, label='Error (2 components)', color='blue', lw=2)
    plt.plot(time, err3, label='Error (3 components - full)', color='black', alpha=0.3)
    
    plt.title(f"SVD Reconstruction Error | Theta: {theta_str}°")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.yscale('log') # Error is often small, log scale helps
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Analysis] Reconstruction error plot saved: {output_path}")

# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    theta_input_str = sys.argv[1]
    MODE = sys.argv[2]
    # ... (Keep your data loading part for many_rho here) ...

    # --- Path Definition ---
    output_dir = "../../Results/Densification/SVD_Analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    gif_path = os.path.join(output_dir, f"bloch_svd_evolution_theta_{theta_input_str}.gif")
    svd_data_path = os.path.join(output_dir, f"svd_data_theta_{theta_input_str}.npz")
    analysis_plot_path = os.path.join(output_dir, f"svd_components_theta_{theta_input_str}.png")
    # NEW: Path for the reconstruction error plot
    error_plot_path = os.path.join(output_dir, f"svd_error_theta_{theta_input_str}.png")

    # --- Step 1: Check/Compute SVD Data ---
    recompute_needed = False
    if os.path.exists(svd_data_path):
        print(f"📦 Loading existing SVD data for Theta {theta_input_str}...")
        stored_data = np.load(svd_data_path)
        
        # Check if the old file has the new U_means data
        if 'U_means' in stored_data:
            sing_vals_list = stored_data['sing_vals']
            V_list = stored_data['V_list']
            U_means_list = stored_data['U_means']
        else:
            print("⚠️ Old SVD file found without reconstruction data. Recomputing...")
            recompute_needed = True
    else:
        recompute_needed = True

    if recompute_needed:
        print(f"⚙️ Computing SVD evolution + Reconstruction coefficients for Theta {theta_input_str}...")
        many_rho_subset = many_rho[:N_traj_svd]
        # Use the updated Numba function that returns 3 outputs
        sing_vals_list, V_list, U_means_list = fast_svd_evolution(many_rho_subset)
        
        # Save including the new U_means array
        np.savez(svd_data_path, sing_vals=sing_vals_list, V_list=V_list, U_means=U_means_list)
        print(f"💾 SVD data (with U_means) saved to {svd_data_path}")

    # --- Step 2: Check/Generate GIF ---
    if os.path.exists(gif_path):
        print(f"✅ GIF already exists for Theta {theta_input_str}. Skipping generation.")
    else:
        print(f"🎬 Generating GIF for Theta {theta_input_str}...")
        create_bloch_svd_gif(
            sing_vals_list=sing_vals_list, 
            V_list=V_list, 
            N_traj=N_traj_svd, 
            N_time=many_rho.shape[1], 
            theta_str=theta_input_str, 
            frame_step=10, 
            frame_duration=0.5, 
            filename=gif_path
        )

    # --- Step 3: Always Update Analysis Plots ---
    # 1. The Cartesian components plot (the one we fixed before)
    plot_svd_components_evolution(
        time=time_stepped, 
        sing_vals=sing_vals_list, 
        V_list=V_list, 
        theta_str=theta_input_str, 
        output_path=analysis_plot_path
    )

    # 2. NEW: The Reconstruction Error plot (Comparing SVD vs Master Equation)
    plot_reconstruction_error(
        time=time_stepped,
        many_rho=many_rho[:N_traj_svd], # Use the same subset used for SVD
        sing_vals=sing_vals_list,
        V_list=V_list,
        U_means=U_means_list,
        theta_str=theta_input_str,
        output_path=error_plot_path
    )

    print(f"🏁 Full Analysis for Theta {theta_input_str} completed.")