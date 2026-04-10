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
    Creates 3 subplots (one for each principal axis) showing x, y, z components over time.
    The line alpha is scaled by the relative intensity of the singular value.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    labels = ['x component', 'y component', 'z component']
    axis_names = ['1st Principal Axis', '2nd Principal Axis', '3rd Principal Axis']

    # Max singular value for alpha normalization
    max_s = np.max(sing_vals)

    for i in range(3): # Loop over the 3 principal axes
        ax = axes[i]
        # V_list shape: (N_time, 3_components, 3_axes)
        # We extract components for the i-th axis: V_list[:, :, i]
        x_comp = V_list[:, 0, i]
        y_comp = V_list[:, 1, i]
        z_comp = V_list[:, 2, i]
        
        # Current singular value for alpha (normalized to [0.1, 1.0])
        # Using a base alpha so the line is always slightly visible
        current_s = sing_vals[:, i]
        alphas = 0.1 + 0.9 * (current_s / max_s)

        ax.plot(time, x_comp, label='x', color='red', lw=1.5)
        ax.plot(time, y_comp, label='y', color='green', lw=1.5)
        ax.plot(time, z_comp, label='z', color='blue', lw=1.5)
        
        ax.set_title(f"{axis_names[i]} Evolution")
        ax.set_ylabel("Component Value")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', ncol=3)

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
    
    for t in range(n_time):
        for i in range(n_traj):
            rho = many_rho_subset[i, t]
            bloch_vec = NJIT_bloch_coords(rho) 
            vects[i, 0] = bloch_vec[0]
            vects[i, 1] = bloch_vec[1]
            vects[i, 2] = bloch_vec[2]
            
        _, S, Vh = np.linalg.svd(vects, full_matrices=False)
        sing_vals_list[t, :] = S
        
        for r in range(3):
            for c in range(3):
                V_list[t, c, r] = Vh[r, c]
                
    return sing_vals_list, V_list

print("Computing SVD evolution via Numba...")
many_rho_subset = many_rho[:N_traj_svd]
sing_vals_list, V_list = fast_svd_evolution(many_rho_subset)
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

# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    # --- Arguments Setup --- (Come nel tuo codice)
    if len(sys.argv) < 3:
        sys.exit(1)
    theta_input_str = sys.argv[1]
    MODE = sys.argv[2]
    # ... (Caricamento dati many_rho identico al tuo) ...

    # --- Path Definition ---
    output_dir = "../../Results/Bloch_Sphere/Densification/SVD_Analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    gif_path = os.path.join(output_dir, f"bloch_svd_evolution_theta_{theta_input_str}.gif")
    svd_data_path = os.path.join(output_dir, f"svd_data_theta_{theta_input_str}.npz")
    analysis_plot_path = os.path.join(output_dir, f"svd_components_theta_{theta_input_str}.png")

    # --- Step 1: Check/Compute SVD Data ---
    if os.path.exists(svd_data_path):
        print(f"📦 Loading existing SVD data for Theta {theta_input_str}...")
        stored_data = np.load(svd_data_path)
        sing_vals_list = stored_data['sing_vals']
        V_list = stored_data['V_list']
    else:
        print(f"⚙️ Computing SVD evolution via Numba for Theta {theta_input_str}...")
        many_rho_subset = many_rho[:N_traj_svd]
        sing_vals_list, V_list = fast_svd_evolution(many_rho_subset)
        # Save for future use
        np.savez(svd_data_path, sing_vals=sing_vals_list, V_list=V_list)
        print(f"💾 SVD data saved to {svd_data_path}")

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

    # --- Step 3: Always Generate/Update Analysis Plot ---
    plot_svd_components_evolution(
        time=time_stepped, 
        sing_vals=sing_vals_list, 
        V_list=V_list, 
        theta_str=theta_input_str, 
        output_path=analysis_plot_path
    )

    print(f"🏁 Analysis for Theta {theta_input_str} completed.")