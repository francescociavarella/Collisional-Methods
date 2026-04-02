import numpy as np
from numba import njit
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import imageio.v2 as imageio
import os

# ==========================================
# Custom Modules Import
# ==========================================

from densification import NJIT_syncr_measure_time,  NJIT_bloch_coords, NJIT_vectors_inCartesian_coords
import visualization 

# ====================================
# Physical & Simulation Parameters
# ====================================
# Theta angle in degrees, H_Coll Direction
theta_target_deg = 0.0  # change angle here
theta_rad = np.radians(theta_target_deg)

# Site selector: 0 for |10>, 1 for |01>
site_index = 0

# Time step
dt = 0.01

# Number of trajectories to analyze
N_traj_to_plot = 100         

# =================
# Input Data Setup
# =================
Input_dir = "../../Results/Data/Complete_rho/normal"  # <-- change here if needed

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
    time_downsample_factor = 1
    
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

@njit
def fast_svd_evolution(many_rho_subset):
    """
    Computes the SVD of Bloch vectors over time using preallocated arrays
    and compiling the entire time loop in C with Numba.
    """
    n_traj = many_rho_subset.shape[0]
    n_time = many_rho_subset.shape[1]
    
    # 1. Preallocate the final output arrays (much faster than list.append)
    sing_vals = np.empty((n_time, 3), dtype=np.float64)
    V_list = np.empty((n_time, 3, 3), dtype=np.float64)
    
    # 2. Preallocate the Cartesian vector matrix ONCE.
    # We will overwrite this memory at each time step, saving massive amounts of RAM and time.
    vects = np.empty((n_traj, 3), dtype=np.float64)
    
    for t in range(n_time):
        
        # Populate the vects array for the current time step
        for i in range(n_traj):
            rho = many_rho_subset[i, t]
            bloch_vec = NJIT_bloch_coords(rho) 
            vects[i, 0] = bloch_vec[0]
            vects[i, 1] = bloch_vec[1]
            vects[i, 2] = bloch_vec[2]
            
        # Perform SVD (Numba supports np.linalg.svd natively!)
        # Using full_matrices=False directly inside Numba
        _, S, Vh = np.linalg.svd(vects, full_matrices=False)
        
        # Store the singular values
        sing_vals[t, :] = S
        
        # Store the transposed Vh (which is V)
        # We manually transpose assigning elements to avoid memory copies in Numba
        for r in range(3):
            for c in range(3):
                V_list[t, c, r] = Vh[r, c]
                
    return sing_vals, V_list

# --- Execution ---

N_traj_svd = 10000

# Slice the trajectory array ONCE before passing it to the Numba function
many_rho_subset = many_rho[:N_traj_svd]

# Call the highly optimized function
sing_vals_list, V_list = fast_svd_evolution(many_rho_subset)

def create_bloch_svd_gif(sing_vals, V_list, N_traj, N_time, filename="bloch_svd_evolution.gif", frame_step=100, frame_duration=0.2):
    """
    Generates an animated GIF showing the time evolution of the SVD principal axes
    on the Bloch sphere.
    """
    filenames = []
    temp_dir = "temp_gif_frames"
    
    # Create a temporary directory to store the individual frames
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    print(f"Generating frames (1 every {frame_step} steps)...")
    
    # Loop through the time steps using the specified frame step
    for t_idx in range(0, N_time, frame_step):
        b = qt.Bloch()
        
        # Extract singular values and principal axes
        S = sing_vals[t_idx]
        V = V_list[t_idx]
        
        # Normalize singular values to fit the Bloch sphere (radius = 1)
        S_norm = S / np.sqrt(N_traj)
        
        # Scale the unit vectors by the normalized singular values
        vec_1 = V[:, 0] * S_norm[0]
        vec_2 = V[:, 1] * S_norm[1]
        vec_3 = V[:, 2] * S_norm[2]
        
        # Add the vectors to the sphere
        b.add_vectors([vec_1, vec_2, vec_3])
        b.vector_color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Render the sphere
        b.render()
        
        # Add a dynamic title showing the current time step
        plt.title(f"SVD Principal Axes | Time Step: {t_idx} / {N_time}")
        
        # Save the current frame as a temporary PNG file
        frame_filename = os.path.join(temp_dir, f"frame_{t_idx:05d}.png")
        plt.savefig(frame_filename, bbox_inches='tight', dpi=150)
        filenames.append(frame_filename)
        
        # Close the plot to free up memory
        plt.close()
        b.clear()

    print("Stitching frames into a GIF...")
    
    # Read the saved frames and compile them into an animated GIF
    # The 'duration' parameter controls the time (in seconds) each frame is displayed
    with imageio.get_writer(filename, mode='I', duration=frame_duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"GIF successfully saved as {filename}!")
    
    # Clean up: delete the temporary image files and the directory
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)

# --- Execution ---

# Define the output directory
output_dir = "../../Results/Bloch_Sphere/Densification/SVD_Analysis"

# Create the target directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Construct the full path for the GIF file
gif_path = os.path.join(output_dir, f"bloch_svd_evolution_theta_{theta_str}.gif")

# Generate the GIF:
# frame_step=100 -> takes 1 frame every 100 time steps (results in 100 frames total for N_time=10000)
# frame_duration=0.2 -> slows down the animation (0.2 seconds per frame = 5 FPS)
create_bloch_svd_gif(
    sing_vals=sing_vals, 
    V_list=V_list, 
    N_traj=10000, 
    N_time=N_time, 
    frame_step=10, 
    frame_duration=0.5, 
    filename=gif_path
)
