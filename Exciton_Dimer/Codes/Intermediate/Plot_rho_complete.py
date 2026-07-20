import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARSE COMMAND LINE ARGUMENTS
# =========================================================

# Argument 1: Theta (Angle)
if len(sys.argv) > 1:
    theta_target_deg = float(sys.argv[1])
else:
    theta_target_deg = 90.0

# Argument 2: Mode ('normal' or 'close_to_90')
if len(sys.argv) > 2:
    MODE = sys.argv[2]
else:
    MODE = 'normal'

print(f"\n🚀 STARTING POPULATION ANALYSIS FOR THETA = {theta_target_deg}° ({MODE})")

# ====================================
# Physical & Simulation Parameters
# ====================================
theta_rad = np.radians(theta_target_deg)

# Site selector: 0 for |10>, 1 for |01>
site_index = 0
dt = 0.01

# Number of trajectories for the ensemble average
N_traj_plot = 10000 

# =========================================================
# DIRECTORY CONFIGURATION
# =========================================================
if MODE == 'normal':
    Input_dir = "../Results/Data/Complete_rho/normal"
elif MODE == 'close_to_90':
    Input_dir = "../Results/Data/Complete_rho/close_90_deg"
else:
    raise ValueError(f"Unknown mode: {MODE}")

# Create an angle string for the folder name
angle_folder = str(int(theta_target_deg)) if theta_target_deg.is_integer() else str(theta_target_deg)

# Dynamically set the Output directory splitting by MODE
Output_dir = os.path.join("../Results/Plot/Populations", MODE, angle_folder)
os.makedirs(Output_dir, exist_ok=True)

# Global Style Settings
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.figsize': (10, 5), 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': ':', 'figure.autolayout': True
})

def save_fig(fig, filename):
    """Saves the generated matplotlib figure as a PNG file."""
    path_png = os.path.join(Output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")

# =================
# Input Data Setup
# =================
theta_str = f"{theta_rad:.6f}".replace(".", "p")
dt_str = f"{dt:.6f}".replace(".", "p")

filename = f"result_theta{theta_str}_dt{dt_str}_Ntraj20000.npz"
filepath = os.path.join(Input_dir, filename)

if not os.path.exists(filepath):
    print(f"ERROR: File {filepath} not found. Skipping angle {theta_target_deg}°...")
    sys.exit(0)

print("Extracting data matrices...")
data = np.load(filepath)
times = data['times']

# Analytical Baseline Curves
pop_iso = data['pop_traj_isolated']
pops_trace = data['pops_trace']

# Extract relevant diagonal elements from the 4x4 Lindblad matrix
rho_lindblad = data['rho_list_lindblad']
pop_lindblad_10 = np.real(rho_lindblad[:, 2, 2])
pop_lindblad_01 = np.real(rho_lindblad[:, 1, 1])

# Raw distribution matrices
raw_pop_10 = data['pop_00']
raw_pop_01 = data['pop_11']

# Calculate averages over N trajectories
avg_pop_10 = np.mean(raw_pop_10[:, :N_traj_plot], axis=1)
avg_pop_01 = np.mean(raw_pop_01[:, :N_traj_plot], axis=1)

# Extract first 50 single trajectories for visual background
single_trajs_10 = raw_pop_10[:, :50]
single_trajs_01 = raw_pop_01[:, :50]

print("Data extracted successfully!")

# ============================
# Site Selection and Labeling
# ============================
if site_index == 0:
    pop_lindblad = pop_lindblad_10
    avg_pop = avg_pop_10
    label_site = r"1 ($|10\rangle$)"
    single_trajs = single_trajs_10
else:
    pop_lindblad = pop_lindblad_01
    avg_pop = avg_pop_01
    label_site = r"2 ($|01\rangle$)"
    single_trajs = single_trajs_01

# =================================================
# Plot 1: Convergence Avg vs Trace vs Lindblad
# =================================================
plt.close('all')
fig01, ax = plt.subplots()
ax.plot(times, pop_lindblad, label=r'Lindblad', linewidth=2, linestyle='--')
ax.plot(times, pops_trace[site_index, :], label=r'Anc_trace', linewidth=2, linestyle=':')
ax.plot(times, avg_pop, label=r'Avg_traj', linewidth=2, alpha=0.5)

ax.set_title(f'Comparison Lindblad, Trace, Avg Trajectories | $\\theta$={theta_target_deg}°')
ax.set_xlabel('Time')
ax.set_ylabel(f'Population {label_site}')
ax.legend()

save_fig(fig01, f"Convergence_Avg_Trace_Lindblad_Theta_{theta_str}")

# =================================================
# Plot 2: Collisional vs Isolated Trajectories
# =================================================
plt.close('all')
fig02, ax = plt.subplots()
ax.plot(times, single_trajs[:, 0], label='Single Traj', linewidth=2)
ax.plot(times, pop_iso[site_index, :], label='Traj Isolated', linewidth=2, linestyle=':')

ax.set_title(f'Trajectories Collisional vs Isolated | $\\theta$={theta_target_deg}°')
ax.set_xlabel('Time')
ax.set_ylabel(f'Population {label_site}')
ax.legend()

save_fig(fig02, f"Collisional_vs_Isolated_Theta_{theta_str}")

# ===================================================
# Plot 3: Single Trajectories vs Average vs Lindblad
# ===================================================
plt.close('all')
fig03, ax = plt.subplots()
# Plot the background spaghetti trajectories
for i in range(single_trajs.shape[1]):
    ax.plot(times, single_trajs[:, i], color='gray', alpha=0.15, linewidth=0.5, 
             label='Single Traj' if i==0 else "")

ax.plot(times, pop_lindblad, label=r'Lindblad', linewidth=2, linestyle='--', color='blue')
ax.plot(times, avg_pop, label=r'Avg_traj', linewidth=2, color='red')

ax.set_title(f'Single Trajectories vs Average vs Lindblad | $\\theta$={theta_target_deg}°')
ax.set_xlabel('Time')
ax.set_ylabel(f'Population {label_site}')
ax.legend()

save_fig(fig03, f"SingleTraj_vs_Avg_vs_Lindblad_Theta_{theta_str}")

print(f"All Population plots successfully generated for {theta_target_deg}°!\n")

# Close datasets and clear memory
data.close()
import gc
gc.collect()