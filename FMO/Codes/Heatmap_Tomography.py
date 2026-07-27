#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================
# Input Parsing from Bash
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 0.0  

dt = 1.0
N_traj = 10000

dt_str = f"{dt:.2f}".replace(".", "p")
theta_str = f"{theta_deg:.3f}".replace(".", "p")

results_dir = "../Results/Data/"
Output_dir = f"../Results/GIFs/{theta_str}"
os.makedirs(Output_dir, exist_ok=True)

fname = os.path.join(results_dir, f"result_FMO_theta{theta_str}_dt{dt_str}_Ntraj{N_traj}.npz")

try:
    data = np.load(fname)
    print(f"Loading data for Heatmap GIFs (Theta = {theta_deg} deg)...")
except FileNotFoundError:
    print(f"Error: File {fname} not found.")
    sys.exit(1)

times = data['times']
N_site = int(data['N_site'])
n_times = len(times)

# Estraiamo solo la matrice media (ignoriamo la Redfield per questo plot)
rho_traj_avg_site = data['rho_traj_avg_site']       # (n_times, N_site, N_site)

# Per la singola traiettoria, prendiamo la prima che ha registrato un jump
jump_counts = data['jump_counts']
psi_traj_exc = data['psi_traj']
eigenvectors = data['eigenvectors']

n_jumps_per_traj = jump_counts.sum(axis=0)
jump_indices = np.where(n_jumps_per_traj > 0)[0]
sample_idx = jump_indices[0] if len(jump_indices) > 0 else 0

psi_single_exc = psi_traj_exc[:, :, sample_idx] # (N_site, n_times)
psi_single_site = eigenvectors @ psi_single_exc # (N_site, n_times)

# Costruiamo la matrice di densità della singola traiettoria
rho_single_site = np.einsum('it,jt->tij', psi_single_site, np.conj(psi_single_site))

print(f"Data loaded. Single trajectory chosen: #{sample_idx} (Total jumps: {n_jumps_per_traj[sample_idx]})")

# ==========================================
# Impostazioni per l'Animazione
# ==========================================
target_frames = 150
stride = max(1, n_times // target_frames)
frames_to_plot = np.arange(0, n_times, stride)

# Velocità della GIF (più è basso, più è lenta)
fps_speed = 4

cmap = 'RdBu_r'
vmin, vmax = -1.0, 1.0  

SITE_TICKS = np.arange(N_site)
SITE_LABELS = [f"{i+1}" for i in range(N_site)]

def setup_heatmap_axis(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_xticks(SITE_TICKS)
    ax.set_yticks(SITE_TICKS)
    ax.set_xticklabels(SITE_LABELS)
    ax.set_yticklabels(SITE_LABELS)
    ax.tick_params(axis='both', which='major', labelsize=8)

# ==========================================
# GIF 1: Dinamica Average Trajectories (1x2 Grid)
# ==========================================
print("Generating GIF 1: Average Trajectories Dynamics...")

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
fig1.suptitle('', fontsize=14, y=0.98)

im_avg_re = axes1[0].imshow(np.zeros((N_site, N_site)), cmap=cmap, vmin=vmin, vmax=vmax)
im_avg_im = axes1[1].imshow(np.zeros((N_site, N_site)), cmap=cmap, vmin=vmin, vmax=vmax)

setup_heatmap_axis(axes1[0], "Avg MC: Real Part")
setup_heatmap_axis(axes1[1], "Avg MC: Imaginary Part")

fig1.colorbar(im_avg_im, ax=axes1.ravel().tolist(), shrink=0.8, pad=0.05)

def update_fig1(frame_idx):
    t = times[frame_idx]
    fig1.suptitle(f'Average Density Matrix (Theta={theta_deg}°)\nTime = {t:.1f} fs', fontsize=14)
    
    im_avg_re.set_data(np.real(rho_traj_avg_site[frame_idx]))
    im_avg_im.set_data(np.imag(rho_traj_avg_site[frame_idx]))
    return [im_avg_re, im_avg_im]

ani1 = animation.FuncAnimation(fig1, update_fig1, frames=frames_to_plot, blit=False)
path_gif1 = os.path.join(Output_dir, f"Heatmap_Avg_Theta_{theta_str}.mp4")
ani1.save(path_gif1, writer='ffmpeg', fps=fps_speed)
plt.close(fig1)
print(f"Saved: {path_gif1}")

# ==========================================
# GIF 2: Dinamica della Singola Traiettoria (1x2 Grid)
# ==========================================
print("Generating GIF 2: Single Trajectory Dynamics...")

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle('', fontsize=14, y=0.98)

im_sing_re = axes2[0].imshow(np.zeros((N_site, N_site)), cmap=cmap, vmin=vmin, vmax=vmax)
im_sing_im = axes2[1].imshow(np.zeros((N_site, N_site)), cmap=cmap, vmin=vmin, vmax=vmax)

setup_heatmap_axis(axes2[0], f"Traj #{sample_idx}: Real Part")
setup_heatmap_axis(axes2[1], f"Traj #{sample_idx}: Imaginary Part")

fig2.colorbar(im_sing_im, ax=axes2.ravel().tolist(), shrink=0.8, pad=0.05)

def update_fig2(frame_idx):
    t = times[frame_idx]
    fig2.suptitle(f'Single Trajectory Dynamics (Theta={theta_deg}°)\nTime = {t:.1f} fs', fontsize=14)
    
    im_sing_re.set_data(np.real(rho_single_site[frame_idx]))
    im_sing_im.set_data(np.imag(rho_single_site[frame_idx]))
    return [im_sing_re, im_sing_im]

ani2 = animation.FuncAnimation(fig2, update_fig2, frames=frames_to_plot, blit=False)
path_gif2 = os.path.join(Output_dir, f"Heatmap_SingleTraj_Theta_{theta_str}.mp4")
ani2.save(path_gif2, writer='ffmpeg', fps=fps_speed)
plt.close(fig2)
print(f"Saved: {path_gif2}")

print("All GIF generation completed!")