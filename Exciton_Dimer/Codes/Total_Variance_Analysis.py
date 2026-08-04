#!/usr/bin/env python
# coding: utf-8
"""
Law-of-Total-Variance analysis for the excitonic dimer (full ground+excited
basis, dim=4), restricted to the single-exciton manifold {|10>, |01>}, matching
the data saved by the trajectory-generation script (pop_00, pop_11, coh_01_10,
coh_10_01, rho_list_lindblad, pops_trace, ...).

Data layout notes (see chat for details):
- pop_00 = population of |10> (projectors[0] = P_10), pop_11 = population of
  |01> (projectors[1] = P_01). The variable names in the generation script are
  misleading; state labels below use the physical |10>/|01> notation instead.
- coh_01_10 and coh_10_01 are complex conjugates of each other (same physical
  information); only coh_01_10 is used here.
- Only the two single-exciton populations and their coherence were saved, not
  the full 4x4 density matrix per trajectory - so (unlike the previous two
  scripts) there is no full trace-distance/fidelity convergence check
  possible. Instead we check convergence of the trajectory-averaged MEANS
  against the exact Lindblad expectation values for the same observables,
  which is the check the available data actually supports.

Reference for the "exact" Total Variance: rho_list_lindblad (n_times, 4, 4),
the full-basis master-equation solution. pops_trace (populations from tracing
out the ancilla in the full collisional Hamiltonian) is used as an
independent secondary cross-check on the populations only.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# ==========================================
# Operators in the full 4-dim (ground+excited) basis - copied verbatim from
# the generation script's conventions to guarantee consistent definitions.
# ==========================================
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

P0 = (np.eye(2, dtype=complex) + sz) / 2  # projector on |0>
P1 = (np.eye(2, dtype=complex) - sz) / 2  # projector on |1>

P_10 = np.kron(P1, P0)       # |10><10|
P_01 = np.kron(P0, P1)       # |01><01|
P_01_10 = np.kron(sp, sm)    # |01><10|
P_10_01 = np.kron(sm, sp)    # |10><01|

O_real = P_01_10 + P_10_01                 # |01><10| + |10><01|  =  sigma_x on {|10>,|01>}
O_imag = -1j * P_01_10 + 1j * P_10_01       # -i|01><10| + i|10><01|  =  sigma_y on {|10>,|01>}
sigma_z_op = P_01 - P_10                    # sigma_z on {|10>,|01>} (matches compute_Bloch_Sphere's r_z convention)

# ==========================================
# EXACT VARIANCE FROM THE MASTER-EQUATION DENSITY MATRIX
# ==========================================

def get_exact_variance(observable_matrix, rho_t):
    """Var(O) = Tr(O^2 rho) - Tr(O rho)^2 . rho_t : (n_times, N, N)."""
    observable_sq = observable_matrix @ observable_matrix
    E_O = np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))
    E_O2 = np.real(np.einsum('ik,tki->t', observable_sq, rho_t))
    return np.maximum(E_O2 - E_O ** 2, 0.0)


def get_exact_expectation(observable_matrix, rho_t):
    """<O>(t) = Tr(O rho(t)). rho_t : (n_times, N, N)."""
    return np.real(np.einsum('ik,tki->t', observable_matrix, rho_t))


# ==========================================
# FAST ANALYTIC LAW-OF-TOTAL-VARIANCE
# (populations and coherence are already given directly per trajectory here -
# no extraction from a stored full density matrix needed, see chat)
# ==========================================

def total_variance_projector(pop_k):
    """pop_k : (n_times, n_traj). O^2=O for a projector -> quantum var = p(1-p)."""
    var_quant_k = pop_k * (1.0 - pop_k)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(pop_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


def total_variance_coherence(coh_k, pop_m_k, pop_n_k, part='real'):
    """
    coh_k = <psi|n><m|psi> = rho_{m,n} (as saved in coh_01_10), shape (n_times, n_traj).
    Using O_real^2 = O_imag^2 = |m><m|+|n><n| (see FMO script for derivation):
        <O_real> = 2 Re(coh_k) ,  <O_imag> = -2 Im(coh_k) ,  <O^2> = p_m + p_n
    """
    E_k = 2.0 * np.real(coh_k) if part == 'real' else -2.0 * np.imag(coh_k)
    return total_variance_qubit_observable(E_k, pop_m_k, pop_n_k)


def total_variance_qubit_observable(E_k, pop_m_k, pop_n_k):
    """
    Generic LTV decomposition for ANY Pauli-like observable (sigma_x, sigma_y,
    sigma_z) on the 2-level subspace {|m>,|n>}. Every Pauli operator squares to
    the identity restricted to that subspace, O^2 = P_m + P_n, so <O^2> = p_m+p_n
    exactly regardless of which Pauli component E_k represents - the same
    identity used for the coherence quadratures above, just made explicit here
    so it can be reused for sigma_z (population difference) too.
    """
    E2_k = pop_m_k + pop_n_k
    var_quant_k = np.maximum(E2_k - E_k ** 2, 0.0)
    var_quant = np.mean(var_quant_k, axis=1)
    var_stat = np.var(E_k, axis=1)
    return var_quant + var_stat, var_quant, var_stat


# ==========================
# PLOTTING HELPERS
# ==========================

def save_fig(fig, filename, output_dir):
    path_png = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {path_png}")
    plt.close(fig)


def plot_ltv_panel(ax, times, var_exact, var_stat, var_quant, var_sum, ylabel,
                    show_legend=False, title=None):
    ax.plot(times, var_exact, color='black', linewidth=3, linestyle='--', label='Total Variance (Lindblad)')
    ax.plot(times, var_stat, color='red', linewidth=2, alpha=0.8, label='Statistical Variance')
    ax.plot(times, var_quant, color='blue', linewidth=2, alpha=0.8, label='Quantum Variance')
    ax.plot(times, var_sum, color='limegreen', linewidth=3, linestyle=':', label='Sum (Stat + Quant)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right', fontsize=8)


# ==========================
# Input parsing & file location (mirrors the generation script exactly)
# ==========================
if len(sys.argv) > 1:
    theta_deg = float(sys.argv[1])
else:
    theta_deg = 90.0

# --- Must match the values used in the trajectory-generation script ---
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
N_traj = 20000

theta_rad = np.radians(theta_deg)

# NOTE: the generation script writes to one of two subfolders depending on
# MODE ("normal" or "close_90_deg"). Adjust this line if you generated
# data with MODE="normal" instead.
results_dir = "../Results/Data/Complete_rho/close_90_deg/"

Output_dir = f"../Results/Plot/Total_Variance_Analysis/{theta_deg}"
os.makedirs(Output_dir, exist_ok=True)


def _make_fname_npz(results_dir, theta, dt, N_traj):
    """Copied verbatim from the generation script - note theta here is in RADIANS."""
    t_str = f"{theta:.6f}".replace(".", "p")
    dt_str = f"{dt:.6f}".replace(".", "p")
    return os.path.join(results_dir, f"result_theta{t_str}_dt{dt_str}_Ntraj{N_traj}.npz")


theta_str = f"{theta_deg:.4f}".replace(".", "p")  # used only in filenames below, not in Output_dir
fname = _make_fname_npz(results_dir, theta_rad, dt, N_traj)

try:
    data = np.load(fname)
    print(f"Data extraction completed successfully for Theta = {theta_deg} deg ({theta_rad:.4f} rad)")
except FileNotFoundError:
    print(f"Error: File {fname} not found. Ensure the simulation for this angle has completed.")
    sys.exit(1)

# ==========================
# Data extraction
# ==========================
times = data['times']
pop_10 = data['pop_00']            # population of |10> (see naming note above)
pop_01 = data['pop_11']            # population of |01>
coh_01_10 = data['coh_01_10']      # <psi|01><10|psi> = rho_{10,01}(t) per trajectory
rho_list_lindblad = data['rho_list_lindblad']  # (n_times, 4, 4) - exact reference
pops_trace = data['pops_trace']    # (2, n_times) - independent collision-model population cross-check

n_times = len(times)
n_traj = pop_10.shape[1]

# ==========================
# Statistical moments over trajectories (single-exciton populations)
# ==========================
print("Computing Statistical Moments over time...")
pop_stack = np.stack([pop_10, pop_01], axis=0)  # (2, n_times, n_traj)
STATE_LABELS = ['Population |10>', 'Population |01>']
colors = ['tab:blue', 'tab:orange']

mean_pop_time = np.mean(pop_stack, axis=2)
var_pop_time = np.var(pop_stack, axis=2)
skew_pop_time = skew(pop_stack, axis=2, nan_policy='omit')
kurt_pop_time = kurtosis(pop_stack, axis=2, fisher=True, nan_policy='omit')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':',
    'figure.autolayout': True
})

fig1, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax_mean, ax_var, ax_skew, ax_kurt = axes
for i in range(2):
    ax_mean.plot(times, mean_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_var.plot(times, var_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_skew.plot(times, skew_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])
    ax_kurt.plot(times, kurt_pop_time[i, :], color=colors[i], linewidth=2, label=STATE_LABELS[i])

ax_mean.set_title(f'Statistical Moments over Trajectories (Theta = {theta_deg}°)')
ax_mean.set_ylabel('Mean')
ax_mean.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax_var.set_ylabel('Variance')
ax_skew.set_ylabel('Skewness ($\\gamma_1$)')
ax_skew.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_skew.set_ylim(-5, 5)
ax_kurt.set_ylabel('Excess Kurtosis ($K - 3$)')
ax_kurt.set_xlabel('Time')
ax_kurt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_kurt.set_ylim(-5, 15)
save_fig(fig1, f'All_Statistical_Moments_Time_Theta_{theta_str}', Output_dir)

# ==========================
# PLOT 2: All Statistical Moments over Time - Expectation Values (sigma_x, sigma_y, sigma_z)
# ==========================
print("Computing Statistical Moments over time (expectation values)...")
Ex_k = 2.0 * np.real(coh_01_10)   # <sigma_x>_k per trajectory
Ey_k = -2.0 * np.imag(coh_01_10)  # <sigma_y>_k per trajectory
Ez_k = pop_01 - pop_10            # <sigma_z>_k per trajectory

exp_stack = np.stack([Ex_k, Ey_k, Ez_k], axis=0)  # (3, n_times, n_traj)
EXP_LABELS = ['$\\langle\\sigma_x\\rangle$', '$\\langle\\sigma_y\\rangle$', '$\\langle\\sigma_z\\rangle$']
exp_colors = ['tab:red', 'tab:green', 'tab:purple']

mean_exp_time = np.mean(exp_stack, axis=2)
var_exp_time = np.var(exp_stack, axis=2)
skew_exp_time = skew(exp_stack, axis=2, nan_policy='omit')
kurt_exp_time = kurtosis(exp_stack, axis=2, fisher=True, nan_policy='omit')

fig2, axes2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
ax_mean2, ax_var2, ax_skew2, ax_kurt2 = axes2
for i in range(3):
    ax_mean2.plot(times, mean_exp_time[i, :], color=exp_colors[i], linewidth=2, label=EXP_LABELS[i])
    ax_var2.plot(times, var_exp_time[i, :], color=exp_colors[i], linewidth=2, label=EXP_LABELS[i])
    ax_skew2.plot(times, skew_exp_time[i, :], color=exp_colors[i], linewidth=2, label=EXP_LABELS[i])
    ax_kurt2.plot(times, kurt_exp_time[i, :], color=exp_colors[i], linewidth=2, label=EXP_LABELS[i])

ax_mean2.set_title(f'Statistical Moments over Trajectories - Expectation Values (Theta = {theta_deg}°)')
ax_mean2.set_ylabel('Mean')
ax_mean2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax_var2.set_ylabel('Variance')
ax_skew2.set_ylabel('Skewness ($\\gamma_1$)')
ax_skew2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_skew2.set_ylim(-5, 5)
ax_kurt2.set_ylabel('Excess Kurtosis ($K - 3$)')
ax_kurt2.set_xlabel('Time')
ax_kurt2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax_kurt2.set_ylim(-5, 15)
save_fig(fig2, f'All_Statistical_Moments_ExpVal_Theta_{theta_str}', Output_dir)

# ==========================
# Mean-convergence check (trajectory avg vs Lindblad exact), restricted to the
# observables actually available (no full rho per trajectory was saved here)
# ==========================
print("Computing Mean Convergence Check: Trajectory Average vs Lindblad...")
traj_avg_pop10 = np.mean(pop_10, axis=1)
traj_avg_pop01 = np.mean(pop_01, axis=1)
traj_avg_coh_re = np.mean(2.0 * np.real(coh_01_10), axis=1)
traj_avg_coh_im = np.mean(-2.0 * np.imag(coh_01_10), axis=1)

exact_pop10 = get_exact_expectation(P_10, rho_list_lindblad)
exact_pop01 = get_exact_expectation(P_01, rho_list_lindblad)
exact_coh_re = get_exact_expectation(O_real, rho_list_lindblad)
exact_coh_im = get_exact_expectation(O_imag, rho_list_lindblad)

fig_conv, axes_conv = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
labels_conv = ['Population |10>', 'Population |01>', 'Re(coherence)', 'Im(coherence)']
pairs_conv = [(exact_pop10, traj_avg_pop10), (exact_pop01, traj_avg_pop01),
              (exact_coh_re, traj_avg_coh_re), (exact_coh_im, traj_avg_coh_im)]
for ax, lbl, (exact, traj) in zip(axes_conv, labels_conv, pairs_conv):
    ax.plot(times, exact, color='black', linewidth=2.5, linestyle='--', label='Lindblad (exact)')
    ax.plot(times, traj, color='crimson', linewidth=1.5, alpha=0.85, label='Trajectory average')
    ax.set_ylabel(lbl)
    if ax is axes_conv[0]:
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'Mean Convergence: Trajectory Average vs Lindblad (Theta = {theta_deg}°)')
axes_conv[-1].set_xlabel('Time')
fig_conv.tight_layout()
save_fig(fig_conv, f'Mean_Convergence_TrajAvg_vs_Lindblad_Theta_{theta_str}', Output_dir)

# Secondary cross-check: collision-model populations (pops_trace) vs Lindblad populations
print("Computing Population Cross-Check: Collision-Model Trace vs Lindblad...")
fig_cc, axes_cc = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes_cc[0].plot(times, exact_pop10, color='black', linewidth=2.5, linestyle='--', label='Lindblad (exact)')
axes_cc[0].plot(times, pops_trace[0, :], color='seagreen', linewidth=1.5, alpha=0.85, label='Collision-model trace')
axes_cc[0].set_ylabel('Population |10>')
axes_cc[0].legend(loc='best', fontsize=8)
axes_cc[0].set_title(f'Collision-Model Trace vs Lindblad - Populations (Theta = {theta_deg}°)')
axes_cc[1].plot(times, exact_pop01, color='black', linewidth=2.5, linestyle='--')
axes_cc[1].plot(times, pops_trace[1, :], color='seagreen', linewidth=1.5, alpha=0.85)
axes_cc[1].set_ylabel('Population |01>')
axes_cc[1].set_xlabel('Time')
fig_cc.tight_layout()
save_fig(fig_cc, f'Population_CrossCheck_CollisionTrace_vs_Lindblad_Theta_{theta_str}', Output_dir)

# ==========================
# A. Population LTV (single-exciton states |10>, |01>)
# ==========================
print("Computing Total Variance Theorem: Populations...")
fig_pop, axes_pop = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, (pop_traj, O_pop, label) in enumerate([(pop_10, P_10, 'Population |10>'),
                                               (pop_01, P_01, 'Population |01>')]):
    var_tot_traj, var_quant, var_stat = total_variance_projector(pop_traj)
    var_tot_exact = get_exact_variance(O_pop, rho_list_lindblad)

    max_err = np.max(np.abs(var_tot_exact - var_tot_traj))
    print(f"  -> {label} Law of Total Variance max error: {max_err:.2e}")

    plot_ltv_panel(
        axes_pop[i], times, var_tot_exact, var_stat, var_quant, var_tot_traj,
        ylabel=label, show_legend=(i == 0),
        title=f'Law of Total Variance - Populations (Theta = {theta_deg}°)' if i == 0 else None,
    )
axes_pop[-1].set_xlabel('Time')
fig_pop.tight_layout()
save_fig(fig_pop, f'Law_Total_Variance_Populations_Theta_{theta_str}', Output_dir)

# ==========================
# B. Coherence LTV (|10> <-> |01>, real & imaginary parts)
# ==========================
print("Computing Total Variance Theorem: Coherence...")
vt_real, vq_real, vs_real = total_variance_coherence(coh_01_10, pop_10, pop_01, part='real')
vt_imag, vq_imag, vs_imag = total_variance_coherence(coh_01_10, pop_10, pop_01, part='imag')

vt_real_exact = get_exact_variance(O_real, rho_list_lindblad)
vt_imag_exact = get_exact_variance(O_imag, rho_list_lindblad)

err_real = np.max(np.abs(vt_real_exact - vt_real))
err_imag = np.max(np.abs(vt_imag_exact - vt_imag))
print(f"  -> Coherence |10><01| | Max Error - Real: {err_real:.2e}, Imag: {err_imag:.2e}")

fig_coh, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
plot_ltv_panel(ax_re, times, vt_real_exact, vs_real, vq_real, vt_real,
               ylabel='Re( coherence )', show_legend=True,
               title=f'Law of Total Variance - Coherence |10><01| (Theta = {theta_deg}°)')
plot_ltv_panel(ax_im, times, vt_imag_exact, vs_imag, vq_imag, vt_imag,
               ylabel='Im( coherence )')
ax_im.set_xlabel('Time')
fig_coh.tight_layout()
save_fig(fig_coh, f'Law_Total_Variance_Coherence_Theta_{theta_str}', Output_dir)

# ==========================
# C. Bloch vector LTV (sigma_x, sigma_y, sigma_z on the {|10>,|01>} qubit)
#
# sigma_x = O_real, sigma_y = O_imag: these are the SAME operators as the
# coherence section above (Re/Im coherence quadratures ARE <sigma_x>/<sigma_y>
# on this subspace), so their LTV decomposition is just reused, not
# recomputed. sigma_z = P_01 - P_10 is new. All three square to the subspace
# identity P_10+P_01, so E2_k = pop_10+pop_01 in every case.
# ==========================
print("Computing Total Variance Theorem: Bloch Vector (sigma_x, sigma_y, sigma_z)...")

vt_z, vq_z, vs_z = total_variance_qubit_observable(Ez_k, pop_10, pop_01)
vt_z_exact = get_exact_variance(sigma_z_op, rho_list_lindblad)

err_z = np.max(np.abs(vt_z_exact - vt_z))
print(f"  -> sigma_z Law of Total Variance max error: {err_z:.2e}")

fig_bloch, axes_bloch = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
plot_ltv_panel(axes_bloch[0], times, vt_real_exact, vs_real, vq_real, vt_real,
               ylabel='$\\langle\\sigma_x\\rangle$', show_legend=True,
               title=f'Law of Total Variance - Bloch Vector (Theta = {theta_deg}°)')
plot_ltv_panel(axes_bloch[1], times, vt_imag_exact, vs_imag, vq_imag, vt_imag,
               ylabel='$\\langle\\sigma_y\\rangle$')
plot_ltv_panel(axes_bloch[2], times, vt_z_exact, vs_z, vq_z, vt_z,
               ylabel='$\\langle\\sigma_z\\rangle$')
axes_bloch[-1].set_xlabel('Time')
fig_bloch.tight_layout()
save_fig(fig_bloch, f'Law_Total_Variance_Bloch_Theta_{theta_str}', Output_dir)