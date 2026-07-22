import numpy as np
from scipy.linalg import expm
from qutip import *
import numba
from numba import njit, prange
import os
import time
import sys

sz = np.array(([[1,0], [0,-1]]), dtype=complex); sx = np.array(([[0,1],[1,0]]), dtype=complex); sy = np.array(([[0,-1j],[1j,0]]), dtype=complex) ; sm = np.array(([[0.0, 1.0],[0.0,0.0]]), dtype=complex) ; sp = np.array(([[0.0,0.0],[1.0,0.0]]), dtype=complex)

# ===================================================
# Environment SPectral Function : Drude-Lorentz Bath
# ===================================================

def check_KMS(C_func, omega_test, beta, rtol=1e-8):
    """
    Sanity check of the KMS relation C(-omega) = C(omega) * exp(-beta*omega).
    Returns the relative error (should be ~0).
    """
    lhs = C_func(-omega_test)
    rhs = C_func(omega_test) * np.exp(-beta * omega_test)
    return np.abs(lhs - rhs) / (np.abs(rhs) + 1e-30)


def bose_factor_stable(x):
    """
    Numerically stable computation of 1/(1 - exp(-x)) for any real x
    (avoids overflow for large |x|).
    """
    scalar_input = np.isscalar(x) or np.asarray(x).ndim == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)

    pos = x > 0
    out[pos] = 1.0 / (1.0 - np.exp(-x[pos]))

    neg = ~pos
    ex = np.exp(x[neg])          # x[neg] <= 0, so ex in (0,1], safe
    out[neg] = ex / (ex - 1.0)

    return out[0] if scalar_input else out


def C_drude_lorentz(omega, lam, Omega, beta, omega_tol=1e-10):
    """
    Drude-Lorentz spectral function C(omega) = 4*lam*omega*Omega/(omega^2+Omega^2) * 1/(1-exp(-omega*beta)).
    Handles the omega -> 0 limit analytically, and uses a numerically stable
    Bose factor to avoid overflow for large |omega|.
    """
    scalar_input = np.isscalar(omega) or np.asarray(omega).ndim == 0
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.empty_like(omega)

    small = np.abs(omega) < omega_tol
    out[small] = 4.0 * lam / (Omega * beta)

    w = omega[~small]
    if w.size > 0:
        lorentz = 4.0 * lam * w * Omega / (w**2 + Omega**2)
        out[~small] = lorentz * bose_factor_stable(w * beta)

    return out[0] if scalar_input else out


def lamb_shift_Lambda(omega, C_func, bound=500.0, limit=500):
    """
    Computes Lambda(omega) = (1/2pi) * P.V. Integral[ C(omega')/(omega-omega'), d omega' ]
    via scipy.integrate.quad with a Cauchy principal-value weight.
    """
    integrand = lambda wp: float(np.real(C_func(wp)))

    pv_result, _ = quad(integrand, -bound, bound, weight='cauchy', wvar=omega,
                         limit=limit)   # 'points' removed, incompatible with weight='cauchy'

    return -pv_result / (2.0 * np.pi)


# ====================
# Physical parameters
# ====================

# ---------------------------------------------------
# Physical constants and unit conversion (hbar = 1)
# ---------------------------------------------------
c_light_cm_fs = 2.99792458e-5          # speed of light, cm/fs
cm1_to_radfs = 2.0 * np.pi * c_light_cm_fs   # convert cm^-1 -> rad/fs
KB_cm1_per_K = 0.695034800            # Boltzmann constant, cm^-1/K
beta = 1.0 / (KB_cm1_per_K * T_kelvin * cm1_to_radfs)  # inverse temperature, fs/rad

# ---------------------------------------------------
# FMO exciton Hamiltonian (Adolphs & Renger), cm^-1
# ---------------------------------------------------
H_exc_cm1 = np.array([
    [200,  -96,    5,  -4.4,   4.7, -12.6,  -6.2],
    [-96,  320, 33.1,   6.8,   4.5,   7.4,  -0.3],
    [5,   33.1,    0, -51.1,   0.8,  -8.4,   7.6],
    [-4.4,  6.8,-51.1,   110, -76.6, -14.2,   -67],
    [4.7,   4.5,  0.8, -76.6,   270,  78.3,  -0.1],
    [-12.6, 7.4, -8.4, -14.2,  78.3,   420,  38.3],
    [-6.2, -0.3,  7.6,   -67,  -0.1,  38.3,   230]
], dtype=complex)

N_site = H_exc_cm1.shape[0]

# Convert to rad/fs: from now on all energies/rates are in fs^-1
H_exc = H_exc_cm1 * cm1_to_radfs

# Eigenerengies and eigenvectors of the exciton Hamiltonian
eigenergies, eigenvectors = np.linalg.eigh(H_exc)