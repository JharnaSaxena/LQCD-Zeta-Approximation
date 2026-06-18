#!/usr/bin/env python3
"""
Implementation of the finite-volume modification function F^(d)(κL)
for the bound-state region q*² < 0.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00

def F_function(kappa_L, gamma, d, alpha=0.5, mmax=5):
    d_arr = np.array(d, dtype=float)
    d_norm = np.linalg.norm(d_arr)
    if d_norm > 1e-12:
        d_hat = d_arr / d_norm
    else:
        d_hat = np.zeros(3)
    
    summation = 0.0
    for mx in range(-mmax, mmax+1):
        for my in range(-mmax, mmax+1):
            for mz in range(-mmax, mmax+1):
                m = np.array([mx, my, mz])
                if np.all(m == 0):
                    continue
                
                # Compute γ̂m = γ m_parallel + m_perp
                if d_norm > 1e-12:
                    m_parallel = np.dot(m, d_hat) * d_hat
                else:
                    m_parallel = np.zeros(3)
                m_perp = m - m_parallel
                gamma_m_hat = gamma * m_parallel + m_perp
                abs_gamma_m = np.linalg.norm(gamma_m_hat)
                
                if abs_gamma_m < 1e-12:
                    continue
                
                phase = np.cos(2.0 * np.pi * alpha * np.dot(m, d_arr))
                exp_factor = np.exp(-abs_gamma_m * kappa_L)
                summation += (1.0 / abs_gamma_m) * phase * exp_factor
    
    return summation

def zeta_from_F(q2_scaled, gamma, d, L=48, alpha=0.5, mmax=5):
    """
    Compute Z from Equation (6):
    Z = (γ L √π / 2) * [ (1/L) F^{(d)}(κL) - κ ]
    
    where κ = sqrt(-q*²) with q* = (2π/L) * sqrt(q2_scaled)
    """
    if q2_scaled >= 0:
        return np.nan
    
    kappa = np.sqrt(-q2_scaled) * (2 * np.pi / L)  # κ = sqrt(-q*²) in physical units
    kappa_L = kappa * L  # κL dimensionless
    
    # Compute F^{(d)}
    F_val = F_function(kappa_L, gamma, d, alpha, mmax)
    # Using Eq. (6):
    # q*cot(delta) = F^(d)(κL)/L - κ
    # Combined with Eq. (1):
    # Z00 = (γ L √π / 2) q*cot(delta)    
    qcotd = (1.0 / L) * F_val - kappa
    Z_val = (gamma * L * np.sqrt(np.pi) / 2.0) * qcotd
    
    return Z_val

def exact_Z_scaled(q2_scaled, d, gamma):
    try:
        return Z_00(q2_scaled, d=d, gamma=gamma).real
    except Exception:
        return np.nan

print("Comparing Eq. (6) with direct Z00 evaluation")

# Test parameters
d = (0, 0, 0)
gamma = 1.0
L = 48
alpha = 0.5

# Test q*² values (negative, physical)
qstar2_tests = [-0.5, -0.3, -0.1]

print(f"\n{'q*²':>10} {'q²_scaled':>12} {'Exact Z':>12} {'Eq (6) Z':>12} {'Error':>12}")

for qstar2 in qstar2_tests:
    q2_scaled = qstar2 * (L / (2 * np.pi))**2
    exact = exact_Z_scaled(q2_scaled, d, gamma)
    eq6 = zeta_from_F(q2_scaled, gamma, d, L, alpha, mmax=5)
    
    if not np.isnan(exact) and not np.isnan(eq6):
        err = abs(exact - eq6)
        print(f"{qstar2:10.2f} {q2_scaled:12.2f} {exact:12.6f} {eq6:12.6f} {err:12.6e}")
    else:
        print(f"{qstar2:10.2f} {q2_scaled:12.2f} {'FAILED':>12} {'FAILED':>12} {'---':>12}")
print("\nComparison complete.")
