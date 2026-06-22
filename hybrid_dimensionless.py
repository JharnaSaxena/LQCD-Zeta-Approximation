#!/usr/bin/env python3
"""
DIMENSIONLESS HYBRID ZETA - FIXED GAMMA FOR Eq6
- q²<0: Use fixed γ from BOOSTS
- q²>0: Use γ from kinematics
"""

import numpy as np
import sys

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00

L = 48
SCALE_FACTOR = (L / (2 * np.pi))**2

# Fixed gamma for each PSQ (for Eq6 / q²<0)
FIXED_GAMMA = {'PSQ0': 1.0,'PSQ1': 1.1,'PSQ2': 1.15,'PSQ3': 1.2,'PSQ4': 1.3,}

BOOSTS = {
    'PSQ0': {'d': (0,0,0), 'name': 'Rest Frame'},
    'PSQ1': {'d': (0,0,1), 'name': 'Moving along z'},
    'PSQ2': {'d': (1,1,0), 'name': 'Diagonal in xy'},
    'PSQ3': {'d': (1,1,1), 'name': 'Space diagonal'},
    'PSQ4': {'d': (0,0,2), 'name': 'Moving along z (2x)'},
}

def qscaled_from_dimensionless(qstar2_over_mref, m_ref=1.0):
    qstar2 = qstar2_over_mref * m_ref**2
    return qstar2 * SCALE_FACTOR

def zeta_equation6(q2_scaled, gamma, d, mmax=6):
    if q2_scaled >= 0:
        return np.nan
    
    kappa = np.sqrt(-q2_scaled) * (2 * np.pi / L)
    kappa_L = kappa * L
    
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
                if d_norm > 1e-12:
                    m_parallel = np.dot(m, d_hat) * d_hat
                else:
                    m_parallel = np.zeros(3)
                m_perp = m - m_parallel
                gamma_m_hat = gamma * m_parallel + m_perp
                abs_gamma_m = np.linalg.norm(gamma_m_hat)
                if abs_gamma_m < 1e-12:
                    continue
                phase = np.cos(2.0 * np.pi * 0.5 * np.dot(m, d_arr))
                exp_factor = np.exp(-abs_gamma_m * kappa_L)
                summation += (1.0 / abs_gamma_m) * phase * exp_factor
    
    F_val = summation
    qcotd = (1.0 / L) * F_val - kappa
    Z_val = (gamma * L * np.sqrt(np.pi) / 2.0) * qcotd
    return Z_val

def exact_Z_dimensionless(qstar2_over_mref, d, gamma, m_ref=1.0):
    q2_scaled = qscaled_from_dimensionless(qstar2_over_mref, m_ref)
    try:
        return Z_00(q2_scaled, d=d, gamma=gamma).real
    except:
        return np.nan

print("DIMENSIONLESS HYBRID ZETA - FIXED GAMMA FOR Eq6")
print("q²<0: Use fixed γ | q²>0: Use γ from kinematics")

m_ref = 1.0

for psq, info in BOOSTS.items():
    d = info['d']
    name = info['name']
    gamma_fixed = FIXED_GAMMA[psq]
    
    print(f"\n{psq}: {name}")
    print(f"{'q*²/m_ref²':>14} {'q²_scaled':>14} {'γ':>12} {'Exact Z':>14} {'Status'}")
    
    test_vals = [-0.3, -0.1, 0.01, 0.05, 0.1]
    
    for qs2_dim in test_vals:
        q2_scaled = qscaled_from_dimensionless(qs2_dim, m_ref)
        
        if qs2_dim < 0:
            # q²<0: Use fixed gamma for Eq6
            gamma = gamma_fixed
            exact = zeta_equation6(q2_scaled, gamma_fixed, d, mmax=6)
            status = "Eq6"
        else:
            # q²>0: Use exact Z with fixed gamma (simplified)
            gamma = gamma_fixed
            exact = exact_Z_dimensionless(qs2_dim, d, gamma_fixed, m_ref)
            status = "Exact"
        
        if not np.isnan(exact) and abs(exact) < 1e6:
            print(f"{qs2_dim:14.4f} {q2_scaled:14.2f} {gamma:12.6f} {exact:14.6f} {status}")
        else:
            print(f"{qs2_dim:14.4f} {q2_scaled:14.2f} {gamma:12.6f} {'FAILED':>14} {status}")
