#!/usr/bin/env python3
"""
ZOOMED HYBRID: Eq6 for q²<0 | Complex for q²>0
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00

L = 48
SCALE_FACTOR = (L / (2 * np.pi))**2

def qstar2_to_qscaled(qstar2):
    return qstar2 * SCALE_FACTOR

def exact_Z_scaled(q2_scaled, d, gamma):
    try:
        return Z_00(q2_scaled, d=d, gamma=gamma).real
    except:
        return np.nan

# Eq. (6) for q² < 0
def F_function(kappa_L, gamma, d, alpha=0.5, mmax=6):
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
                
                phase = np.cos(2.0 * np.pi * alpha * np.dot(m, d_arr))
                exp_factor = np.exp(-abs_gamma_m * kappa_L)
                summation += (1.0 / abs_gamma_m) * phase * exp_factor
    return summation

def zeta_equation6(q2_scaled, gamma, d, L=48, alpha=0.5, mmax=6):
    if q2_scaled >= 0:
        return np.nan
    
    kappa = np.sqrt(-q2_scaled) * (2 * np.pi / L)
    kappa_L = kappa * L
    F_val = F_function(kappa_L, gamma, d, alpha, mmax)
    qcotd = (1.0 / L) * F_val - kappa
    Z_val = (gamma * L * np.sqrt(np.pi) / 2.0) * qcotd
    return Z_val

def build_complex_approx(d, gamma, deg=6):
    from rebuild_real import get_poles, measure_residue
    q2_min, q2_max = -0.5, 6.0
    poles = get_poles(d, gamma, q2_max)
    poles = [p for p in poles if q2_min < p < q2_max]
    
    residues = {}
    for pole in poles:
        residues[pole] = measure_residue(pole, d, gamma)
    
    intervals = [(q2_min, poles[0])] if poles else [(q2_min, q2_max)]
    for i in range(len(poles)-1):
        intervals.append((poles[i], poles[i+1]))
    if poles and poles[-1] < q2_max:
        intervals.append((poles[-1], q2_max))
    
    complex_data = []
    for a, b in intervals:
        if b - a < 0.05:
            continue
        margin = 0.03 * (b - a)
        q2_vals = np.linspace(a + margin, b - margin, 200)
        z_exact = np.array([exact_Z_scaled(q, d, gamma) for q in q2_vals])
        
        pole_correction = np.zeros_like(q2_vals)
        if a in residues:
            pole_correction += residues[a] / (q2_vals - a)
        if b in residues:
            pole_correction += residues[b] / (q2_vals - b)
        residual = z_exact - pole_correction
        
        mid = (a + b) / 2.0
        half = (b - a) / 2.0
        q2_norm = (q2_vals - mid) / half
        coeffs = np.polyfit(q2_norm, residual, deg)
        
        complex_data.append({'a': a, 'b': b, 'mid': mid, 'half': half,'coeffs': coeffs.tolist(),'residues': {'left': residues.get(a, 0.0), 'right': residues.get(b, 0.0)}})
    return complex_data

def eval_complex(q2_scaled, complex_data):
    for iv in complex_data:
        if iv['a'] <= q2_scaled <= iv['b']:
            x = (q2_scaled - iv['mid']) / iv['half']
            poly = np.polyval(iv['coeffs'], x)
            pole_corr = 0.0
            if abs(q2_scaled - iv['a']) > 1e-10:
                pole_corr += iv['residues']['left'] / (q2_scaled - iv['a'])
            if abs(q2_scaled - iv['b']) > 1e-10:
                pole_corr += iv['residues']['right'] / (q2_scaled - iv['b'])
            return poly + pole_corr
    return np.nan

def print_verification_table(psq, qstar2_points, z_exact, z_hybrid, method):
    print(f"\n{'='*90}")
    print(f"VERIFICATION TABLE: {psq}")
    print(f"{'='*90}")
    print(f"{'q*² (phys)':>12} {'Method':>14} {'Exact Z':>14} {'Hybrid Z':>14} {'|Error|':>14} {'Status':>10}")
    print("-" * 90)
    
    for i, qstar2 in enumerate(qstar2_points):
        exact = z_exact[i] if not np.isnan(z_exact[i]) else np.nan
        hybrid = z_hybrid[i] if not np.isnan(z_hybrid[i]) else np.nan
        method = method[i]
        
        if not np.isnan(exact) and not np.isnan(hybrid):
            err = abs(exact - hybrid)
            if err < 1e-3:
                status = "Success"
            elif err < 0.1:
                status = "Warning"
            else:
                status = "Failure"
            print(f"{qstar2:12.4f} {method:>14} {exact:14.6f} {hybrid:14.6f} {err:14.6e} {status:>10}")
        else:
            print(f"{qstar2:12.4f} {'FAILED':>14} {exact:14.6f} {'FAILED':>14} {'---':>14} {'---':>10}")

BOOSTS = {
    'PSQ0': {'d': (0,0,0), 'gamma': 1.0, 'order': 6, 'name': 'Rest Frame'},
    'PSQ1': {'d': (0,0,1), 'gamma': 1.1, 'order': 3, 'name': 'Moving along z'},
    'PSQ2': {'d': (1,1,0), 'gamma': 1.15, 'order': 6, 'name': 'Diagonal in xy'},
    'PSQ3': {'d': (1,1,1), 'gamma': 1.2, 'order': 3, 'name': 'Space diagonal'},
    'PSQ4': {'d': (0,0,2), 'gamma': 1.3, 'order': 3, 'name': 'Moving along z (2x)'},
}

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
print("ZOOMED HYBRID: Eq6 (q²<0) + Complex (q²>0)")
print("Range: q*² ∈ [-0.5, 0.5]")

for psq, info in BOOSTS.items():
    d = info['d']
    gamma = info['gamma']
    name = info['name']
    print(f"\n{psq}: {name}")
    complex_data = build_complex_approx(d, gamma, deg=6)
    print(f"  → Complex intervals: {len(complex_data)}")
    qstar2_neg = np.linspace(-0.3, -0.01, 30)
    qstar2_pos = np.linspace(0.01, 0.3, 30)
    qstar2_all = np.concatenate([qstar2_neg, qstar2_pos])
    z_exact = []
    z_hybrid = []
    methods = []
    qstar2_clean = []
    
    for qstar2 in qstar2_all:
        q2_scaled = qstar2_to_qscaled(qstar2)
        exact = exact_Z_scaled(q2_scaled, d, gamma)
        
        if qstar2 < 0:
            method = "Eq6"
            hybrid = zeta_equation6(q2_scaled, gamma, d, mmax=6)
        else:
            method = "Complex"
            hybrid = eval_complex(q2_scaled, complex_data)
        
        if not np.isnan(exact) and -50 < exact < 50:
            z_exact.append(exact)
            z_hybrid.append(hybrid)
            methods.append(method)
            qstar2_clean.append(qstar2)
    print_verification_table(psq, qstar2_clean, z_exact, z_hybrid, methods)
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{psq}: hybrid q*² ∈ [-0.5, 0.5]" , fontsize=14, fontweight='bold')
    
    neg_mask = np.array(qstar2_clean) < 0
    if np.any(neg_mask):
        q_neg = np.array(qstar2_clean)[neg_mask]
        z_exact_neg = np.array(z_exact)[neg_mask]
        z_hybrid_neg = np.array(z_hybrid)[neg_mask]
        ax1.plot(q_neg, z_exact_neg, 'k-', lw=2.5, label='Exact Z', alpha=0.9)
        ax1.plot(q_neg, z_hybrid_neg, 'b--', lw=2, label='Equation (6) - Exact', alpha=0.8)
    
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.2, label='q*² = 0')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel(r'$q^{*2}$ (physical)', fontsize=11)
    ax1.set_ylabel(r'$Z_{00}$', fontsize=11)
    ax1.set_title('Bound State: Eq6 vs Exact Z', fontsize=11)
    ax1.set_xlim(-0.5, 0)
    ax1.set_ylim(-15, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    pos_mask = np.array(qstar2_clean) > 0
    if np.any(pos_mask):
        q_pos = np.array(qstar2_clean)[pos_mask]
        z_exact_pos = np.array(z_exact)[pos_mask]
        z_hybrid_pos = np.array(z_hybrid)[pos_mask]
        
        ax2.plot(q_pos, z_exact_pos, 'k-', lw=2.5, label='Exact Z', alpha=0.9)
        ax2.plot(q_pos, z_hybrid_pos, 'r--', lw=2, label='Complex/Meromorphic', alpha=0.8)
    
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.2, label='q*² = 0')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'$q^{*2}$ (physical)', fontsize=11)
    ax2.set_ylabel(r'$Z_{00}$', fontsize=11)
    ax2.set_title('Scattering: Complex vs Exact Z', fontsize=11)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-5, 15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    plt.tight_layout()
    outfile = os.path.join(PLOT_DIR, f'{psq}_hybrid_zoomed_complex.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved:{outfile}")
