#!/usr/bin/env python3
"""
COMPARE EXACT Z vs POLYNOMIAL and EXACT Z vs PADÉ
ULTRA ZOOMED: q*² ∈ [-0.2, 0.2]
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

COEFF_DIR = os.path.join(os.path.dirname(__file__), 'coefficients')

def pade_33(x, a0, a1, a2, a3, b1, b2, b3):
    return (a0 + a1*x + a2*x**2 + a3*x**3) / (1 + b1*x + b2*x**2 + b3*x**3)

def pade_66(x, a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6):
    num = (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6)
    den = (1 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 + b5*x**5 + b6*x**6)
    return num / den

def load_pade_coeffs(psq, order):
    fname = os.path.join(COEFF_DIR, f'{psq}_coeffs.json')
    if not os.path.exists(fname):
        return None
    with open(fname, 'r') as f:
        data = json.load(f)
    key = 'pade_3_3' if order == 3 else 'pade_6_6'
    return data.get(key, [])

def eval_pade(q2_scaled, pade_data, order):
    if not pade_data:
        return np.nan
    for iv in pade_data:
        if iv['a'] <= q2_scaled <= iv['b']:
            x = (q2_scaled - iv['mid']) / iv['half']
            if order == 3:
                a0,a1,a2,a3,b1,b2,b3 = iv['coeffs']
                val = pade_33(x, a0,a1,a2,a3,b1,b2,b3)
            else:
                val = pade_66(x, *iv['coeffs'])
            pole_corr = 0.0
            if iv['a'] > -0.4:
                pole_corr += iv['residues']['left'] / (q2_scaled - iv['a'] + 1e-10)
            pole_corr += iv['residues']['right'] / (q2_scaled - iv['b'] + 1e-10)
            return val + pole_corr
    return np.nan

def build_polynomial_approx(d, gamma, deg=6):
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
    
    poly_data = []
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
        
        poly_data.append({'a': a, 'b': b, 'mid': mid, 'half': half,'coeffs': coeffs.tolist(),'residues': {'left': residues.get(a, 0.0), 'right': residues.get(b, 0.0)}})
    return poly_data

def eval_polynomial(q2_scaled, poly_data):
    for iv in poly_data:
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

BOOSTS = {
    'PSQ0': {'d': (0,0,0), 'gamma': 1.0, 'order': 6, 'name': 'Rest Frame'},
    'PSQ1': {'d': (0,0,1), 'gamma': 1.1, 'order': 3, 'name': 'Moving along z'},
    'PSQ2': {'d': (1,1,0), 'gamma': 1.15, 'order': 6, 'name': 'Diagonal in xy'},
    'PSQ3': {'d': (1,1,1), 'gamma': 1.2, 'order': 3, 'name': 'Space diagonal'},
    'PSQ4': {'d': (0,0,2), 'gamma': 1.3, 'order': 3, 'name': 'Moving along z (2x)'},
}

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

print("ZOOMED COMPARISON: Exact Z vs Polynomial | Exact Z vs Padé")
print("Range: q*² ∈ [-0.2, 0.2]")
for psq, info in BOOSTS.items():
    d = info['d']
    gamma = info['gamma']
    order = info['order']
    name = info['name']
    
    print(f"\n{psq}: {name}")
    poly_data = build_polynomial_approx(d, gamma, deg=6)
    print(f"{len(poly_data)} intervals")
    print("Padé coefficients")
    pade_data = load_pade_coeffs(psq, order)
    print(f"{len(pade_data) if pade_data else 0} intervals")
    qstar2_vals = np.linspace(-0.2, 0.2, 800)
    z_exact = []
    z_poly = []
    z_pade = []
    qstar2_clean = []
    for qstar2 in qstar2_vals:
        q2_scaled = qstar2_to_qscaled(qstar2)
        
        exact = exact_Z_scaled(q2_scaled, d, gamma)
        poly = eval_polynomial(q2_scaled, poly_data) if poly_data else np.nan
        pade = eval_pade(q2_scaled, pade_data, order) if pade_data else np.nan
        
        if not np.isnan(exact) and -50 < exact < 50:
            z_exact.append(exact)
            z_poly.append(poly)
            z_pade.append(pade)
            qstar2_clean.append(qstar2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{psq}: {name} - Exact Z vs Polynomial vs Padé\nULTRA ZOOMED: q*² ∈ [-0.2, 0.2]', 
                fontsize=14, fontweight='bold')
    ax1.plot(qstar2_clean, z_exact, 'k-', lw=2.5, label='Exact Z', alpha=0.9)
    valid_poly = ~np.isnan(z_poly)
    if np.any(valid_poly):
        ax1.plot(np.array(qstar2_clean)[valid_poly], 
                np.array(z_poly)[valid_poly], 'r--', lw=1.8, 
                label='Polynomial (deg 6)', alpha=0.7)
    
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, lw=1.2, label='q*² = 0')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel(r'$q^{*2}$ (physical)', fontsize=11)
    ax1.set_ylabel(r'$Z_{00}$', fontsize=11)
    ax1.set_title('Exact Z vs Polynomial (Ultra Zoomed)', fontsize=11)
    ax1.set_xlim(-0.2, 0.2)
    ax1.set_ylim(-5, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax2.plot(qstar2_clean, z_exact, 'k-', lw=2.5, label='Exact Z', alpha=0.9)
    valid_pade = ~np.isnan(z_pade)
    if np.any(valid_pade):
        ax2.plot(np.array(qstar2_clean)[valid_pade], 
                np.array(z_pade)[valid_pade], 'b--', lw=1.8, 
                label=f'Padé [{order}/{order}]', alpha=0.7)
    
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, lw=1.2, label='q*² = 0')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel(r'$q^{*2}$ (physical)', fontsize=11)
    ax2.set_ylabel(r'$Z_{00}$', fontsize=11)
    ax2.set_title(f'Exact Z vs Padé [{order}/{order}] (Ultra Zoomed)', fontsize=11)
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    outfile = os.path.join(PLOT_DIR, f'{psq}_poly_pade_ultrazoomed.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")