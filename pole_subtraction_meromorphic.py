#!/usr/bin/env python3
"""
POLE SUBTRACTION + POLYNOMIAL (Meromorphic-style)
Z(q²) = R_left/(q²-p_left) + R_right/(q²-p_right) + P_N(q²)
This is the right way to do meromorphic - subtract poles first,
then fit the smooth residual.
Compare with exact Z and Padé [6/6].
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from scipy.optimize import curve_fit

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00
L = 48
SCALE_FACTOR = (L / (2 * np.pi))**2
def get_poles(d, gamma, q2_max=6.0):
    d_arr = np.array(d)
    if np.linalg.norm(d_arr) < 1e-12:
        return free_poles_d0(q2_max)
    return free_poles_d_nonzero(d_arr, gamma, q2_max)

def free_poles_d0(q2_max):
    N = int(np.sqrt(q2_max)) + 3
    vals = set()
    for nx in range(-N, N+1):
        for ny in range(-N, N+1):
            for nz in range(-N, N+1):
                v = nx*nx + ny*ny + nz*nz
                if 1e-9 < v <= q2_max:
                    vals.add(float(v))
    return sorted(vals)

def free_poles_d_nonzero(d, gamma, q2_max):
    d_arr = np.array(d, dtype=float)
    nmax = int(np.sqrt(q2_max) * max(gamma, 1.0)) + 4
    pts = []
    for nx in range(-nmax, nmax+1):
        for ny in range(-nmax, nmax+1):
            for nz in range(-nmax, nmax+1):
                n = np.array([nx, ny, nz], dtype=float)
                raw = n + d_arr/2.0
                if np.linalg.norm(d_arr) > 1e-12:
                    d_hat = d_arr / np.linalg.norm(d_arr)
                    par = np.dot(raw, d_hat) * d_hat
                    perp = raw - par
                    r = perp + par / gamma
                else:
                    r = raw.copy()
                pts.append(r)
    vals = set()
    for r in pts:
        r2 = np.dot(r, r)
        if 1e-9 < r2 <= q2_max + 0.05:
            vals.add(round(r2, 8))
    return sorted(vals)

def measure_residue(pole, d, gamma):
    epsilons = [1e-4, 7e-5, 5e-5, 3e-5, 2e-5]
    estimates = []
    for eps in epsilons:
        for sign in [+1, -1]:
            try:
                e = sign * eps
                z1 = Z_00(pole + e, d, gamma).real
                z2 = Z_00(pole + 2*e, d, gamma).real
                z4 = Z_00(pole + 4*e, d, gamma).real
                R = (8.0*e*z1 - 6.0*(2*e)*z2 + (4*e)*z4) / 3.0
                estimates.append(R)
            except:
                continue
    if not estimates:
        return 1.0/(4.0*np.pi)
    return np.median(estimates)

def exact_Z_scaled(q2_scaled, d, gamma):
    try:
        return Z_00(q2_scaled, d=d, gamma=gamma).real
    except:
        return np.nan
def build_pole_subtracted(d, gamma, q2_min=-0.5, q2_max=6.0, deg=6, n_samples=200):
    """Build pole-subtracted polynomial approximation"""
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
        q2_vals = np.linspace(a + margin, b - margin, n_samples)
        z_exact = np.array([exact_Z_scaled(q2, d, gamma) for q2 in q2_vals])
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

def eval_pole_subtracted(q2, data):
    for iv in data:
        if iv['a'] <= q2 <= iv['b']:
            q2_norm = (q2 - iv['mid']) / iv['half']
            poly = np.polyval(iv['coeffs'], q2_norm)
            pole_corr = 0.0
            if abs(q2 - iv['a']) > 1e-10:
                pole_corr += iv['residues']['left'] / (q2 - iv['a'])
            if abs(q2 - iv['b']) > 1e-10:
                pole_corr += iv['residues']['right'] / (q2 - iv['b'])
            return poly + pole_corr
    return np.nan

sys.path.insert(0, '/home/jasmine/Desktop/zeta_approx_comparison')
from rebuild_real import eval_pade, build_pade_approx

def plot_comparison(psq, d, gamma, name, poly_data, pade_data):
    print(f"\n{psq}: {name}")
    
    q2_test = np.linspace(-0.5, 6.0, 800)
    poles = get_poles(d, gamma, 6.0)
    for p in poles:
        if -0.5 < p < 6.0:
            q2_test = q2_test[np.abs(q2_test - p) > 0.02]
    z_exact = np.array([exact_Z_scaled(q, d, gamma) for q in q2_test])
    z_poly = np.array([eval_pole_subtracted(q, poly_data) for q in q2_test])
    z_pade = np.array([eval_pade(q, pade_data) for q in q2_test])
    valid_poly = ~np.isnan(z_poly) & ~np.isnan(z_exact)
    valid_pade = ~np.isnan(z_pade) & ~np.isnan(z_exact)
    
    poly_err = np.mean(np.abs(z_exact[valid_poly] - z_poly[valid_poly])) if np.any(valid_poly) else np.nan
    pade_err = np.mean(np.abs(z_exact[valid_pade] - z_pade[valid_pade])) if np.any(valid_pade) else np.nan
    
    print(f"Pole-subtracted Poly (deg 6): {poly_err:.2e}")
    print(f"Padé [6/6]:{pade_err:.2e}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{psq}: Pole-Subtracted Polynomial vs Exact vs Padé [6/6]', fontsize=14, fontweight='bold')
    ax = axes[0, 0]
    ax.plot(q2_test, z_exact, 'k-', lw=2, label='Exact Z')
    ax.plot(q2_test, z_poly, 'r--', lw=1.5, label=f'Poly (deg 6), err={poly_err:.2e}')
    ax.plot(q2_test, z_pade, 'g-.', lw=1.5, label=f'Padé [6/6], err={pade_err:.2e}')
    for p in poles:
        if -0.5 < p < 6.0:
            ax.axvline(p, color='gray', ls=':', alpha=0.3)
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(-15, 15)
    ax.set_xlabel(r'$q^2$ (scaled)')
    ax.set_ylabel(r'$Z_{00}$')
    ax.set_title('Full Range')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[0, 1]
    ax.plot(q2_test, z_exact, 'k-', lw=2, label='Exact')
    ax.plot(q2_test, z_poly, 'r--', lw=1.5, label='Poly')
    ax.plot(q2_test, z_pade, 'g-.', lw=1.5, label='Padé')
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r'$q^2$ (scaled)')
    ax.set_ylabel(r'$Z_{00}$')
    ax.set_title('Zoom: Bound State Region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[1, 0]
    ax.semilogy(q2_test, np.abs(z_exact - z_poly) + 1e-16, 'r--', lw=1.2, label='Poly')
    ax.semilogy(q2_test, np.abs(z_exact - z_pade) + 1e-16, 'g-.', lw=1.2, label='Padé')
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel(r'$q^2$ (scaled)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error Comparison (Log Scale)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax = axes[1, 1]
    methods = ['Pole-Subtracted\nPoly (deg 6)', 'Padé [6/6]']
    errors = [poly_err, pade_err]
    colors = ['red', 'green']
    bars = ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_yscale('log')
    ax.set_title('Mean Error Comparison')
    for bar, err in zip(bars, errors):
        if not np.isnan(err):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{err:.2e}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    outfile = os.path.join(os.path.dirname(__file__), f'{psq}_pole_subtracted.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")
    return {'poly': poly_err, 'pade': pade_err}

BOOSTS = {
    "PSQ0": {"d": (0,0,0), "gamma": 1.0, "name": "Rest Frame"},
    "PSQ1": {"d": (0,0,1), "gamma": 1.1, "name": "Moving along z"},
    "PSQ2": {"d": (1,1,0), "gamma": 1.15, "name": "Diagonal in xy"},
    "PSQ3": {"d": (1,1,1), "gamma": 1.2, "name": "Space diagonal"},
    "PSQ4": {"d": (0,0,2), "gamma": 1.3, "name": "Moving along z (2x)"},
}

print("POLE-SUBTRACTED MEROMORPHIC")
print("Z = R_left/(q²-p_left) + R_right/(q²-p_right) + P_N(q²)")
print("Compare with Padé [6/6]")

results = {}
for psq, info in BOOSTS.items():
    d = info['d']
    gamma = info['gamma']
    name = info['name']
    
    print(f"\nBuild {psq}")
    poly_data = build_pole_subtracted(d, gamma, deg=6)
    pade_data = build_pade_approx(d, gamma, 6)
    err = plot_comparison(psq, d, gamma, name, poly_data, pade_data)
    results[psq] = err

print("SUMMARY: Pole-Subtracted Polynomial vs Padé [6/6]")
print(f"{'PSQ':<8} {'Pole-Subtracted Poly (deg6)':<30} {'Padé [6/6]':<20}")
for psq, err in results.items():
    print(f"{psq:<8} {err['poly']:.2e},{err['pade']:.2e}")
