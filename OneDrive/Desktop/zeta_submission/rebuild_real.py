#!/usr/bin/env python3
"""
REAL ZETA APPROXIMATIONS - Using actual QC2 implementation
This matches high_accuracy_zeta.py but organized for zeta_approx_comparison
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00
from QC2.tools.kinematics import gamma as gamma_kin

L = 48
ref = 1.0
m1 = m2 = 1.0
q2_min, q2_max = -0.5, 6.0

# Boosts with representative gamma (same as before)
BOOSTS = {
    "PSQ0": {"d": (0,0,0), "gamma": 1.0, "name": "Rest Frame"},
    "PSQ1": {"d": (0,0,1), "gamma": 1.1, "name": "Moving along z"},
    "PSQ2": {"d": (1,1,0), "gamma": 1.15, "name": "Diagonal in xy"},
    "PSQ3": {"d": (1,1,1), "gamma": 1.2, "name": "Space diagonal"},
    "PSQ4": {"d": (0,0,2), "gamma": 1.3, "name": "Moving along z (2x)"},
}

# Output
COEFF_DIR = os.path.join(os.path.dirname(__file__), 'coefficients')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(COEFF_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("REAL ZETA APPROXIMATIONS - Using QC2 exact Z")
# POLE FUNCTIONS (from high_accuracy_zeta.py)
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

# RESIDUE MEASUREMENT
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

# PADÉ FUNCTIONS
def pade_33(x, a0, a1, a2, a3, b1, b2, b3):
    return (a0 + a1*x + a2*x**2 + a3*x**3) / (1 + b1*x + b2*x**2 + b3*x**3)

def pade_44(x, a0, a1, a2, a3, a4, b1, b2, b3, b4):
    return (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4) / (1 + b1*x + b2*x**2 + b3*x**3 + b4*x**4)

def pade_55(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5):
    return (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5) / (1 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 + b5*x**5)

def pade_66(x, a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6):
    return (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6) / (1 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 + b5*x**5 + b6*x**6)

# BUILD PADÉ APPROXIMATION (generic for any order)
def build_pade_approx(d, gamma, order, q2_min=-0.5, q2_max=6.0, n_samples=150):
    pade_funcs = {3: pade_33, 4: pade_44, 5: pade_55, 6: pade_66}
    pade_func = pade_funcs[order]
    n_params = 2 * order + 1
    
    poles = get_poles(d, gamma, q2_max)
    poles = [p for p in poles if q2_min < p < q2_max]
    
    intervals = [(q2_min, poles[0])] if poles else [(q2_min, q2_max)]
    for i in range(len(poles)-1):
        intervals.append((poles[i], poles[i+1]))
    if poles and poles[-1] < q2_max:
        intervals.append((poles[-1], q2_max))
    
    residues = {}
    for pole in poles:
        residues[pole] = measure_residue(pole, d, gamma)
    
    pade_data = []
    for a, b in intervals:
        if b - a < 0.08:
            continue
        margin = 0.05 * (b - a)
        q2_vals = np.linspace(a + margin, b - margin, n_samples)
        z_exact = np.array([Z_00(q2, d, gamma).real for q2 in q2_vals])
        
        pole_correction = np.zeros_like(q2_vals)
        if a in residues:
            pole_correction += residues[a] / (q2_vals - a)
        if b in residues:
            pole_correction += residues[b] / (q2_vals - b)
        residual = z_exact - pole_correction
        
        try:
            mid = (a + b) / 2.0
            half = (b - a) / 2.0
            q2_norm = (q2_vals - mid) / half
            p0 = [np.mean(residual)] + [0.1] * (n_params - 1)
            popt, _ = curve_fit(pade_func, q2_norm, residual, p0=p0, maxfev=10000)
            pade_data.append({
                'a': a, 'b': b, 'mid': mid, 'half': half,
                'coeffs': popt.tolist(),
                'order': order,
                'residues': {'left': residues.get(a, 0.0), 'right': residues.get(b, 0.0)}
            })
        except Exception as e:
            pass
    return pade_data

def eval_pade(q2, data):
    pade_funcs = {3: pade_33, 4: pade_44, 5: pade_55, 6: pade_66}
    for iv in data:
        if iv['a'] <= q2 <= iv['b']:
            x = (q2 - iv['mid']) / iv['half']
            pade_func = pade_funcs[iv['order']]
            pade_val = pade_func(x, *iv['coeffs'])
            pole_corr = 0.0
            if abs(q2 - iv['a']) > 1e-10:
                pole_corr += iv['residues']['left'] / (q2 - iv['a'])
            if abs(q2 - iv['b']) > 1e-10:
                pole_corr += iv['residues']['right'] / (q2 - iv['b'])
            return pade_val + pole_corr
    return np.nan

# ADAPTIVE SPLINE
def build_adaptive_spline(d, gamma, q2_min=-0.5, q2_max=6.0):
    poles = get_poles(d, gamma, q2_max)
    poles = [p for p in poles if q2_min < p < q2_max]
    
    q2_samples = []
    q2_samples.extend(np.linspace(q2_min, poles[0] if poles else q2_max, 50))
    for i in range(len(poles)-1):
        q2_samples.extend(np.linspace(poles[i], poles[i] + 0.05, 10))
        q2_samples.extend(np.linspace(poles[i], poles[i+1], 80))
        q2_samples.extend(np.linspace(poles[i+1] - 0.05, poles[i+1], 10))
    if poles:
        q2_samples.extend(np.linspace(poles[-1], q2_max, 50))
    q2_samples = np.unique(np.clip(q2_samples, q2_min, q2_max))
    
    z_exact = np.array([Z_00(q2, d, gamma).real for q2 in q2_samples])
    for p in poles:
        mask = np.abs(q2_samples - p) > 0.01
        q2_samples = q2_samples[mask]
        z_exact = z_exact[mask]
    spline = CubicSpline(q2_samples, z_exact, bc_type='not-a-knot')
    return {'q2_samples': q2_samples.tolist(), 'z_samples': z_exact.tolist(), 'spline': spline}

def eval_spline(q2, data):
    if q2 < data['q2_samples'][0] or q2 > data['q2_samples'][-1]:
        return np.nan
    return float(data['spline'](q2))

# POLYNOMIAL APPROXIMATION (deg 6)
def build_polynomial_approx(d, gamma, deg=6, q2_min=-0.5, q2_max=6.0, n_samples=200):
    poles = get_poles(d, gamma, q2_max)
    poles = [p for p in poles if q2_min < p < q2_max]
    
    intervals = [(q2_min, poles[0])] if poles else [(q2_min, q2_max)]
    for i in range(len(poles)-1):
        intervals.append((poles[i], poles[i+1]))
    if poles and poles[-1] < q2_max:
        intervals.append((poles[-1], q2_max))
    
    residues = {}
    for pole in poles:
        residues[pole] = measure_residue(pole, d, gamma)
    
    poly_data = []
    for a, b in intervals:
        if b - a < 0.05:
            continue
        margin = 0.03 * (b - a)
        q2_vals = np.linspace(a + margin, b - margin, n_samples)
        z_exact = np.array([Z_00(q2, d, gamma).real for q2 in q2_vals])
        
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
        
        poly_data.append({
            'a': a, 'b': b, 'mid': mid, 'half': half,
            'coeffs': coeffs.tolist(),
            'residues': {'left': residues.get(a, 0.0), 'right': residues.get(b, 0.0)}
        })
    return poly_data

def eval_polynomial(q2, data):
    for iv in data:
        if iv['a'] <= q2 <= iv['b']:
            x = (q2 - iv['mid']) / iv['half']
            poly = np.polyval(iv['coeffs'], x)
            pole_corr = 0.0
            if abs(q2 - iv['a']) > 1e-10:
                pole_corr += iv['residues']['left'] / (q2 - iv['a'])
            if abs(q2 - iv['b']) > 1e-10:
                pole_corr += iv['residues']['right'] / (q2 - iv['b'])
            return poly + pole_corr
    return np.nan

# PLOT FUNCTION 
def plot_comparison(boost_key, d, gamma, name, poly_data, pade3_data, pade4_data, pade5_data, pade6_data, spline_data):
    poles = get_poles(d, gamma, q2_max)
    q2_full = np.linspace(q2_min, q2_max, 800)
    for p in poles:
        if q2_min < p < q2_max:
            q2_full = q2_full[np.abs(q2_full - p) > 0.02]
    
    print(f"Computing exact Z")
    z_exact = np.array([Z_00(q2, d, gamma).real for q2 in q2_full])
    print(f"Evaluating approximations")
    z_poly = np.array([eval_polynomial(q2, poly_data) for q2 in q2_full])
    z_pade3 = np.array([eval_pade(q2, pade3_data) for q2 in q2_full])
    z_pade4 = np.array([eval_pade(q2, pade4_data) for q2 in q2_full])
    z_pade5 = np.array([eval_pade(q2, pade5_data) for q2 in q2_full])
    z_pade6 = np.array([eval_pade(q2, pade6_data) for q2 in q2_full])
    z_spline = np.array([eval_spline(q2, spline_data) for q2 in q2_full])
    for arr in [z_exact, z_poly, z_pade3, z_pade4, z_pade5, z_pade6, z_spline]:
        arr[np.abs(arr) > 50] = np.nan
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'High Accuracy Zeta Approximations: {name}\nd={tuple(d)}, γ={gamma}', fontsize=14, fontweight='bold')
    
    # Top-left: All methods comparison
    ax = axes[0, 0]
    ax.plot(q2_full, z_exact, 'k-', lw=2, label='Exact Z', alpha=0.8)
    ax.plot(q2_full, z_poly, 'r--', lw=1.5, label='Polynomial (deg 6)', alpha=0.7)
    ax.plot(q2_full, z_pade3, 'orange', '--', lw=1.5, label='Padé [3/3]', alpha=0.7)
    ax.plot(q2_full, z_pade4, 'g-.', lw=1.5, label='Padé [4/4]', alpha=0.7)
    ax.plot(q2_full, z_pade5, 'b-.', lw=1.5, label='Padé [5/5]', alpha=0.7)
    ax.plot(q2_full, z_pade6, 'purple', '-.', lw=1.5, label='Padé [6/6]', alpha=0.7)
    ax.plot(q2_full, z_spline, 'c:', lw=1.5, label='Adaptive Spline', alpha=0.7)
    ax.set_ylabel(r'$Z^d_{00}(q^2)$')
    ax.set_xlim(q2_min, q2_max)
    ax.set_ylim(-20, 20)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_title('All Methods Comparison')
    
    # Top-right: Zoom bound state region
    ax = axes[0, 1]
    ax.plot(q2_full, z_exact, 'k-', lw=2, label='Exact', alpha=0.8)
    ax.plot(q2_full, z_poly, 'r--', lw=1.5, alpha=0.7)
    ax.plot(q2_full, z_pade3, 'orange', '--', lw=1.5, alpha=0.7)
    ax.plot(q2_full, z_pade4, 'g-.', lw=1.5, alpha=0.7)
    ax.plot(q2_full, z_pade5, 'b-.', lw=1.5, alpha=0.7)
    ax.plot(q2_full, z_pade6, 'purple', '-.', lw=1.5, alpha=0.7)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r'$q^{*2}$')
    ax.set_title('Zoom: Bound State Region')
    ax.grid(True, alpha=0.2)
    
    # Bottom-left: Error comparison (log scale)
    ax = axes[1, 0]
    error_poly = np.abs(z_exact - z_poly)
    error_pade3 = np.abs(z_exact - z_pade3)
    error_pade4 = np.abs(z_exact - z_pade4)
    error_pade5 = np.abs(z_exact - z_pade5)
    error_pade6 = np.abs(z_exact - z_pade6)
    error_spline = np.abs(z_exact - z_spline)
    
    ax.semilogy(q2_full, error_poly + 1e-16, 'r-', lw=1.2, label='Polynomial', alpha=0.7)
    ax.semilogy(q2_full, error_pade3 + 1e-16, 'orange', '--', lw=1.2, label='Padé [3/3]', alpha=0.7)
    ax.semilogy(q2_full, error_pade4 + 1e-16, 'g-.', lw=1.2, label='Padé [4/4]', alpha=0.7)
    ax.semilogy(q2_full, error_pade5 + 1e-16, 'b-.', lw=1.2, label='Padé [5/5]', alpha=0.7)
    ax.semilogy(q2_full, error_pade6 + 1e-16, 'purple', '-.', lw=1.2, label='Padé [6/6]', alpha=0.7)
    ax.semilogy(q2_full, error_spline + 1e-16, 'c:', lw=1.2, label='Spline', alpha=0.7)
    ax.set_xlabel(r'$q^{*2}$')
    ax.set_ylabel('Absolute Error')
    ax.set_xlim(q2_min, q2_max)
    ax.set_ylim(1e-12, 1e3)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_title('Error Comparison (Log Scale)')
    
    # Bottom-right: Mean error bar chart
    ax = axes[1, 1]
    valid = np.isfinite(error_poly) & np.isfinite(error_pade3) & np.isfinite(error_spline)
    if valid.any():
        errors = [
            np.mean(error_poly[valid]),
            np.mean(error_pade3[valid]),
            np.mean(error_pade4[valid]),
            np.mean(error_pade5[valid]),
            np.mean(error_pade6[valid]),
            np.mean(error_spline[valid])
        ]
        methods = ['Poly deg6', 'Padé [3/3]', '[4/4]', '[5/5]', '[6/6]', 'Spline']
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'cyan']
        bars = ax.bar(methods, errors, color=colors, alpha=0.7)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_yscale('log')
        ax.set_title('Mean Error Comparison')
        ax.tick_params(axis='x', rotation=45)
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{err:.2e}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    outfile = os.path.join(PLOT_DIR, f'{boost_key}_comparison.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")
    
    return {
        'poly_mean': np.mean(error_poly[valid]) if valid.any() else np.nan,
        'pade3_mean': np.mean(error_pade3[valid]) if valid.any() else np.nan,
        'pade4_mean': np.mean(error_pade4[valid]) if valid.any() else np.nan,
        'pade5_mean': np.mean(error_pade5[valid]) if valid.any() else np.nan,
        'pade6_mean': np.mean(error_pade6[valid]) if valid.any() else np.nan,
        'spline_mean': np.mean(error_spline[valid]) if valid.any() else np.nan,
    }

# MAIN
def main():
    print("BUILDING ALL APPROXIMATIONS WITH REAL QC2 ZETA")
    
    results = {}
    
    for boost_key, info in BOOSTS.items():
        d = info["d"]
        gamma = info["gamma"]
        name = info["name"]
        
        print(f"\n{boost_key}: {name} (γ={gamma})")
        print("Building Polynomial (deg 6)...")
        poly_data = build_polynomial_approx(d, gamma, deg=6)
        print(f"{len(poly_data)} intervals")
        
        print("Building Padé [3/3]...")
        pade3_data = build_pade_approx(d, gamma, 3)
        print(f"    → {len(pade3_data)} intervals")
        
        print("  Building Padé [4/4]...")
        pade4_data = build_pade_approx(d, gamma, 4)
        print(f"{len(pade4_data)} intervals")
        
        print("Building Padé [5/5]...")
        pade5_data = build_pade_approx(d, gamma, 5)
        print(f"{len(pade5_data)} intervals")
        
        print("  Building Padé [6/6]...")
        pade6_data = build_pade_approx(d, gamma, 6)
        print(f"{len(pade6_data)} intervals")
        
        print("Building Adaptive Spline...")
        spline_data = build_adaptive_spline(d, gamma)
        print(f"{len(spline_data['q2_samples'])} sample points")
        
        print("  Generating plots")
        err = plot_comparison(boost_key, d, gamma, name, poly_data, pade3_data, pade4_data, pade5_data, pade6_data, spline_data)
        results[boost_key] = err
        coeffs = {
            'boost': boost_key, 'd': d, 'gamma': gamma,
            'polynomial_deg6': poly_data,
            'pade_3_3': pade3_data,
            'pade_4_4': pade4_data,
            'pade_5_5': pade5_data,
            'pade_6_6': pade6_data,
            'spline': {'q2_samples': spline_data['q2_samples'], 'z_samples': spline_data['z_samples']}
        }
        with open(os.path.join(COEFF_DIR, f'{boost_key}_coeffs.json'), 'w') as f:
            json.dump(coeffs, f, indent=2)
        print(f"Coefficients saved")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Absolute Errors")
    print(f"{'Boost':<12} {'Poly deg6':<15} {'Padé[3/3]':<15} {'[4/4]':<15} {'[5/5]':<15} {'[6/6]':<15} {'Spline':<15}")
    print("-" * 100)
    for boost_key, err in results.items():
        print(f"{boost_key:<12} {err['poly_mean']:.2e}     {err['pade3_mean']:.2e}     {err['pade4_mean']:.2e}     {err['pade5_mean']:.2e}     {err['pade6_mean']:.2e}     {err['spline_mean']:.2e}")

    print(f"Coefficients: {COEFF_DIR}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
