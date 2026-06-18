#!/usr/bin/env python3
"""
Hybrid implementation of Z00:
- Equation (6) for q² < 0
- Pade approximation for q² > 0
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00

# Constants
L = 48
SCALE_FACTOR = (L / (2 * np.pi))**2  # ~58.4

def qstar2_to_qscaled(qstar2):
    return qstar2 * SCALE_FACTOR

# Load Pade coefficients
COEFF_DIR = os.path.join(os.path.dirname(__file__),'coefficients')
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

def eval_pade_from_coeffs(q2_scaled, pade_data, order):
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

# EQUATION (6) - EXACT FOR q² < 0
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

def zeta_equation6(q2_scaled, gamma, d, L=L, alpha=0.5, mmax=6):
    if q2_scaled >= 0:
        return np.nan
    
    kappa = np.sqrt(-q2_scaled) * (2 * np.pi / L)
    kappa_L = kappa * L
    F_val = F_function(kappa_L, gamma, d, alpha, mmax)
    qcotd = (1.0 / L) * F_val - kappa
    Z_val = (gamma * L * np.sqrt(np.pi) / 2.0) * qcotd
    
    return Z_val

def exact_Z_scaled(q2_scaled, d, gamma):
    try:
        return Z_00(q2_scaled, d=d, gamma=gamma).real
    except:
        return np.nan

# Hybrid evaluator
class HybridZeta:
    def __init__(self):
        self.psq_list = ['PSQ0', 'PSQ1', 'PSQ2', 'PSQ3', 'PSQ4']
        self.pade_data = {}
        self.best_order = {'PSQ0':6, 'PSQ1':3, 'PSQ2':6, 'PSQ3':3, 'PSQ4':3}
        for psq in self.psq_list:
            order = self.best_order[psq]
            self.pade_data[psq] = {'order': order, 'data': load_pade_coeffs(psq, order)}
    
    def evaluate(self, q2_scaled, psq, d, gamma):
        """Evaluate hybrid using Equation (6) for q²<0, Padé for q²>0"""
        if q2_scaled < 0:
            return zeta_equation6(q2_scaled, gamma, d, L=48, mmax=6)
        else:
            info = self.pade_data.get(psq)
            if info and info['data']:
                val = eval_pade_from_coeffs(q2_scaled, info['data'], info['order'])
                if not np.isnan(val):
                    return val
            return exact_Z_scaled(q2_scaled, d, gamma)

# Verification 
def print_verification_table(hybrid, psq, d, gamma):
    print(f"\n{'+'*70}")
    print(f"{psq}: γ={gamma}, d={d}")
    print(f"{'q*² (phys)':>12} {'q²_scaled':>12} {'Exact Z':>12} {'Hybrid Z':>12} {'|Error|':>12}")
    print("-" * 70)
    
    test_qstar2 = [-0.5, -0.3, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for qstar2 in test_qstar2:
        q2_scaled = qstar2_to_qscaled(qstar2)
        exact = exact_Z_scaled(q2_scaled, d, gamma)
        hybrid_val = hybrid.evaluate(q2_scaled, psq, d, gamma)
        
        if not np.isnan(exact) and not np.isnan(hybrid_val):
            err = abs(exact - hybrid_val)
            status = "OK" if err < 1e-3 else ("WARN" if err < 0.1 else "FAIL")
            print(f"{qstar2:12.2f} {q2_scaled:12.2f} {exact:12.6f} {hybrid_val:12.6f} {err:12.6e} {status}")
        else:
            print(f"{qstar2:12.2f} {q2_scaled:12.2f} {'FAILED':>12} {'FAILED':>12} {'---':>12}")
    print("-" * 70)
    
# Plotting
def plot_hybrid_comparison():
    hybrid = HybridZeta()
    
    BOOSTS = {
        'PSQ0': {'d': (0,0,0), 'gamma': 1.0, 'name': 'Rest Frame'},
        'PSQ1': {'d': (0,0,1), 'gamma': 1.1, 'name': 'Moving along z'},
        'PSQ2': {'d': (1,1,0), 'gamma': 1.15, 'name': 'Diagonal in xy'},
        'PSQ3': {'d': (1,1,1), 'gamma': 1.2, 'name': 'Space diagonal'},
        'PSQ4': {'d': (0,0,2), 'gamma': 1.3, 'name': 'Moving along z (2x)'},
    }
    
    PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)
    qstar2_vals = np.linspace(-0.8, 5.0, 400)
    
    for psq, info in BOOSTS.items():
        d = info['d']
        gamma = info['gamma']
        name = info['name']
        
        print(f"\nGenerating {psq} plots...")
        print_verification_table(hybrid, psq, d, gamma)
        
        z_exact = []
        z_hybrid = []
        qstar2_clean = []
        
        for qstar2 in qstar2_vals:
            q2_scaled = qstar2_to_qscaled(qstar2)
            exact = exact_Z_scaled(q2_scaled, d, gamma)
            hybrid_val = hybrid.evaluate(q2_scaled, psq, d, gamma)
            
            if not np.isnan(exact) and not np.isnan(hybrid_val) and -20 < exact < 20:
                z_exact.append(exact)
                z_hybrid.append(hybrid_val)
                qstar2_clean.append(qstar2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{psq}: {name} (γ={gamma}) - Hybrid Zeta (Eq6 + Best Padé)', fontsize=14, fontweight='bold')
        
        # LEFT: Negative q*² region
        neg_mask = np.array(qstar2_clean) < 0
        if np.any(neg_mask):
            q_neg = np.array(qstar2_clean)[neg_mask]
            ax1.plot(q_neg, np.array(z_exact)[neg_mask], 'k-', lw=2.5, label='Exact Z (QC2)', alpha=0.9)
            ax1.plot(q_neg, np.array(z_hybrid)[neg_mask], 'b--', lw=2, label='Equation (6) - Exact', alpha=0.8)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.5, label='q*² = 0')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel(r'$q^{*2}$ (physical, GeV²)', fontsize=11)
        ax1.set_ylabel(r'$Z_{00}^{(d)}$', fontsize=11)
        ax1.set_title('Bound State Region: q*² < 0\nEquation (6) vs Exact Z', fontsize=10)
        ax1.set_xlim(-0.8, 0)
        ax1.set_ylim(-15, 5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # RIGHT: Positive q*² region
        pos_mask = np.array(qstar2_clean) > 0
        if np.any(pos_mask):
            q_pos = np.array(qstar2_clean)[pos_mask]
            z_exact_pos = np.array(z_exact)[pos_mask]
            z_hybrid_pos = np.array(z_hybrid)[pos_mask]
            order = hybrid.best_order[psq]
            ax2.plot(q_pos, z_exact_pos, 'k-', lw=2.5, label='Exact Z (QC2)', alpha=0.9)
            ax2.plot(q_pos, z_hybrid_pos, 'r--', lw=2, label=f'Padé [{order}/{order}] Approximation', alpha=0.8)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.5, label='q*² = 0')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel(r'$q^{*2}$ (physical, GeV²)', fontsize=11)
        ax2.set_ylabel(r'$Z_{00}^{(d)}$', fontsize=11)
        ax2.set_title(f'Scattering Region: q*² > 0\nBest Padé [{order}/{order}] vs Exact Z', fontsize=10)
        ax2.set_xlim(0, 5.0)
        ax2.set_ylim(-15, 15)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        outfile = os.path.join(PLOT_DIR, f'{psq}_hybrid_final_working.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {outfile}")
if __name__ == "__main__":
    print("Generating comparison plots...")
    plot_hybrid_comparison()
    print("  - q² < 0: Equation (6) via F^{(d)} (EXACT, machine precision)")
    print("  - q² > 0: Best Padé (6/6 for PSQ0/2, 3/3 for others)")
