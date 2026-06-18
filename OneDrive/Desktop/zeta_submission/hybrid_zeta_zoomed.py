#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00
L = 48
SCALE_FACTOR = (L / (2 * np.pi))**2
PADE_MAX_SCALED = 6.0

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

class HybridZeta:
    def __init__(self):
        self.psq_list = ['PSQ0', 'PSQ1', 'PSQ2', 'PSQ3', 'PSQ4']
        self.pade_data = {}
        self.best_order = {'PSQ0':6, 'PSQ1':3, 'PSQ2':6, 'PSQ3':3, 'PSQ4':3}
        for psq in self.psq_list:
            order = self.best_order[psq]
            self.pade_data[psq] = {'order': order, 'data': load_pade_coeffs(psq, order)}
    
    def evaluate(self, q2_scaled, psq, d, gamma):
        if q2_scaled < 0:
            return exact_Z_scaled(q2_scaled, d, gamma)
        elif q2_scaled <= PADE_MAX_SCALED:
            info = self.pade_data.get(psq)
            if info and info['data']:
                val = eval_pade_from_coeffs(q2_scaled, info['data'], info['order'])
                if not np.isnan(val):
                    return val
        return exact_Z_scaled(q2_scaled, d, gamma)

def plot_hybrid_zoomed():
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
    # zoom range: -0.5 to 0.5
    qstar2_vals = np.linspace(-0.5, 0.5, 500)
    for psq, info in BOOSTS.items():
        d = info['d']
        gamma = info['gamma']
        name = info['name']
        print(f"{psq}")
        z_exact = []
        z_hybrid = []
        qstar2_clean = []
        for qstar2 in qstar2_vals:
            q2_scaled = qstar2_to_qscaled(qstar2)
            exact = exact_Z_scaled(q2_scaled, d, gamma)
            hybrid_val = hybrid.evaluate(q2_scaled, psq, d, gamma)
            if not np.isnan(exact) and not np.isnan(hybrid_val) and -50 < exact < 50:
                z_exact.append(exact)
                z_hybrid.append(hybrid_val)
                qstar2_clean.append(qstar2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{psq}: {name} (γ={gamma}) - Zoomed View (q*² ∈ [-0.5, 0.5])', fontsize=14, fontweight='bold')
        
        # LEFT: Negative q*² region
        neg_mask = np.array(qstar2_clean) < 0
        if np.any(neg_mask):
            q_neg = np.array(qstar2_clean)[neg_mask]
            ax1.plot(q_neg, np.array(z_exact)[neg_mask], 'k-', lw=2.5, label='Exact Z (QC2)', alpha=0.9)
            ax1.plot(q_neg, np.array(z_hybrid)[neg_mask], 'b--', lw=2, label='Hybrid (Exact Z)', alpha=0.8)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.5, label='q*² = 0')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel(r'$q^{*2}$ (physical, GeV²)', fontsize=11)
        ax1.set_ylabel(r'$Z_{00}^{(d)}$', fontsize=11)
        ax1.set_title('Bound State Region: q*² < 0', fontsize=10)
        ax1.set_xlim(-0.5, 0)
        ax1.set_ylim(-15, 5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # RIGHT: Positive q*² region (zoomed)
        pos_mask = np.array(qstar2_clean) > 0
        if np.any(pos_mask):
            q_pos = np.array(qstar2_clean)[pos_mask]
            z_exact_pos = np.array(z_exact)[pos_mask]
            z_hybrid_pos = np.array(z_hybrid)[pos_mask]
            order = hybrid.best_order[psq]
            ax2.plot(q_pos, z_exact_pos, 'k-', lw=2.5, label='Exact Z (QC2)', alpha=0.9)
            ax2.plot(q_pos, z_hybrid_pos, 'r--', lw=2, label=f'Hybrid (Padé [{order}/{order}] + fallback)', alpha=0.8)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, lw=1.5, label='q*² = 0')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel(r'$q^{*2}$ (physical, GeV²)', fontsize=11)
        ax2.set_ylabel(r'$Z_{00}^{(d)}$', fontsize=11)
        ax2.set_title(f'Scattering Region: q*² > 0 (Zoomed)', fontsize=10)
        ax2.set_xlim(0, 0.5)
        ax2.set_ylim(-5, 15)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        outfile = os.path.join(PLOT_DIR, f'{psq}_hybrid_zoomed.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Zoomed plot saved: {outfile}")

if __name__ == '__main__':
    plot_hybrid_zoomed()

