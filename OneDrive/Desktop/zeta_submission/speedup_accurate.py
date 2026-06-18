#!/usr/bin/env python3
"""
Speed comparison of direct Z00 evaluation and Padé approximations.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.insert(0, '/home/jasmine/Desktop/PyCALQ_Week4etc')
from QC2.tools.zeta_wrapper import Z_00
sys.path.insert(0, os.path.dirname(__file__))
from rebuild_real import eval_pade, get_poles


# Parameters
L = 48
q2_min, q2_max = -0.5, 6.0
N_EVALS = 2000  

BOOSTS = {"PSQ0": {"d": (0,0,0), "gamma": 1.0, "order": 6},
    "PSQ1": {"d": (0,0,1), "gamma": 1.1, "order": 3},
    "PSQ2": {"d": (1,1,0), "gamma": 1.15, "order": 6},
    "PSQ3": {"d": (1,1,1), "gamma": 1.2, "order": 3},
    "PSQ4": {"d": (0,0,2), "gamma": 1.3, "order": 3},
}

COEFF_DIR = os.path.join(os.path.dirname(__file__), 'coefficients')

def load_pade_data(psq, order):
    fname = os.path.join(COEFF_DIR, f'{psq}_coeffs.json')
    if not os.path.exists(fname):
        return None
    with open(fname, 'r') as f:
        data = json.load(f)
    key = 'pade_3_3' if order == 3 else 'pade_6_6'
    return data.get(key, [])

def measure_speedup(psq, d, gamma, order, n_evals=2000):
    """Measure speedup using same method as June 8"""
    poles = get_poles(d, gamma, q2_max)
    q2_vals = np.linspace(q2_min + 0.1, q2_max - 0.1, n_evals)
    for p in poles:
        q2_vals = q2_vals[np.abs(q2_vals - p) > 0.03]
    pade_data = load_pade_data(psq, order)
    if not pade_data:
        return None
    
    # Time Exact Z
    start = time.perf_counter()
    for q2 in q2_vals:
        try:
            Z_00(q2, d=d, gamma=gamma).real
        except Exception:
            pass
    exact_time = time.perf_counter() - start
    exact_per_call = (exact_time / len(q2_vals)) * 1e6  # μs
    
    # Time Pade
    start = time.perf_counter()
    for q2 in q2_vals:
        try:
            eval_pade(q2, pade_data)
        except:
            pass
    pade_time = time.perf_counter() - start
    pade_per_call = (pade_time / len(q2_vals)) * 1e6  # μs
    speedup = exact_per_call / pade_per_call if pade_per_call > 0 else 0
    return {'exact_us': exact_per_call,'pade_us': pade_per_call,'speedup': speedup,'n_evals': len(q2_vals)}

def main():
    print(f"Speed Analysis: {N_EVALS} evaluations per boost")
    results = {}
    for psq, info in BOOSTS.items():
        d = info['d']
        gamma = info['gamma']
        order = info['order']
        name = BOOSTS[psq]['name'] if 'name' in BOOSTS[psq] else psq
        print(f"\n {psq}: {name} (γ={gamma}, Padé [{order}/{order}]) ")
        print(f"  Evaluating {N_EVALS} points.")
        result = measure_speedup(psq, d, gamma, order, N_EVALS)
        if result:
            results[psq] = {
                'name': name,
                'order': order,
                'exact_us': result['exact_us'],
                'pade_us': result['pade_us'],
                'speedup': result['speedup'],
                'n_evals': result['n_evals']
            }
            print(f"Exact Z: {result['exact_us']:.2f} μs/call")
            print(f"Pade [{order}/{order}]: {result['pade_us']:.2f} μs/call")
            print(f"Speedup: {result['speedup']:.1f}x")
        else:
            print(f"No Pade coefficients found")

    print("SUMMARY: Speedup Comparison")
    print(f"{'Boost':<8} {'Method':<15} {'Exact (μs)':<15} {'Padé (μs)':<15} {'Speedup':<15} {'N_evals'}")
    
    avg_speedup = 0
    count = 0
    for psq, data in results.items():
        method = f"Padé [{data['order']}/{data['order']}]"
        print(f"{psq:<8} {method:<15} {data['exact_us']:<15.2f} {data['pade_us']:<15.2f} {data['speedup']:<15.1f} {data['n_evals']}")
        avg_speedup += data['speedup']
        count += 1
    
    if count > 0:
        avg_speedup /= count
        print("-" * 90)
        print(f"{'AVERAGE':<8} {'':<15} {'':<15} {'':<15} {avg_speedup:<15.1f}")
    print("=" * 80)
    
    # Generate plot
    plot_speedup(results)
    # Save results
    with open('speedup_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to speedup_results.json")
print("Saved speedup_accurate.png")

def plot_speedup(results):
    psq_labels = []
    speedups = []
    exact_times = []
    pade_times = []
    orders = []
    
    for psq, data in results.items():
        psq_labels.append(psq)
        speedups.append(data['speedup'])
        exact_times.append(data['exact_us'])
        pade_times.append(data['pade_us'])
        orders.append(f"[{data['order']}/{data['order']}]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Speedup: Exact Z vs Padé Approximations ({len(results)} boosts, 2000 evals each)', 
                fontsize=14, fontweight='bold')
    
    # Left: Speedup bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(psq_labels, speedups, color=colors, alpha=0.7)
    ax1.set_ylabel('Speedup Factor', fontsize=12)
    ax1.set_xlabel('Boost Case', fontsize=12)
    ax1.set_title('Padé Speedup over Exact Z', fontsize=12)
    ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='1000×')
    ax1.axhline(y=2000, color='gray', linestyle=':', alpha=0.5, label='2000×')
    
    for bar, speedup, order in zip(bars, speedups, orders):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{speedup:.0f}×\n{order}', ha='center', va='bottom', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(speedups) * 1.3)
    
    # Right: Time comparison (log scale)
    x = np.arange(len(psq_labels))
    width = 0.35
    
    ax2.bar(x - width/2, exact_times, width, label='Exact Z', color='red', alpha=0.7)
    ax2.bar(x + width/2, pade_times, width, label='Padé', color='blue', alpha=0.7)
    
    ax2.set_ylabel('Time (μs per call)', fontsize=12)
    ax2.set_xlabel('Boost Case', fontsize=12)
    ax2.set_title('Evaluation Time Comparison (Log Scale)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(psq_labels)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('speedup_accurate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: speedup_accurate.png")

if __name__ == "__main__":
    main()
