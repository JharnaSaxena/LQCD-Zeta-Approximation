import matplotlib.pyplot as plt
import numpy as np
psq = ['PSQ0', 'PSQ1', 'PSQ2', 'PSQ3', 'PSQ4']
errors = [6.27e-7, 6.42e-5, 2.65e-5, 8.34e-4, 4.07e-3]
methods = ['Padé [6/6]', 'Padé [3/3]', 'Padé [6/6]', 'Padé [3/3]', 'Padé [3/3]']
colors = ['purple', 'orange', 'purple', 'orange', 'orange']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(psq, errors, color=colors, alpha=0.7)
ax.set_yscale('log')
ax.set_ylabel('Mean Absolute Error', fontsize=12)
ax.set_xlabel('Boost Case', fontsize=12)
ax.set_title('Padé Approximation Error by PSQ', fontsize=14, fontweight='bold')

for bar, err, method in zip(bars, errors, methods):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
            f'{err:.1e}\n{method}', ha='center', va='bottom', fontsize=9)

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('error_summary_chart.png', dpi=150, bbox_inches='tight')
plt.close()
