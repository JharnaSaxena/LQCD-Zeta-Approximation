# Zeta Approximation Framework

## Files
- `rebuild_real.py` - Build Padé coefficients
- `rebuild_real_tabular.py` - Same + tabular output
- `hybrid_zeta_final_working.py` - Final hybrid (Eq6 + Padé)
- `hybrid_zeta_zoomed.py` - Zoomed plots
- `speedup_accurate.py` - Speedup analysis
- `make_error_chart.py` - Error summary chart
- `implement_eq6.py` - Eq6 verification
- `kinematics_qstar.py` - q* kinematics (reference)

## Best Methods
- PSQ0: Padé [6/6] (6.27e-7 error)
- PSQ1: Padé [3/3] (6.42e-5 error)
- PSQ2: Padé [6/6] (2.65e-5 error)
- PSQ3: Padé [3/3] (8.34e-4 error)
- PSQ4: Padé [3/3] (4.07e-3 error)

## Speedup
Average: ~2233×
