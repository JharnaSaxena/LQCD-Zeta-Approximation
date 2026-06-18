# LQCD-Zeta-Approximation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Fast Padé and hybrid approximations for the **Lüscher zeta function** in Lattice Quantum Chromodynamics (LQCD). Built for baryon scattering analysis with **~2233× speedup** and **6.27×10⁻⁷ accuracy** for PSQ0. Includes q* kinematics, Equation (6) for bound states, and meromorphic tests.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [References](#references)
- [License](#license)

---

## Features

- **Padé Approximations** – [3/3] to [6/6] with measured residues
- **Hybrid Zeta Function** – Equation (6) for q² < 0 (exact) + Padé for q² > 0
- **q* Kinematics** – Joseph's chain: q* → q̃ → E* → E → γ
- **Speedup** – ~2233× faster than exact Z evaluation
- **Accuracy** – Best error: 6.27×10⁻⁷ for PSQ0 (Padé [6/6])
- **Equation (6)** – Exact finite-volume modification function for bound states (errors ~1e-15)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/JharnaSaxena/LQCD-Zeta-Approximation.git
cd LQCD-Zeta-Approximation

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0
- h5py >= 3.8.0

---

## Usage

### 1. Build Padé Coefficients
```bash
python rebuild_real.py
```
Generates Padé coefficients for PSQ0–PSQ4 using measured residues.

### 2. Generate Tabular Summary
```bash
python rebuild_real_tabular.py
```
Prints detailed error comparison table.

### 3. Generate Hybrid Plots
```bash
python hybrid_zeta_final_working.py
```
Creates hybrid zeta plots (Eq6 + Padé) for all PSQ cases.

### 4. Generate Zoomed Plots
```bash
python hybrid_zeta_zoomed.py
```
Creates zoomed plots focusing on q*² ∈ [-0.5, 0.5].

### 5. Run Speedup Analysis
```bash
python speedup_accurate.py
```
Measures speedup of Padé approximations vs exact Z.

### 6. Verify Equation (6)
```bash
python implement_eq6.py
```
Verifies Eq (6) for bound state region (errors ~1e-15).

### 7. Generate Error Summary
```bash
python make_error_chart.py
```
Creates error summary bar chart.

---

## Results

### Best Methods per Boost

| PSQ | Best Method | Mean Error | Speedup |
|-----|-------------|------------|---------|
| **PSQ0** | **Padé [6/6]** | **6.27×10⁻⁷** | **~1200×** |
| PSQ1 | Padé [3/3] | 6.42×10⁻⁵ | ~2000× |
| PSQ2 | Padé [6/6] | 2.65×10⁻⁵ | ~2000× |
| PSQ3 | Padé [3/3] | 8.34×10⁻⁴ | ~2000× |
| PSQ4 | Padé [3/3] | 4.07×10⁻³ | ~2000× |

### Average Speedup
**~2233×**

### Hybrid Zeta Configuration
- **q² < 0**: Equation (6) → **Exact** (errors ~1e-15)
- **q² > 0**: **Best Padé** → Fast & Accurate
- **Fallback**: Exact Z outside trained intervals

---

## File Structure

```
LQCD-Zeta-Approximation/
├── coefficients/
│   ├── PSQ0_coeffs.json          # Padé coefficients for PSQ0
│   ├── PSQ1_coeffs.json
│   ├── PSQ2_coeffs.json
│   ├── PSQ3_coeffs.json
│   └── PSQ4_coeffs.json
├── docs/
│   └── results_summary.md        # Detailed results
├── rebuild_real.py               # Build Padé coefficients
├── rebuild_real_tabular.py       # Tabular error summary
├── hybrid_zeta_final_working.py  # Final hybrid zeta
├── hybrid_zeta_zoomed.py         # Zoomed comparison plots
├── speedup_accurate.py           # Speedup analysis
├── make_error_chart.py           # Error summary chart
├── implement_eq6.py              # Equation (6) verification
├── kinematics_qstar.py           # q* kinematics (Joseph's chain)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## References

1. Rummukainen, K., & Gottlieb, S. (1995). "Lüscher's zeta function for moving frames." *Physical Review D*.
2. Davoudi, Z. (2016). "Finite-volume modification function for bound states." *Physical Review D*.
3. Lüscher, M. (1986). "Two-particle states on a torus and their relation to the scattering matrix." *Nuclear Physics B*.
4. Moscoso, J. (2026). Personal communication regarding q* kinematics.

---

License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

Contributing

This code was developed as part of a research internship under the supervision of Dr. Joseph Moscoso at the University of Maryland. 

---

**Made with ❤️ for Lattice QCD Research**
