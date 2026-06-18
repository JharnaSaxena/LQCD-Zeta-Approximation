# LQCD-Zeta-Approximation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Fast Padé and hybrid approximations for the **Lüscher zeta function** in Lattice Quantum Chromodynamics (LQCD). Built for baryon scattering analysis with **~2233× speedup** and **6.27×10⁻⁷ accuracy** for PSQ0. Includes q* kinematics, Equation (6) for bound states, and meromorphic tests.

---

## Table of Contents
- [Features](#features)
- [Mathematical Background](#mathematical-background)
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

## Mathematical Background

### 1. q* Kinematics (Joseph's Chain)

All quantities are derived from the center-of-mass momentum **q***:

| Quantity | Formula | Description |
|----------|---------|-------------|
| **q̃** (qtilde) | `q̃ = q* × L / (2π)` | Dimensionless momentum |
| **E*** (CM energy) | `E* = √(q*² + m₁²) + √(q*² + m₂²)` | Center-of-mass energy |
| **P** (total momentum) | `P = (2π/L) × d` | Total momentum (d = boost vector) |
| **E** (lab energy) | `E = √(E*² + P²)` | Laboratory-frame energy |
| **γ** (Lorentz factor) | `γ = E / E*` | Lorentz boost factor |
| **q²_scaled** | `q²_scaled = (q* × L / 2π)²` | What the zeta function takes as input |

**Why it matters:** Once q* is specified, all other quantities are consistently derived. γ is not independent – it's determined by the kinematics.

---

### 2. The Lüscher Zeta Function

The zeta function is defined as:

```
Z_00(q²) = Σ_{n∈Z³} 1/(|r|² - q²)
```

where `r` is the boosted lattice vector:

```
r = (n_parallel - αd)/γ + n_perp
```

**What it does:** Relates finite-volume energy levels (in a box) to infinite-volume scattering amplitudes.

**Inputs:**
- `q²` = q²_scaled (dimensionless)
- `d` = boost vector (PSQ0 = (0,0,0), PSQ1 = (0,0,1), etc.)
- `γ` = Lorentz boost factor
- `α` = mass asymmetry parameter (0.5 for equal masses)

---

### 3. Equation (6) – For Bound States (q² < 0)

For q² < 0, we use **Equation (6)** which is exact:

```
F^(d)(κL) = Σ_{m≠0} 1/|γ̂m| e^{i2παm·d} e^{-|γ̂m|κL}
```

where:

| Symbol | Formula | Description |
|--------|---------|-------------|
| `κ` | `κ = sqrt(-q²_scaled) × (2π/L)` | Imaginary momentum |
| `κL` | `κ × L` | Dimensionless κ |
| `γ̂m` | `γ̂m = γ × m_parallel + m_perp` | Boosted integer vector |
| `m` | `(m_x, m_y, m_z)` | Integer triplet |
| `α` | `0.5` (equal masses) | Mass asymmetry |

Then:

```
qcotd = (1/L) × F^(d)(κL) - κ
Z(q²) = (γ × L × √π / 2) × qcotd
```

**Why it works:** This is the **exact** finite-volume modification function. Errors are ~1e-15 (machine precision).

---

### 4. Padé Approximations – For Scattering (q² > 0)

For q² > 0, we use Padé approximants of the form:

```
P(x) = (a₀ + a₁x + a₂x² + ... + a_Nx^N) / (1 + b₁x + b₂x² + ... + b_Mx^M)
```

where `x` is normalized to [-1, 1] within each interval:

```
x = (q² - mid) / half
mid = (a + b) / 2
half = (b - a) / 2
```

#### Padé Orders Tested:

| Order | Parameters | Formula |
|-------|------------|---------|
| Padé [3/3] | 7 | `(a₀ + a₁x + a₂x² + a₃x³) / (1 + b₁x + b₂x² + b₃x³)` |
| Padé [4/4] | 9 | `(a₀ + ... + a₄x⁴) / (1 + b₁x + ... + b₄x⁴)` |
| Padé [5/5] | 11 | `(a₀ + ... + a₅x⁵) / (1 + b₁x + ... + b₅x⁵)` |
| Padé [6/6] | 13 | `(a₀ + ... + a₆x⁶) / (1 + b₁x + ... + b₆x⁶)` |

---

### 5. Pole Subtraction Method & Richardson Extrapolation

#### 5.1 Pole Subtraction Method

Near each pole, the zeta function behaves like:

```
Z(q²) = R/(q² - a) + S(q²)
```

where:
- `a` = pole position
- `R` = residue (strength of the pole)
- `S(q²)` = smooth residual function

To get a smooth function for fitting, we **subtract pole contributions**:

```
S(q²) = Z(q²) - R_left/(q² - a) - R_right/(q² - b)
```

where:
- `a` = left pole
- `b` = right pole
- `R_left`, `R_right` = measured residues at each pole

**Steps:**

1. Identify all poles in the interval
2. Measure residues using Richardson extrapolation
3. For each interval between two poles, subtract the pole contributions
4. Fit the smooth residual with a polynomial or Padé
5. Add back pole contributions during evaluation:

```
Z_approx(q²) = P(x) + R_left/(q² - a) + R_right/(q² - b)
```

#### 5.2 Richardson Extrapolation

To measure the residue `R` at a pole `a`:

```
f(ε) = ε × Z(a + ε)
```

The residue is then:

```
R = [8 × f(ε) - 6 × f(2ε) + f(4ε)] / 3
```

**What this does:** Cancels higher-order terms, giving a more accurate residue measurement.

**Implementation steps:**

1. Evaluate Z at points slightly away from the pole: `a + ε`, `a + 2ε`, `a + 4ε`
2. Compute `f(ε) = ε × Z(a + ε)` for each ε
3. Use the Richardson formula to get R
4. Repeat with different ε values and take the median

#### 5.3 Why Pole Subtraction Matters

| Without Pole Subtraction | With Pole Subtraction |
|--------------------------|----------------------|
| Polynomials cannot fit near poles | Smooth residual fits well |
| Large errors near poles | Accurate everywhere |
| Padé struggles with fixed poles | Works perfectly |

**Conclusion:** Pole subtraction is essential for accurate approximation of the zeta function.

---

### 6. Complex Formula (Tested)

```
Z ≈ P_N(q²) / ∏(q² - pᵢ)
```

where `pᵢ` are fixed pole positions.

**Result:**  **FAILED** – errors ~1e79 due to fixed poles in denominator.

**Why it failed:** Numerator polynomial cannot compensate for all poles simultaneously.

**Alternative that worked:** Pole subtraction + polynomial (same as above).

---

### 7. The Hybrid Approach (Recommended)

```
q² < 0  →  Equation (6)  →  EXACT (errors ~1e-15)
q² > 0  →  Best Padé    →  FAST & ACCURATE
```

| PSQ | Best Padé Order | Reason |
|-----|-----------------|--------|
| PSQ0 | Padé [6/6] | Rest frame is symmetric |
| PSQ1 | Padé [3/3] | Moving frame – higher orders unstable |
| PSQ2 | Padé [6/6] | Diagonal boost – higher order works |
| PSQ3 | Padé [3/3] | Moving frame – higher orders unstable |
| PSQ4 | Padé [3/3] | Moving frame – higher orders unstable |

**Fallback:** Exact Z outside trained intervals.

---

### 8. Non-Interacting Energy Levels (Poles)

For L = 48, poles occur at:

```
q²_pole = |n|² × (L/2π)²
```

First poles (scaled q²):
- 58.36, 116.72, 175.08, 233.44, ...

These correspond to `|n|² = 1, 2, 3, 4, ...`

---

### 9. Gamma Comparison

| Approach | Formula | Result |
|----------|---------|--------|
| **Variable γ** | `γ = E/E*` from kinematics | Same result |
| **Constant γ** | Fixed per PSQ (1.0, 1.1, 1.15, 1.2, 1.3) | Same result |

**Conclusion:** Both give identical results because γ is uniquely determined once q* is specified.

---

## Installation

```bash
git clone https://github.com/JharnaSaxena/LQCD-Zeta-Approximation.git
cd LQCD-Zeta-Approximation
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

## Formula Quick Reference

| Concept | Formula |
|---------|---------|
| **q̃ (qtilde)** | `q* × L / 2π` |
| **E*** | `√(q*² + m₁²) + √(q*² + m₂²)` |
| **P** | `(2π/L) × d` |
| **E** | `√(E*² + P²)` |
| **γ** | `E / E*` |
| **q²_scaled** | `(q* × L / 2π)²` |
| **F^(d)(κL)** | `Σ 1/|γ̂m| e^{i2παm·d} e^{-|γ̂m|κL}` |
| **qcotd** | `(1/L) × F - κ` |
| **Z(q²) (Eq6)** | `(γ × L × √π / 2) × qcotd` |
| **Padé** | `P(x) = (a₀ + ... + a_Nx^N) / (1 + b₁x + ... + b_Mx^M)` |
| **Residue (Richardson)** | `R = [8f(ε) - 6f(2ε) + f(4ε)] / 3` |
| **Pole Subtraction** | `S(q²) = Z(q²) - R_left/(q²-a) - R_right/(q²-b)` |
| **Final Evaluation** | `Z_approx = P(x) + R_left/(q²-a) + R_right/(q²-b)` |
| **Non-Interacting Poles** | `q²_pole = |n|² × (L/2π)²` |
| **Speedup** | `Time_Exact / Time_Padé` |

---

## References

1. Rummukainen, K., & Gottlieb, S. (1995). "Lüscher's zeta function for moving frames." *Physical Review D*.
2. Davoudi, Z. (2016). "Finite-volume modification function for bound states." *Physical Review D*.
3. Lüscher, M. (1986). "Two-particle states on a torus and their relation to the scattering matrix." *Nuclear Physics B*.
4. Moscoso, J. (2026). Personal communication regarding q* kinematics.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Contributing

This code was developed as part of a research internship under the supervision of Dr. Joseph Moscoso at the University of Maryland. For questions or suggestions, please contact the repository owner.
