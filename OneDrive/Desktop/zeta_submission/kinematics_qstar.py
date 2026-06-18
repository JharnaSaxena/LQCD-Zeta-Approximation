"""
KINEMATICS USING q* (Joseph's chain)
q* → q̃ = q*L/2π → E* → E → γ
"""
import numpy as np
import math

def momentum_state(psq):
    mapping = {
        'PSQ0': np.array([0,0,0]), 'PSQ1': np.array([0,0,1]),
        'PSQ2': np.array([1,1,0]), 'PSQ3': np.array([1,1,1]),
        'PSQ4': np.array([0,0,2]), 'PSQ5': np.array([2,0,0]),
    }
    return mapping.get(psq, np.array([0,0,0]))

def qtilde(qstar, L=48):
    return qstar * L / (2 * math.pi)

def energy_cm(qstar, m1=1.0, m2=1.0):
    return np.sqrt(qstar**2 + m1**2) + np.sqrt(qstar**2 + m2**2)

def total_momentum(psq, L=48):
    d = momentum_state(psq)
    return (2 * math.pi / L) * d

def energy_lab(qstar, psq, L=48, m1=1.0, m2=1.0):
    E_star = energy_cm(qstar, m1, m2)
    P_mag = np.linalg.norm(total_momentum(psq, L))
    return np.sqrt(E_star**2 + P_mag**2)

def gamma_factor(qstar, psq, L=48, m1=1.0, m2=1.0):
    return energy_lab(qstar, psq, L, m1, m2) / energy_cm(qstar, m1, m2)

def qstar_to_qscaled(qstar, L=48):
    return (qstar * L / (2 * math.pi))**2
