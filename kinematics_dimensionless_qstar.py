"""
Utilities for converting between q*, q̃ and energies.
Workflow:
    q* (in units of m_ref)->q̃ = q*L / 2π->E_cm->E_lab -> γ = E_lab / E_cm
All quantities are assumed to be dimensionless unless a different
reference occurs
"""
import math
import numpy as np
MOMENTUM_STATES = {
    "PSQ0": np.array([0, 0, 0]),
    "PSQ1": np.array([0, 0, 1]),
    "PSQ2": np.array([1, 1, 0]),
    "PSQ3": np.array([1, 1, 1]),
    "PSQ4": np.array([0, 0, 2]),
    "PSQ5": np.array([2, 0, 0]),
}
def get_momentum_vector(psq):
    return MOMENTUM_STATES.get(psq, np.zeros(3))


def compute_qtilde(qstar_over_mref, L=48, m_ref=1.0):
    scale = L * m_ref / (2.0 * math.pi)
    return qstar_over_mref * scale


def compute_cm_energy(qstar_over_mref,m1_over_mref=1.0,m2_over_mref=1.0,m_ref=1.0):
    e1 = np.sqrt(qstar_over_mref**2 + m1_over_mref**2)
    e2 = np.sqrt(qstar_over_mref**2 + m2_over_mref**2)
    return m_ref * (e1 + e2)


def compute_total_momentum(psq, L=48, m_ref=1.0):
    dvec = get_momentum_vector(psq)
    prefactor = 2.0 * math.pi / (L * m_ref)

    return prefactor * dvec

def compute_lab_energy(qstar_over_mref,psq,L=48,m1_over_mref=1.0,m2_over_mref=1.0,m_ref=1.0):
    e_cm = compute_cm_energy(qstar_over_mref,m1_over_mref,m2_over_mref,m_ref)
    p_mag = np.linalg.norm(compute_total_momentum(psq, L, m_ref))

    return np.sqrt(e_cm**2 + p_mag**2)

def compute_gamma(qstar_over_mref,psq,L=48,m1_over_mref=1.0,m2_over_mref=1.0,m_ref=1.0):

    e_cm = compute_cm_energy(qstar_over_mref,m1_over_mref,m2_over_mref,m_ref)
    e_lab = compute_lab_energy(qstar_over_mref,psq,L,m1_over_mref,m2_over_mref,m_ref)
    return e_lab / e_cm


def qstar2_to_qscaled(qstar2_over_mref2, L=48, m_ref=1.0):
    qstar_over_mref = np.sqrt(qstar2_over_mref2)
    q_tilde = compute_qtilde(qstar_over_mref, L, m_ref)

    return q_tilde**2


def qscaled_to_qstar2(qscaled, L=48, m_ref=1.0):
    scale = (L * m_ref / (2.0 * math.pi))**2
    return qscaled / scale
