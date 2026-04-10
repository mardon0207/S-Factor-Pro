
import numpy as np
from functools import lru_cache
from fractions import Fraction
from sympy.physics.wigner import clebsch_gordan as sympy_cg
from sympy.physics.wigner import wigner_6j as sympy_w6j
from sympy import N as sympy_N, Rational as sympy_Rational

def _to_rat(x, tol=1e-9):
    if abs(x) < tol: return Fraction(0, 1)
    rx = round(x)
    if abs(x - rx) < tol: return Fraction(int(rx), 1)
    rx2 = round(2 * x)
    if abs(2*x - rx2) < tol: return Fraction(int(rx2), 2)
    return Fraction(x).limit_denominator(1_000_000)

def _sr(x):
    f = _to_rat(x)
    return sympy_Rational(f.numerator, f.denominator)

@lru_cache(maxsize=4096)
def CGC(j1, m1, j2, m2, J, M):
    try:
        return float(sympy_N(sympy_cg(_sr(j1), _sr(j2), _sr(J),
                                       _sr(m1), _sr(m2), _sr(M))))
    except Exception: return 0.0

@lru_cache(maxsize=4096)
def Wigner6j(j1, j2, j3, j4, j5, j6):
    try:
        return float(sympy_N(sympy_w6j(*[_sr(x) for x in (j1,j2,j3,j4,j5,j6)])))
    except Exception: return 0.0

def x_hat(x):
    return np.sqrt(2 * x + 1)

def O_if_lam(Ji, Ii, lam, Jf, li, lf):
    """Angular part for Electric Transitions (Eq 16)"""
    phase = (-1) ** (int(Ji + Ii + lam + lf)) * (1j) ** (li - lf)
    return phase * x_hat(Ji) * x_hat(li) * Wigner6j(Ji, Jf, lam, lf, li, Ii)

def N_if_lam(JA, Ja, Ji, Jf, Ii, If, li):
    """Angular part for Magnetic Transitions (Eq 19)"""
    phase  = (-1) ** (int(JA + Ja - Jf - li))
    hatVal = x_hat(Ji) * x_hat(Ii) * x_hat(JA) * x_hat(If)
    wVal   = Wigner6j(Ji, 1, Jf, If, li, Ii) * Wigner6j(JA, Ja, JA, Ii, 1, If)
    cg     = np.sqrt((JA + 1.0) / JA) if JA > 0 else 0.0
    return phase * hatVal * wVal * cg

def D_if_lam(lam, l_f, Ji, Jf, Ii, If, li):
    """General geometric factor for multipole transitions"""
    phase  = (-1) ** (int(Ji + Ii + lam + l_f)) * (1j) ** (li - l_f)
    hatVal = x_hat(Ji) * x_hat(li)
    cg     = CGC(li, 0, lam, 0, l_f, 0)
    if abs(cg) < 1e-10: return 0.0
    wVal = Wigner6j(Ji, Jf, lam, l_f, li, Ii)
    if abs(wVal) < 1e-10: return 0.0
    return phase * hatVal * wVal * cg
