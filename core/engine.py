
import numpy as np
import numba as nb
from numba import njit
import mpmath
import scipy.integrate as integrate

# Constants
ALPHA    = 1.0 / 137.036
HBAR_C   = 197.327
M_P      = 938.272
E2_MEVFM = ALPHA * HBAR_C
MU_N     = HBAR_C * np.sqrt(E2_MEVFM) / (2.0 * M_P)

@njit(cache=True)
def _numerov_core(r_grid, mu_val, hbar_c_val,
                  V0, rn, a_diff,
                  rc, Z1, Z2, e2,
                  Vso, S_spin,
                  l_val, j_val,
                  E):
    nmax = len(r_grid)
    u = np.zeros(nmax)
    Q = np.zeros(nmax)
    h = r_grid[1] - r_grid[0]
    s_dot_l = 0.5 * (j_val*(j_val+1) - l_val*(l_val+1) - S_spin*(S_spin+1))
    inv_hc2 = 1.0 / (hbar_c_val * hbar_c_val)
    u[0] = 0.0;  u[1] = 1.0e-10
    Q[0] = 0.0;  Q[1] = 0.0

    for i in range(1, nmax - 1):
        ri1 = r_grid[i+1]
        arg_n = (ri1 - rn) / a_diff
        v_nuc = -V0 / (1.0 + np.exp(arg_n)) if arg_n <= 30.0 else 0.0
        v_c   = (Z1*Z2*e2/(2.0*rc))*(3.0-(ri1/rc)**2) if ri1 <= rc else Z1*Z2*e2/ri1
        arg_s = (ri1 - rn) / a_diff
        v_so  = -Vso*s_dot_l/(1.0+np.exp(arg_s)) if arg_s <= 30.0 else 0.0

        f_val = 2.0*mu_val*(v_nuc + v_c + v_so - E)*inv_hc2 + l_val*(l_val+1)/(ri1*ri1)
        Q[i+1] = 1.0 - (h*h/12.0)*f_val
        u[i+1] = ((12.0 - 10.0*Q[i])*u[i] - Q[i-1]*u[i-1]) / Q[i+1]

    return u, Q

class PhysicsEngine:
    def __init__(self, params):
        self.params = params
        self.mu     = params['mu']
        self.Z1     = params['z1']
        self.Z2     = params['z2']
        self.r_grid = params['r_grid']

    def solve_scattering(self, E, l, j, S, V0):
        pot = self.params['pot_params']
        return _numerov_core(
            self.r_grid, self.mu, HBAR_C,
            float(V0), float(pot['rn']), float(pot['a']),
            float(pot['rc']), float(self.Z1), float(self.Z2), float(E2_MEVFM),
            float(pot['Vso']), float(S),
            float(l), float(j), float(E)
        )

    def solve_for_channel(self, l, j, S, e_idx=None, E_rel=None):
        """High-level solver that handles LUT selection and normalization"""
        pot = self.params['pot_params']
        V0 = pot['V02'] if l % 2 == 0 else pot['V01']
        E = E_rel if E_rel is not None else self.params['energy_grid'][e_idx]
        
        u, fn = self.solve_scattering(E, l, j, S, V0)
        
        if e_idx is not None and 'coulomb_lut' in self.params:
            u_norm = self.match_coulomb(u, l, E, self.params['coulomb_lut'], e_idx)
        else:
            u_norm = self.match_coulomb(u, l, E)
            
        return u_norm, 0, fn, 0 # matching other return structures

    def match_coulomb(self, u, la, E, lut=None, e_idx=None):
        k   = np.sqrt(2.0 * self.mu * E) / HBAR_C
        eta = self.Z1 * self.Z2 * ALPHA * np.sqrt(self.mu / (2.0 * E))
        
        if lut and e_idx is not None:
            F1, G1, F2, G2 = lut[(la, e_idx)]
            idx  = self.params['coulomb_idx']
            idx2 = self.params['coulomb_idx2']
        else:
            idx, idx2 = -1, -2
            r1, r2 = self.r_grid[idx], self.r_grid[idx2]
            F1 = float(mpmath.coulombf(la, eta, k * r1))
            G1 = float(mpmath.coulombg(la, eta, k * r1))
            F2 = float(mpmath.coulombf(la, eta, k * r2))
            G2 = float(mpmath.coulombg(la, eta, k * r2))
            
        u1, u2 = u[idx], u[idx2]
        CR    = (G1 - G2 * F1 / F2) / (u1 - u2 * F1 / F2)
        B     = (F2 * u1 / u2 - F1) / (G1 - G2 * u1 / u2)
        delta = np.arctan(B)
        u_norm = u * CR * np.sin(delta)
        return u_norm

def build_coulomb_lut(L_list, Elist, mu, Z1, Z2, r_grid):
    idx  = -1; idx2 = -2
    r1, r2 = r_grid[idx], r_grid[idx2]
    lut = {}
    for la in L_list:
        for ie, E in enumerate(Elist):
            k   = np.sqrt(2.0 * mu * E) / HBAR_C
            eta = Z1 * Z2 * ALPHA * np.sqrt(mu / (2.0 * E))
            F1 = float(mpmath.coulombf(la, eta, k * r1))
            G1 = float(mpmath.coulombg(la, eta, k * r1))
            F2 = float(mpmath.coulombf(la, eta, k * r2))
            G2 = float(mpmath.coulombg(la, eta, k * r2))
            lut[(la, ie)] = (F1, G1, F2, G2)
    return lut, idx, idx2

def radial_integral(Fi_r, Ksi_r, r_grid, lam):
    mask = r_grid > 0.0
    # Orthogonality for M1 (lam=0)
    if lam == 0:
        over = integrate.simpson(Fi_r[mask] * Ksi_r[mask], x=r_grid[mask])
        Ksi_r = Ksi_r - over * Fi_r
        
    integrand = Fi_r[mask] * Ksi_r[mask]
    if lam != 0:
        integrand *= r_grid[mask] ** lam
    return integrate.simpson(integrand, x=r_grid[mask])
