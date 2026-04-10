
import numpy as np
from .engine import MU_N, radial_integral
from .angular import O_if_lam, N_if_lam, D_if_lam, x_hat

def get_ef_charge(lam, z1, z2, m1, m2, e2_mevfm):
    mu_val = (m1 * m2) / (m1 + m2)
    return np.sqrt(e2_mevfm) * (mu_val ** lam) * (
        z2 / (m2 ** lam) + ((-1) ** lam) * z1 / (m1 ** lam)
    )

class TransitionModule:
    @staticmethod
    def calculate_e_matrix_element(lam, params, li, Ji, Ii, engine, ef_charge, e_idx=None):
        """Calculates Electric matrix element according to Equation (15)"""
        # Note: lam=1 (E1), lam=2 (E2)
        Jf = params['final_state']['J_f']
        If = params['final_state']['I_f']
        lf = params['final_state']['l_f']
        
        # 1. Geometry factor C_if_lam
        c_val = D_if_lam(lam, lf, Ji, Jf, Ii, If, li)
        if abs(c_val) < 1e-15:
            return 0.0
            
        # 2. Scattering wave function (calculated in loop)
        Ksi_r, _, _, _ = engine.solve_for_channel(li, Ji, Ii, e_idx)
        
        # 3. Radial Integral I_lam
        i_val = radial_integral(params['Fi_r'], Ksi_r, params['r_grid'], lam)
        
        # M_E_lam = e_eff * C_if_lam * I_lam
        return ef_charge * c_val * i_val

    @staticmethod
    def calculate_m1_matrix_element(params, li, Ji, Ii, engine, e_idx=None):
        """Calculates Magnetic matrix element according to Equations (15, 18)"""
        Jf = params['final_state']['J_f']
        If = params['final_state']['I_f']
        lf = params['final_state']['l_f']
        
        # li must be equal to lf for M1 in potential model (unless tensor force)
        if li != lf: return 0.0
        
        orb_part, spin_part = 0.0, 0.0
        
        # 1. Orbital Part (Eq 15)
        if li > 0 and Ii == If:
            c_if1 = O_if_lam(Ji, Ii, 1, Jf, li, lf)
            g_L = params['masses']['m_p'] * (
                params['charges']['z_A'] * params['masses']['ma']**2 + 
                params['charges']['z_a'] * params['masses']['mA']**2
            ) / (params['masses']['mA'] * params['masses']['ma'] * (params['masses']['mA'] + params['masses']['ma']))
            orb_part += g_L * c_if1 * np.sqrt(li * (li + 1))
            
        # 2. Spin Part (Eq 18)
        # S_int = sqrt(3) * mu_A * N_if_lam + sqrt(3) * mu_a * N_if_lam_reversed
        mu_A, mu_a = params['magnetic_moments']['mu_A'], params['magnetic_moments']['mu_a']
        Ja, JA = params['spins']['J_a'], params['spins']['J_A']
        
        if JA > 0:
            spin_part += np.sqrt(3) * mu_A * N_if_lam(JA, Ja, Ji, Jf, Ii, If, li)
        if Ja > 0:
            spin_part += np.sqrt(3) * mu_a * N_if_lam(Ja, JA, Ji, Jf, Ii, If, li)
            
        if abs(orb_part + spin_part) < 1e-20:
            return 0.0
            
        # 3. Overlap Integral I_0
        Ksi_r, _, _, _ = engine.solve_for_channel(li, Ji, Ii, e_idx)
        i0_val = radial_integral(params['Fi_r'], Ksi_r, params['r_grid'], 0)
        
        return (orb_part + spin_part) * MU_N * i0_val
