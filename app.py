import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
import scipy.special as sc
from io import StringIO

# Import physics core
from core.engine import PhysicsEngine, build_coulomb_lut, HBAR_C, ALPHA, E2_MEVFM
from core.transitions import TransitionModule, get_ef_charge

st.set_page_config(page_title="S-Factor-Pro Web", layout="wide", page_icon="⚛️")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main { background: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1 { color: #2c3e50; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #4ECDC4; color: white; border-radius: 0.5rem; border: none; padding: 0.5rem 2rem; font-weight: 600; }
    .stButton>button:hover { background-color: #45B7D1; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("⚛️ S-Factor-Pro (Web Version)")
st.markdown("Cloud-optimized astrophysical simulation dashboard. Upload your wave functions and run calculations directly in the browser.")

class SFactorWebEngine:
    def __init__(self, params, f_interp, energy_grid):
        self.p = params
        self.f_interp = f_interp
        self.energy_grid = energy_grid
        self.L_MAP = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}
        self._setup_engine()
        
    def _setup_engine(self):
        mA, ma = self.p['masses']['mA'], self.p['masses']['ma']
        mu = (mA * ma) / (mA + ma)
        self.r_grid = np.linspace(1e-2, 40.0, 4001)
        
        Eb = self.p['final_state']['Eb']
        z1, z2 = self.p['charges']['z_a'], self.p['charges']['z_A']
        lf = int(self.p['final_state']['l_f'])
        
        kb = np.sqrt(2 * mu * Eb) / HBAR_C
        eta_b = z1 * z2 * ALPHA * np.sqrt(mu / (2 * Eb))
        
        RMatch = 10.0
        z10 = 2 * kb * RMatch
        W10 = np.exp(-z10 / 2) * (z10 ** (-eta_b)) * sc.hyperu(lf + 1 + eta_b, 2 * lf + 2, z10)
        file_val = self.f_interp(RMatch)
        ANC = file_val / W10
        
        r_tail = self.r_grid[self.r_grid >= RMatch]
        z_r = 2 * kb * r_tail
        W_vals = np.exp(-z_r / 2) * (z_r ** (-eta_b)) * sc.hyperu(lf + 1 + eta_b, 2 * lf + 2, z_r)
        
        Fi_r_1 = self.f_interp(self.r_grid[self.r_grid < RMatch])
        Fi_r_2 = ANC * W_vals
        self.Fi_r = np.concatenate((Fi_r_1, Fi_r_2))
        
        self.p['Fi_r'] = self.Fi_r
        self.p['r_grid'] = self.r_grid
        self.p['mu'] = mu
        self.p['z1'] = z1
        self.p['z2'] = z2
        self.p['energy_grid'] = self.energy_grid
        self.engine = PhysicsEngine(self.p)
        
    def run(self):
        l_list = set()
        for t_type, channels in self.p['transitions'].items():
            for ch in channels: l_list.add(int(ch['li']))
        
        lut, idx, idx2 = build_coulomb_lut(list(l_list), self.energy_grid, 
                                           self.p['mu'], self.p['z1'], 
                                           self.p['z2'], self.r_grid)
        self.p['coulomb_lut'] = lut
        self.p['coulomb_idx'] = idx
        self.p['coulomb_idx2'] = idx2
        
        results = []
        tm = TransitionModule()
        
        progress_bar = st.progress(0)
        for ie, E_rel in enumerate(self.energy_grid):
            k_i = np.sqrt(2 * self.p['mu'] * E_rel) / HBAR_C
            k_gamma = (E_rel + self.p['final_state']['Eb']) / HBAR_C
            eta = self.p['z1'] * self.p['z2'] * ALPHA * np.sqrt(self.p['mu'] / (2 * E_rel))
            
            row = {'E_MeV': E_rel}
            
            for t_type, channels in self.p['transitions'].items():
                total_sigma_t = 0.0
                lam = 1 if t_type == 'E1' else (2 if t_type == 'E2' else 0)
                
                if t_type.startswith('E'):
                    c_vals = {1: 2/3, 2: 1/30}
                    pref = (2 * self.p['final_state']['J_f'] + 1) / (
                        (2 * self.p['spins']['J_a'] + 1) * (2 * self.p['spins']['J_A'] + 1)
                    ) * 4 * np.pi * c_vals[lam] * (k_gamma ** (2 * lam + 1)) / (k_i * E_rel)
                    ef_charge = get_ef_charge(lam, self.p['charges']['z_a'], self.p['charges']['z_A'], 
                                             self.p['masses']['ma'], self.p['masses']['mA'], E2_MEVFM)
                else:
                    pref = (2 * self.p['final_state']['J_f'] + 1) / (
                        (2 * self.p['spins']['J_a'] + 1) * (2 * self.p['spins']['J_A'] + 1)
                    ) * 4 * np.pi * (2/3) * (k_gamma ** 3) / (k_i * E_rel)

                for ch in channels:
                    li = ch['li']
                    Ii = ch['Ii']
                    for Ji in ch['Ji']:
                        if t_type.startswith('E'):
                            M = tm.calculate_e_matrix_element(lam, self.p, li, Ji, Ii, self.engine, ef_charge, ie)
                        else:
                            M = tm.calculate_m1_matrix_element(self.p, li, Ji, Ii, self.engine, ie)
                        
                        term_sigma = pref * (np.abs(M)**2)
                        term_s = term_sigma * E_rel * np.exp(2 * np.pi * eta) * 0.01 * 1e9
                        
                        label = f"{t_type}_3{self.L_MAP[int(li)]}{int(Ji)}"
                        row[label] = row.get(label, 0.0) + term_s
                        total_sigma_t += term_s
                
                row[f"{t_type}_total"] = total_sigma_t
            
            results.append(row)
            if ie % max(1, len(self.energy_grid)//20) == 0: 
                progress_bar.progress(ie / len(self.energy_grid))
                
        progress_bar.progress(1.0)
        return pd.DataFrame(results)

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("🔬 Parameters")
    
    with st.expander("Masses & Charges", expanded=True):
        mA = st.number_input("Mass mA (MeV)", value=3727.33, format="%.2f")
        ma = st.number_input("Mass ma (MeV)", value=1875.59, format="%.2f")
        zA = st.number_input("Charge zA", value=2, step=1)
        za = st.number_input("Charge za", value=1, step=1)
        
    with st.expander("Potentials", expanded=True):
        V01 = st.slider("V01 (MeV)", 50.0, 100.0, 75.23)
        V02 = st.slider("V02 (MeV)", 50.0, 100.0, 79.23)
        rc = st.number_input("rc (fm)", value=1.85, format="%.2f")
        rn = st.number_input("rn (fm)", value=1.85, format="%.2f")
        a = st.number_input("Diffuseness a (fm)", value=0.71, format="%.2f")
        Vso = st.number_input("Spin-Orbit Vso", value=3.305, format="%.3f")

    st.header("⚡ Simulation Grid")
    e_max = st.slider("Max Energy (MeV)", 0.5, 5.0, 2.0)
    n_pts = st.slider("Number of points", 20, 200, 80)
    
    st.header("📁 Data")
    uploaded_file = st.file_uploader("Upload wf_bound.txt", type=['txt', 'dat'])

if uploaded_file is None:
    st.warning("Пожалуйста, загрузите файл волновой функции wf_bound.txt в бонковом меню слева.")
else:
    if st.button("🚀 Run Cloud Simulation"):
        try:
            # Read uploaded file
            string_data = uploaded_file.getvalue().decode("utf-8")
            dat = pd.read_table(StringIO(string_data), sep=r'\s+', header=None)
            f_interp = interp1d(dat.iloc[:, 0], dat.iloc[:, 1], fill_value="extrapolate")
            
            params = {
                'masses': {'mA': mA, 'ma': ma, 'm_p': 938.27},
                'charges': {'z_A': zA, 'z_a': za},
                'magnetic_moments': {'mu_A': 0.0, 'mu_a': 0.857},
                'spins': {'J_a': 1.0, 'J_A': 0.0},
                'final_state': {'l_f': 0, 'I_f': 1.0, 'j_f': 1.0, 'J_f': 1.0, 'Eb': 1.4753},
                'pot_params': {'V01': V01, 'V02': V02, 'rc': rc, 'rn': rn, 'a': a, 'Vso': Vso},
                'transitions': {
                    'E1': [{'li': 1, 'Ii': 1.0, 'Ji': [0.0, 1.0, 2.0]}],
                    'E2': [{'li': 2, 'Ii': 1.0, 'Ji': [1.0, 2.0, 3.0]}],
                    'M1': [{'li': 0, 'Ii': 1.0, 'Ji': [1.0]}]
                }
            }
            
            e_grid = np.linspace(0.005, e_max, n_pts)
            
            with st.spinner("Calculating S-factors on cloud server..."):
                engine = SFactorWebEngine(params, f_interp, e_grid)
                df = engine.run()
                
            st.success("Simulation Complete!")
            
            st.subheader("📊 Astrophysical S-Factor Results")
            totals = [c for c in df.columns if c.endswith('_total')]
            fig = go.Figure()
            
            grand_total = np.zeros(len(df))
            for col in totals:
                fig.add_trace(go.Scatter(x=df['E_MeV'], y=df[col], name=col.replace('_', ' ').title(), mode='lines'))
                grand_total += df[col]
            
            fig.add_trace(go.Scatter(x=df['E_MeV'], y=grand_total, name="Total S-factor", line=dict(color='black', width=3)))
            
            fig.update_layout(
                xaxis_title="Energy E_rel (MeV)",
                yaxis_title="S-factor (MeV nb)",
                yaxis_type="log",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Raw Data")
                st.dataframe(df.head(10))
            with col2:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download results.csv", data=csv, file_name="sfactor_results.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"Error reading file or running simulation: {str(e)}")
