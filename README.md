# ⚛️ S-Factor-Pro (Web Version)

S-Factor-Pro is an advanced astrophysical simulation dashboard for calculating radiative capture reactions (including E1, E2, and M1 transitions) from quantum wave functions.

## Features
- **Stateless Cloud Design**: Users can securely upload their bound-state wave function (`wf_bound.txt`).
- **Interactive Visualization**: The calculated astrophysical S-Factor is presented using deep, rich interactable Plotly charts.
- **Data Export**: Users can download the raw output `.csv` data directly from the view.

## Technologies Used
- Streamlit
- Scipy & NumPy (Numerical calculations, integration, Coulomb functions)
- Pandas
- Plotly (Data Visualization)

## Local Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```
