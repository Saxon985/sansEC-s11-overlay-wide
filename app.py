import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from io import BytesIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(page_title="SansEC S11 Overlay - Wide Sweep", layout="wide")
st.title("SansEC S11 Overlay (Wide Sweep: 20 – 200 MHz)")

# -------------------------------------------------
# Ensure data folder exists
# -------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def extract_temperature(filename):
    match = re.search(r"[-+]?\d*\.\d+|\d+", filename)
    return float(match.group()) if match else 0.0

def load_and_analyze(files):
    FREQ_MIN, FREQ_MAX = 80e6, 100e6
    temps, res_freqs, min_s11s = [], [], []
    
    for file in files:
        temp = extract_temperature(os.path.basename(file))
        try:
            data = np.loadtxt(file, skiprows=1)
            freq, real, imag = data[:, 0], data[:, 1], data[:, 2]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            if not np.any(mask):
                continue
            s11_db = 20 * np.log10(np.sqrt(real[mask]**2 + imag[mask]**2))
            min_idx = np.argmin(s11_db)
            temps.append(temp)
            res_freqs.append(freq[mask][min_idx] / 1e6)
            min_s11s.append(s11_db[min_idx])
        except:
            continue
    
    return np.array(temps), np.array(res_freqs), np.array(min_s11s)

# -------------------------------------------------
# File uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more wide-sweep .s1p files",
    type=["s1p"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")

# -------------------------------------------------
# Load files
# -------------------------------------------------
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.s1p")),
    key=lambda x: extract_temperature(os.path.basename(x))
)

st.write(f"Currently stored files: **{len(files)}**")

if len(files) == 0:
    st.info("No .s1p files yet. Upload one or more to get started.")
    st.stop()

# -------------------------------------------------
# Extract features (80–100 MHz window)
# -------------------------------------------------
temps, res_freqs, min_s11s = load_and_analyze(files)

if len(temps) == 0:
    st.warning("No valid data found in the 80–100 MHz window.")
    st.stop()

# Sort
idx = np.argsort(temps)
temps, res_freqs, min_s11s = temps[idx], res_freqs[idx], min_s11s[idx]

# -------------------------------------------------
# Table of results
# -------------------------------------------------
df = pd.DataFrame({
    "Filename": [os.path.basename(f) for f in files],
    "Temperature (°C)": [extract_temperature(os.path.basename(f)) for f in files]
})
# Re-order df to match sorted temps if needed – simplified for now
st.subheader("File Summary")
st.dataframe(df, use_container_width=True)

# -------------------------------------------------
# Fit cubic to depth
# -------------------------------------------------
X = temps.reshape(-1, 1)
poly = PolynomialFeatures(3)
X_poly = poly.fit_transform(X)
depth_model = LinearRegression().fit(X_poly, min_s11s)
depth_r2 = depth_model.score(X_poly, min_s11s)

coefs = depth_model.coef_
intercept = depth_model.intercept_

# -------------------------------------------------
# Display the equation
# -------------------------------------------------
st.subheader("Cubic Equation – Resonance Depth vs Temperature")

eq_text = (
    f"S11_min(T) = {intercept:.4f} + ({coefs[1]:.6f})*T "
    f"+ ({coefs[2]:.6e})*T² + ({coefs[3]:.6e})*T³"
)
st.code(eq_text, language="text")

st.write(f"**R² = {depth_r2:.4f}**  |  Valid range: {temps.min():.1f} °C – {temps.max():.1f} °C")
st.caption("Note: Resonant frequency remains essentially constant (~99.2–99.3 MHz) across this temperature range.")

# -------------------------------------------------
# Plots
# -------------------------------------------------
st.subheader("Plots (80–100 MHz window)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

T_smooth = np.linspace(temps.min(), temps.max(), 300).reshape(-1, 1)

# Frequency plot
axes[0].scatter(temps, res_freqs, color='steelblue', s=40, zorder=3)
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Resonant Frequency (MHz)")
axes[0].set_title("Resonant Frequency vs Temperature")
axes[0].grid(True, alpha=0.3)

# Depth plot + fit
axes[1].scatter(temps, min_s11s, color='darkorange', s=40, label='Data', zorder=3)
axes[1].plot(T_smooth, depth_model.predict(poly.transform(T_smooth)),
             color='black', linewidth=2, label=f'Cubic fit (R²={depth_r2:.3f})')
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Minimum S11 (dB)")
axes[1].set_title("Resonance Depth vs Temperature")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Download button for the figure
buf = BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
st.download_button(
    label="Download plots as PNG",
    data=buf.getvalue(),
    file_name="depth_and_frequency_vs_temperature.png",
    mime="image/png"
)
