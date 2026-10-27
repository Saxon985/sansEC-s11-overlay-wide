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
# Helper: extract temperature from filename
# -------------------------------------------------
def extract_temperature(filename):
    match = re.search(r"[-+]?\d*\.\d+|\d+", filename)
    return float(match.group()) if match else 0.0

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
# Load all .s1p files
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
# Build file info table + extract 80-100 MHz features
# -------------------------------------------------
FREQ_MIN, FREQ_MAX = 80e6, 100e6

file_info = []
temps_list = []
res_freqs_list = []
min_s11_list = []

for file in files:
    temp = extract_temperature(os.path.basename(file))
    try:
        data = np.loadtxt(file, skiprows=1)
        freq = data[:, 0]
        real = data[:, 1]
        imag = data[:, 2]
        
        # Full curve for overlay plot
        s11_full = 20 * np.log10(np.sqrt(real**2 + imag**2))
        
        # Restricted window for temperature analysis
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        if np.any(mask):
            s11_win = 20 * np.log10(np.sqrt(real[mask]**2 + imag[mask]**2))
            min_idx = np.argmin(s11_win)
            res_freq = freq[mask][min_idx] / 1e6
            min_s11 = s11_win[min_idx]
        else:
            res_freq = None
            min_s11 = None
            
        file_info.append({
            "Filename": os.path.basename(file),
            "Temperature (°C)": temp,
            "Resonant Freq (MHz)": round(res_freq, 4) if res_freq is not None else "N/A",
            "Min S11 (dB)": round(min_s11, 3) if min_s11 is not None else "N/A"
        })
        
        if res_freq is not None:
            temps_list.append(temp)
            res_freqs_list.append(res_freq)
            min_s11_list.append(min_s11)
            
    except Exception as e:
        file_info.append({
            "Filename": os.path.basename(file),
            "Temperature (°C)": temp,
            "Resonant Freq (MHz)": "Error",
            "Min S11 (dB)": "Error"
        })

# -------------------------------------------------
# File table + Delete
# -------------------------------------------------
df = pd.DataFrame(file_info)
st.subheader("File Summary")
st.dataframe(df, use_container_width=True)

st.subheader("Delete files")
files_to_delete = st.multiselect("Select files to delete", options=df["Filename"].tolist())
if st.button("Delete selected files") and files_to_delete:
    for fname in files_to_delete:
        os.remove(os.path.join(DATA_DIR, fname))
    st.success(f"Deleted: {', '.join(files_to_delete)}")
    st.rerun()

# -------------------------------------------------
# 1. Original S11 vs Frequency Overlay Plot
# -------------------------------------------------
st.subheader("S11 Magnitude vs Frequency Overlay")

fig1, ax1 = plt.subplots(figsize=(12, 6))

temps_for_color = [extract_temperature(os.path.basename(f)) for f in files]
cmap = plt.cm.plasma
norm = Normalize(vmin=min(temps_for_color), vmax=max(temps_for_color))

for file in files:
    temp = extract_temperature(os.path.basename(file))
    data = np.loadtxt(file, skiprows=1)
    freq_mhz = data[:, 0] / 1e6
    s11_db = 20 * np.log10(np.sqrt(data[:,1]**2 + data[:,2]**2))
    ax1.plot(freq_mhz, s11_db, color=cmap(norm(temp)), linewidth=0.8, alpha=0.85)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label("Temperature (°C)")

ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("S11 Magnitude (dB)")
ax1.set_title("S11 Magnitude vs Frequency (All files)")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# -------------------------------------------------
# 2. Temperature Analysis (80–100 MHz window)
# -------------------------------------------------
if len(temps_list) > 5:
    temps = np.array(temps_list)
    res_freqs = np.array(res_freqs_list)
    min_s11s = np.array(min_s11_list)
    
    # Sort
    sort_idx = np.argsort(temps)
    temps = temps[sort_idx]
    res_freqs = res_freqs[sort_idx]
    min_s11s = min_s11s[sort_idx]
    
    # Fit cubic to depth
    X = temps.reshape(-1, 1)
    poly = PolynomialFeatures(3)
    X_poly = poly.fit_transform(X)
    depth_model = LinearRegression().fit(X_poly, min_s11s)
    depth_r2 = depth_model.score(X_poly, min_s11s)
    coefs = depth_model.coef_
    intercept = depth_model.intercept_
    
    st.subheader("Cubic Equation – Resonance Depth vs Temperature (80–100 MHz)")
    eq_text = (f"S11_min(T) = {intercept:.4f} + ({coefs[1]:.6f})*T + "
               f"({coefs[2]:.6e})*T² + ({coefs[3]:.6e})*T³")
    st.code(eq_text, language="text")
    st.write(f"**R² = {depth_r2:.4f}**  |  Temperature range: {temps.min():.1f} °C – {temps.max():.1f} °C")
    st.caption("Note: Resonant frequency remains essentially constant across this temperature range.")
    
    # Side-by-side temperature plots
    st.subheader("Temperature Dependence (80–100 MHz window)")
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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
    st.pyplot(fig2)
    
    # Download button
    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download temperature analysis plots as PNG",
        data=buf.getvalue(),
        file_name="temperature_analysis.png",
        mime="image/png"
    )
else:
    st.info("Not enough valid files in the 80–100 MHz window to perform temperature analysis.")
