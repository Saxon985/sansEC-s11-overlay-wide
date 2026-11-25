import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import pandas as pd
import zipfile
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from io import BytesIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(page_title="SansEC S11 Overlay", layout="wide")
st.title("SansEC S11 Overlay (Wide Sweep: 20 – 200 MHz)")

# -------------------------------------------------
# Data folder
# -------------------------------------------------
DATA_DIR = "SansEC Data"   # you can rename this later if you want
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# Helper: extract value from filename or folder path
# -------------------------------------------------
def extract_value(file_path):
    """
    Handles extracting values and detecting data types from either 
    the filename or parent directory path (e.g., nested test folders).
    """
    path_str = str(file_path)
    name = os.path.basename(file_path)
    
    # Check for humidity indicators in path or name
    is_hum_context = "hum" in path_str.lower() or "%" in path_str
    
    # Try to find a number followed by %
    match = re.search(r"([-+]?\d*\.?\d+)\s*%", path_str)
    if match:
        return float(match.group(1)), "Humidity (%)"
    
    if is_hum_context:
        match = re.search(r"([-+]?\d*\.?\d+)", name)
        if match:
            return float(match.group(1)), "Humidity (%)"

    # Try to find temperature or general number
    match = re.search(r"([-+]?\d*\.?\d+)", name)
    if match:
        val = float(match.group(1))
        if is_hum_context:
            return val, "Humidity (%)"
        return val, "Temperature (°C)"
    
    return 0.0, "Unknown"

# -------------------------------------------------
# File / ZIP Uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload individual .s1p files, folders, or a .zip archive of your SansEC Data",
    type=["s1p", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(uploaded_file.getbuffer())) as z:
                z.extractall(DATA_DIR)
            st.success(f"Extracted zip archive: {uploaded_file.name}")
        else:
            save_path = os.path.join(DATA_DIR, uploaded_file.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

# -------------------------------------------------
# Load all .s1p files recursively from subfolders
# -------------------------------------------------
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.s1p"), recursive=True),
    key=lambda x: extract_value(x)[0]
)

st.write(f"Currently stored files: **{len(files)}**")

if len(files) == 0:
    st.info("No .s1p files found yet. Upload your data folders or a .zip archive to get started.")
    st.stop()

# -------------------------------------------------
# Detect data type (Temperature or Humidity)
# -------------------------------------------------
_, first_unit = extract_value(files[0])
is_humidity = "Humidity" in first_unit
unit_label = first_unit
x_axis_label = unit_label

# -------------------------------------------------
# Build file info table + extract 80-100 MHz features
# -------------------------------------------------
FREQ_MIN, FREQ_MAX = 80e6, 100e6

file_info = []
values_list = []
res_freqs_list = []
min_s11_list = []

for file in files:
    value, _ = extract_value(file)
    rel_path = os.path.relpath(file, DATA_DIR)
    try:
        data = np.loadtxt(file, skiprows=1)
        freq = data[:, 0]
        real = data[:, 1]
        imag = data[:, 2]

        # Restricted window for analysis
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
            "File Path": rel_path,
            unit_label: value,
            "Resonant Freq (MHz)": round(res_freq, 4) if res_freq is not None else "N/A",
            "Min S11 (dB)": round(min_s11, 3) if min_s11 is not None else "N/A"
        })

        if res_freq is not None:
            values_list.append(value)
            res_freqs_list.append(res_freq)
            min_s11_list.append(min_s11)

    except Exception:
        file_info.append({
            "File Path": rel_path,
            unit_label: value,
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
files_to_delete = st.multiselect("Select files to delete", options=df["File Path"].tolist())
if st.button("Delete selected files") and files_to_delete:
    for fpath in files_to_delete:
        full_path = os.path.join(DATA_DIR, fpath)
        if os.path.exists(full_path):
            os.remove(full_path)
    st.success(f"Deleted: {', '.join(files_to_delete)}")
    st.rerun()

# -------------------------------------------------
# 1. S11 vs Frequency Overlay Plot
# -------------------------------------------------
st.subheader("S11 Magnitude vs Frequency Overlay")

fig1, ax1 = plt.subplots(figsize=(12, 6))

values_for_color = [extract_value(f)[0] for f in files]
cmap = plt.cm.plasma
norm = Normalize(vmin=min(values_for_color), vmax=max(values_for_color))

for file in files:
    value, _ = extract_value(file)
    data = np.loadtxt(file, skiprows=1)
    freq_mhz = data[:, 0] / 1e6
    s11_db = 20 * np.log10(np.sqrt(data[:,1]**2 + data[:,2]**2))
    ax1.plot(freq_mhz, s11_db, color=cmap(norm(value)), linewidth=0.8, alpha=0.85)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label(unit_label)

ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("S11 Magnitude (dB)")
ax1.set_title(f"S11 Magnitude vs Frequency (colored by {unit_label})")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# -------------------------------------------------
# 2. Analysis plots (80–100 MHz window)
# -------------------------------------------------
if len(values_list) > 5:
    values = np.array(values_list)
    res_freqs = np.array(res_freqs_list)
    min_s11s = np.array(min_s11_list)

    # Sort
    sort_idx = np.argsort(values)
    values = values[sort_idx]
    res_freqs = res_freqs[sort_idx]
    min_s11s = min_s11s[sort_idx]

    # Fit cubic to resonance depth
    X = values.reshape(-1, 1)
    poly = PolynomialFeatures(3)
    X_poly = poly.fit_transform(X)
    depth_model = LinearRegression().fit(X_poly, min_s11s)
    depth_r2 = depth_model.score(X_poly, min_s11s)
    coefs = depth_model.coef_
    intercept = depth_model.intercept_

    st.subheader(f"Cubic Equation – Resonance Depth vs {unit_label} (80–100 MHz)")
    eq_text = (f"S11_min = {intercept:.4f} + ({coefs[1]:.6f})*x + "
               f"({coefs[2]:.6e})*x² + ({coefs[3]:.6e})*x³")
    st.code(eq_text, language="text")
    st.write(f"**R² = {depth_r2:.4f}**  |  Range: {values.min():.1f} – {values.max():.1f} {unit_label}")

    # Side-by-side plots
    st.subheader(f"Dependence on {unit_label} (80–100 MHz window)")
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_smooth = np.linspace(values.min(), values.max(), 300).reshape(-1, 1)

    # Frequency plot
    axes[0].scatter(values, res_freqs, color='steelblue', s=40, zorder=3)
    axes[0].set_xlabel(x_axis_label)
    axes[0].set_ylabel("Resonant Frequency (MHz)")
    axes[0].set_title(f"Resonant Frequency vs {unit_label}")
    axes[0].grid(True, alpha=0.3)

    # Depth plot + fit
    axes[1].scatter(values, min_s11s, color='darkorange', s=40, label='Data', zorder=3)
    axes[1].plot(x_smooth, depth_model.predict(poly.transform(x_smooth)),
                 color='black', linewidth=2, label=f'Cubic fit (R²={depth_r2:.3f})')
    axes[1].set_xlabel(x_axis_label)
    axes[1].set_ylabel("Minimum S11 (dB)")
    axes[1].set_title(f"Resonance Depth vs {unit_label}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

    # Download button
    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download analysis plots as PNG",
        data=buf.getvalue(),
        file_name="sansEC_analysis.png",
        mime="image/png"
    )
else:
    st.info("Not enough valid files in the 80–100 MHz window to perform analysis.")
