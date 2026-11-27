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
st.title("SansEC S11 Overlay (Temperature & Humidity Analysis)")

# -------------------------------------------------
# Data folder
# -------------------------------------------------
DATA_DIR = "SansEC Data"   
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# Helper: extract value and category based on filename
# -------------------------------------------------
def extract_value_and_category(file_path):
    name = os.path.basename(file_path)
    name_lower = name.lower()
    
    # Check if humidity (% in filename)
    if "%" in name:
        match = re.search(r"([-+]?\d*\.?\d+)\s*%", name)
        val = float(match.group(1)) if match else 0.0
        return val, "Humidity (%)", "Humidity"
    
    # Check if temperature (C in filename)
    elif "c" in name_lower:
        match = re.search(r"([-+]?\d*\.?\d+)\s*c", name_lower)
        val = float(match.group(1)) if match else 0.0
        return val, "Temperature (°C)", "Temperature"
    
    # Fallback default if neither indicator is found
    match = re.search(r"([-+]?\d*\.?\d+)", name)
    val = float(match.group(1)) if match else 0.0
    return val, "Temperature (°C)", "Temperature"

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
# Load all .s1p files recursively and sort them
# -------------------------------------------------
all_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "**", "*.s1p"), recursive=True),
    key=lambda x: extract_value_and_category(x)[0]
)

st.write(f"Total stored files found: **{len(all_files)}**")

if len(all_files) == 0:
    st.info("No .s1p files found yet. Upload your data folders or a .zip archive to get started.")
    st.stop()

# Split into temperature and humidity file lists based on filename rules
temp_files = [f for f in all_files if extract_value_and_category(f)[2] == "Temperature"]
hum_files = [f for f in all_files if extract_value_and_category(f)[2] == "Humidity"]

FREQ_MIN, FREQ_MAX = 80e6, 100e6

# -------------------------------------------------
# Process Dataset Function
# -------------------------------------------------
def process_dataset(file_list, unit_label):
    file_info = []
    values_list = []
    res_freqs_list = []
    min_s11_list = []

    for file in file_list:
        val, _, _ = extract_value_and_category(file)
        rel_path = os.path.relpath(file, DATA_DIR)
        try:
            data = np.loadtxt(file, skiprows=1)
            freq = data[:, 0]
            real = data[:, 1]
            imag = data[:, 2]

            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            if np.any(mask):
                s11_win = 20 * np.log10(np.sqrt(real[mask]**2 + imag[mask]**2))
                min_idx = np.argmin(s11_win)
                res_freq = freq[mask][min_idx] / 1e6
                min_s11 = s11_win[min_idx]
            else:
                res_freq, min_s11 = None, None

            file_info.append({
                "File Path": rel_path,
                unit_label: val,
                "Resonant Freq (MHz)": round(res_freq, 4) if res_freq is not None else "N/A",
                "Min S11 (dB)": round(min_s11, 3) if min_s11 is not None else "N/A"
            })

            if res_freq is not None:
                values_list.append(val)
                res_freqs_list.append(res_freq)
                min_s11_list.append(min_s11)
        except Exception:
            file_info.append({
                "File Path": rel_path,
                unit_label: val,
                "Resonant Freq (MHz)": "Error",
                "Min S11 (dB)": "Error"
            })
    return pd.DataFrame(file_info), values_list, res_freqs_list, min_s11_list

# =================================================
# TEMPERATURE SECTION
# =================================================
if len(temp_files) > 0:
    st.header("🌡️ Temperature Data Analysis")
    df_temp, t_vals, t_freqs, t_s11s = process_dataset(temp_files, "Temperature (°C)")
    st.dataframe(df_temp, use_container_width=True)

    st.subheader("S11 Magnitude vs Frequency Overlay (Temperature)")
    fig_t, ax_t = plt.subplots(figsize=(12, 6))
    t_colors = [extract_value_and_category(f)[0] for f in temp_files]
    cmap_t = plt.cm.plasma
    norm_t = Normalize(vmin=min(t_colors), vmax=max(t_colors))

    for file in temp_files:
        val, _, _ = extract_value_and_category(file)
        data = np.loadtxt(file, skiprows=1)
        ax_t.plot(data[:, 0] / 1e6, 20 * np.log10(np.sqrt(data[:,1]**2 + data[:,2]**2)),
                  color=cmap_t(norm_t(val)), linewidth=0.8, alpha=0.85)

    sm_t = ScalarMappable(cmap=cmap_t, norm=norm_t)
    sm_t.set_array([])
    cbar_t = plt.colorbar(sm_t, ax=ax_t)
    cbar_t.set_label("Temperature (°C)")

    ax_t.set_xlabel("Frequency (MHz)")
    ax_t.set_ylabel("S11 Magnitude (dB)")
    ax_t.set_title("S11 Magnitude vs Frequency (Colored by Temperature)")
    ax_t.grid(True, alpha=0.3)
    st.pyplot(fig_t)

    buf_t = BytesIO()
    fig_t.savefig(buf_t, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download Temperature Overlay Plot as PNG",
        data=buf_t.getvalue(),
        file_name="temperature_s11_overlay.png",
        mime="image/png",
        key="download_temp_overlay"
    )

    if len(t_vals) > 2:
        tv = np.array(t_vals)
        tf = np.array(t_freqs)
        ts = np.array(t_s11s)

        sort_idx = np.argsort(tv)
        tv, tf, ts = tv[sort_idx], tf[sort_idx], ts[sort_idx]

        deg = min(3, len(tv) - 1)
        X = tv.reshape(-1, 1)
        poly = PolynomialFeatures(deg)
        X_poly = poly.fit_transform(X)
        depth_model = LinearRegression().fit(X_poly, ts)
        depth_r2 = depth_model.score(X_poly, ts)
        coefs = depth_model.coef_
        intercept = depth_model.intercept_

        st.subheader("Cubic Equation – Resonance Depth vs Temperature (80–100 MHz)")
        if len(coefs) > 3:
            eq_text = (f"S11_min = {intercept:.4f} + ({coefs[1]:.6f})*x + "
                       f"({coefs[2]:.6e})*x² + ({coefs[3]:.6e})*x³")
        else:
            eq_text = f"S11_min = {intercept:.4f} + " + " + ".join([f"({coefs[i]:.6e})*x^{i}" for i in range(1, len(coefs))])
        st.code(eq_text, language="text")
        st.write(f"**R² = {depth_r2:.4f}**  |  Range: {tv.min():.1f} – {tv.max():.1f} °C")

        st.subheader("Dependence on Temperature (80–100 MHz window)")
        fig_t_ana, axes_t = plt.subplots(1, 2, figsize=(14, 5))
        x_smooth_t = np.linspace(tv.min(), tv.max(), 300).reshape(-1, 1)

        axes_t[0].scatter(tv, tf, color='steelblue', s=40, zorder=3)
        axes_t[0].set_xlabel("Temperature (°C)")
        axes_t[0].set_ylabel("Resonant Frequency (MHz)")
        axes_t[0].set_title("Resonant Frequency vs Temperature")
        axes_t[0].grid(True, alpha=0.3)

        axes_t[1].scatter(tv, ts, color='darkorange', s=40, label='Data', zorder=3)
        axes_t[1].plot(x_smooth_t, depth_model.predict(poly.transform(x_smooth_t)),
                       color='black', linewidth=2, label=f'Fit (R²={depth_r2:.3f})')
        axes_t[1].set_xlabel("Temperature (°C)")
        axes_t[1].set_ylabel("Minimum S11 (dB)")
        axes_t[1].set_title("Resonance Depth vs Temperature")
        axes_t[1].legend()
        axes_t[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_t_ana)

        buf_t_ana = BytesIO()
        fig_t_ana.savefig(buf_t_ana, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            label="Download Temperature Analysis Plots as PNG",
            data=buf_t_ana.getvalue(),
            file_name="temperature_analysis.png",
            mime="image/png",
            key="download_temp_analysis"
        )

# =================================================
# HUMIDITY SECTION
# =================================================
if len(hum_files) > 0:
    st.header("💧 Humidity Data Analysis")
    df_hum, h_vals, h_freqs, h_s11s = process_dataset(hum_files, "Humidity (%)")
    st.dataframe(df_hum, use_container_width=True)

    # Highlight the absolute deepest S11 dip for Humidity
    if len(h_s11s) > 0:
        deepest_idx = np.argmin(h_s11s)
        deepest_hum = h_vals[deepest_idx]
        deepest_freq = h_freqs[deepest_idx]
        deepest_s11 = h_s11s[deepest_idx]
        st.success(
            f"🎯 **Deepest Humidity S11 Dip Detected:** "
            f"**{deepest_s11:.2f} dB** occurs at **{deepest_freq:.4f} MHz** "
            f"under a humidity level of **{deepest_hum}%**."
        )

    st.subheader("S11 Magnitude vs Frequency Overlay (Humidity)")
    fig_h, ax_h = plt.subplots(figsize=(12, 6))
    h_colors = [extract_value_and_category(f)[0] for f in hum_files]
    cmap_h = plt.cm.viridis
    norm_h = Normalize(vmin=min(h_colors), vmax=max(h_colors))

    for file in hum_files:
        val, _, _ = extract_value_and_category(file)
        data = np.loadtxt(file, skiprows=1)
        ax_h.plot(data[:, 0] / 1e6, 20 * np.log10(np.sqrt(data[:,1]**2 + data[:,2]**2)),
                  color=cmap_h(norm_h(val)), linewidth=0.8, alpha=0.85)

    sm_h = ScalarMappable(cmap=cmap_h, norm=norm_h)
    sm_h.set_array([])
    cbar_h = plt.colorbar(sm_h, ax=ax_h)
    cbar_h.set_label("Humidity (%)")

    ax_h.set_xlabel("Frequency (MHz)")
    ax_h.set_ylabel("S11 Magnitude (dB)")
    ax_h.set_title("S11 Magnitude vs Frequency (Colored by Humidity)")
    ax_h.grid(True, alpha=0.3)
    st.pyplot(fig_h)

    buf_h = BytesIO()
    fig_h.savefig(buf_h, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download Humidity Overlay Plot as PNG",
        data=buf_h.getvalue(),
        file_name="humidity_s11_overlay.png",
        mime="image/png",
        key="download_hum_overlay"
    )

    if len(h_vals) > 2:
        hv = np.array(h_vals)
        hf = np.array(h_freqs)
        hs = np.array(h_s11s)

        sort_idx = np.argsort(hv)
        hv, hf, hs = hv[sort_idx], hf[sort_idx], hs[sort_idx]

        deg = min(3, len(hv) - 1)
        X = hv.reshape(-1, 1)
        poly = PolynomialFeatures(deg)
        X_poly = poly.fit_transform(X)
        depth_model_h = LinearRegression().fit(X_poly, hs)
        depth_r2_h = depth_model_h.score(X_poly, hs)
        coefs_h = depth_model_h.coef_
        intercept_h = depth_model_h.intercept_

        st.subheader("Cubic Equation – Resonance Depth vs Humidity (80–100 MHz)")
        if len(coefs_h) > 3:
            eq_text_h = (f"S11_min = {intercept_h:.4f} + ({coefs_h[1]:.6f})*x + "
                         f"({coefs_h[2]:.6e})*x² + ({coefs_h[3]:.6e})*x³")
        else:
            eq_text_h = f"S11_min = {intercept_h:.4f} + " + " + ".join([f"({coefs_h[i]:.6e})*x^{i}" for i in range(1, len(coefs_h))])
        st.code(eq_text_h, language="text")
        st.write(f"**R² = {depth_r2_h:.4f}**  |  Range: {hv.min():.1f} – {hv.max():.1f} %")

        st.subheader("Dependence on Humidity (80–100 MHz window)")
        fig_h_ana, axes_h = plt.subplots(1, 2, figsize=(14, 5))
        x_smooth_h = np.linspace(hv.min(), hv.max(), 300).reshape(-1, 1)

        axes_h[0].scatter(hv, hf, color='teal', s=40, zorder=3)
        axes_h[0].set_xlabel("Humidity (%)")
        axes_h[0].set_ylabel("Resonant Frequency (MHz)")
        axes_h[0].set_title("Resonant Frequency vs Humidity")
        axes_h[0].grid(True, alpha=0.3)

        axes_h[1].scatter(hv, hs, color='purple', s=40, label='Data', zorder=3)
        axes_h[1].plot(x_smooth_h, depth_model_h.predict(poly.transform(x_smooth_h)),
                       color='black', linewidth=2, label=f'Fit (R²={depth_r2_h:.3f})')
        axes_h[1].set_xlabel("Humidity (%)")
        axes_h[1].set_ylabel("Minimum S11 (dB)")
        axes_h[1].set_title("Resonance Depth vs Humidity")
        axes_h[1].legend()
        axes_h[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_h_ana)

        buf_h_ana = BytesIO()
        fig_h_ana.savefig(buf_h_ana, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            label="Download Humidity Analysis Plots as PNG",
            data=buf_h_ana.getvalue(),
            file_name="humidity_analysis.png",
            mime="image/png",
            key="download_hum_analysis"
        )

# -------------------------------------------------
# Delete Files Section
# -------------------------------------------------
st.subheader("Manage Files")
all_paths = [os.path.relpath(f, DATA_DIR) for f in all_files]
files_to_delete = st.multiselect("Select files to delete", options=all_paths)
if st.button("Delete selected files") and files_to_delete:
    for fpath in files_to_delete:
        full_path = os.path.join(DATA_DIR, fpath)
        if os.path.exists(full_path):
            os.remove(full_path)
    st.success(f"Deleted: {', '.join(files_to_delete)}")
    st.rerun()
