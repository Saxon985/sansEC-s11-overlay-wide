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
    """
    Extracts the first number from the filename.
    Works with names like '37.2C.s1p' or '37.2.s1p'
    """
    match = re.search(r"[-+]?\d*\.\d+|\d+", filename)
    if match:
        return float(match.group())
    return 0.0

# -------------------------------------------------
# Multiple file uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more wide-sweep temperature .s1p files",
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
# Load all existing .s1p files
# -------------------------------------------------
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.s1p")),
    key=lambda x: extract_temperature(os.path.basename(x))
)

file_info = []

for file in files:
    temp = extract_temperature(os.path.basename(file))
    try:
        data = np.loadtxt(file, skiprows=1)
        freq_hz = data[:, 0]
        real = data[:, 1]
        imag = data[:, 2]
        s11_db = 20 * np.log10(np.sqrt(real**2 + imag**2))
        min_idx = np.argmin(s11_db)
        resonant_mhz = freq_hz[min_idx] / 1e6
        min_s11 = s11_db[min_idx]
    except Exception:
        resonant_mhz = None
        min_s11 = None

    file_info.append({
        "Filename": os.path.basename(file),
        "Temperature (°C)": temp,
        "Resonant Freq (MHz)": round(resonant_mhz, 4) if resonant_mhz is not None else "Error",
        "Min S11 (dB)": round(min_s11, 2) if min_s11 is not None else "Error"
    })

st.write(f"Currently stored files: **{len(files)}**")

# -------------------------------------------------
# Table + Delete
# -------------------------------------------------
if file_info:
    df = pd.DataFrame(file_info)
    st.dataframe(df, use_container_width=True)

    st.subheader("Delete files")
    files_to_delete = st.multiselect(
        "Select files to delete",
        options=[info["Filename"] for info in file_info]
    )
    if st.button("Delete selected files") and files_to_delete:
        for fname in files_to_delete:
            os.remove(os.path.join(DATA_DIR, fname))
        st.success(f"Deleted: {', '.join(files_to_delete)}")
        st.rerun()

# -------------------------------------------------
# Plot
# -------------------------------------------------
if len(files) == 0:
    st.info("No .s1p files yet. Upload one or more wide-sweep files to get started.")
else:
    fig, ax = plt.subplots(figsize=(12, 6))

    temps = [extract_temperature(os.path.basename(f)) for f in files]
    cmap = plt.cm.plasma
    norm = Normalize(vmin=min(temps), vmax=max(temps))

    for file in files:
        temp = extract_temperature(os.path.basename(file))
        data = np.loadtxt(file, skiprows=1)
        freq_mhz = data[:, 0] / 1e6
        real = data[:, 1]
        imag = data[:, 2]
        s11_db = 20 * np.log10(np.sqrt(real**2 + imag**2))
        ax.plot(freq_mhz, s11_db, color=cmap(norm(temp)), linewidth=0.8, alpha=0.85)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Temperature (°C)")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 Magnitude (dB)")
    ax.set_title("SansEC S11 Overlay (Wide Sweep: 20 – 200 MHz)")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Download button
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="Download plot as PNG",
        data=buf.getvalue(),
        file_name="s11_overlay_wide.png",
        mime="image/png"
    )