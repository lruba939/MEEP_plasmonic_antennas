import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from visualization.plotter import *
import glob
import re

def plot_ex_spectra_with_offset(folder, offset_step=1.0):
    files = glob.glob(f"{folder}/Ex_z_*.txt")

    data_list = []

    # --- extract z from filename ---
    for f in files:
        match = re.search(r"z_([-\d]+p\d+)", f)
        if match:
            z_str = match.group(1).replace("p", ".")
            z_val = float(z_str)
            data_list.append((z_val, f))

    # --- sort: MIN -> MAX ---
    data_list.sort(key=lambda x: x[0])

    z_values = np.array([z for z, _ in data_list])

    # --- normalize Z to [0,1] ---
    z_min, z_max = z_values.min(), z_values.max()
    z_norm = (z_values - z_min) / (z_max - z_min)

    cmap = plt.get_cmap("inferno")

    plt.figure(figsize=(5, 6))

    for i, ((z, f), zn) in enumerate(zip(data_list, z_norm)):
        data = np.loadtxt(f)

        wavelength = data[:, 0]*1e3
        enhancement = data[:, 3]

        offset = i * offset_step
        color = cmap(zn*0.8)

        y = enhancement + offset
        plt.plot(wavelength, y, color=color)

        z_nm = z *1e3

        # --- tekst przy końcu linii ---
        plt.text(
            wavelength[-1] * 1.01,   # lekko w prawo
            y[-1]+200,
            f"{z_nm:.0f} nm",
            fontsize=8,
            color=color,
            va='center'
        )

    plt.ylim([-500, 22000])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Enhancement + offset [a.u.]")

    plt.tight_layout()

    # --- SAVE ---
    save_path = os.path.join(folder, "gap_spectra_long.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
        
def main():

    for x in ["Cr", "Ti", "Ni", "Al2O3"]:
        file = f"results/bowtie_Xiong_Au{x}_wavleng_0.66_gap_6/gap_spec/"
        plot_ex_spectra_with_offset(file, 1000)
        
    for x in ["SiO2", "Au"]:
        file = f"results/TEST_DFT_bowtie_Xiong_Au{x}_wavleng_0.65_gap_6/gap_spec/"
        plot_ex_spectra_with_offset(file, 1000)

if __name__ == "__main__":
    main()
