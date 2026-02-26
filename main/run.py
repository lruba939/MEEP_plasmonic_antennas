import sys, os
import meep as mp
from meep.materials import Au, Ti, SiO2, Pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sys_utils import *
from src.taskManager import *
from src.antenna_geometries import BowTieEquilateral
from src.config import SimulationConfig
from src.sources import *
from src.geometry import *
from src.volumes import *

xm = 1000

def run():

    # =====================================================
    config = SimulationConfig()

    for gap in [10, 20, 30, 40, 50]:
        # =====================================================
        SIM_NAME = f"test_AFTER_cleaning_{gap}nm"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)

        # =====================================================
        antenna = BowTieEquilateral(
            gap=gap/xm,
            amp=76/xm,
            thickness=24/xm,
            radius=12/xm,
            material=Au
        )
        cell = make_cell(config=config)
        antenna_vols = VolumeSet(cell, antenna=antenna, top_z=antenna.thickness)

        save_and_show_config(config, antenna)

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=antenna.build_geometry(),
            sources=make_source(),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
        )
        dupa = f"antenna_gap_{gap}nm.png"
        save_2D_plot(antenna_vols.volume["XY"], sim, save_name=dupa, path_to_save=config.path_to_save)

    # =====================================================


    # gaps = [50]  # nm
    # for gap in gaps:
    #     print(f"--- Starting simulation for gap size: {gap} nm ---")
    #     p.gap_size = gap / 1000  # Convert nm to um

    #     ### Set paths to save results
    #     SIM_NAME = f"TEST-gap-{int(gap)}nm-lambda-800nm"
    #     #############################
    
    #     p.path_to_save = os.path.join("results", SIM_NAME)
    #     p.animations_folder_path = os.path.join(p.path_to_save, "animations")

    #     if not os.path.exists(p.path_to_save):
    #         os.makedirs(p.path_to_save)
    #     if not os.path.exists(p.animations_folder_path):
    #         os.makedirs(p.animations_folder_path)

    #     #--- Task 7 ---
    #     print_task(7, "3D calculations.")
    #     compute_fields()

    #     #--- Task 9 ---
    #     print_task(9, "Postprocesing - animations and plots.")
    #     animate_enhancement_fields()

    #     p.reset_to_defaults()
    

    # draw_dielectric_constant(sampling_wavelength=200, sim=sim)
    # draw_dielectric_constant(sim=sim)
    
    
if __name__ == "__main__":
    run()
