import sys, os, meep
from meep.materials import Au, Ti, SiO2, Pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.taskManager import *
from utils.sys_utils import *
from src.antenna_geometries import BowTieEquilateral
from src.config import SimulationConfig
from src.sources import *
from src.geometry import *

xm = 1000

def run():

    ### Set paths to save results
    SIM_NAME = "test_AFTER_cleaning"
    #############################
    
    p.path_to_save = os.path.join("results", SIM_NAME)
    p.animations_folder_path = os.path.join(p.path_to_save, "animations")
 
    if not os.path.exists(p.path_to_save):
        os.makedirs(p.path_to_save)
    if not os.path.exists(p.animations_folder_path):
        os.makedirs(p.animations_folder_path)
        
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
    
    p.IMG_CLOSE = True
    # save_2D_plot(p.xy_plane)
    
    antenna = BowTieEquilateral(
        gap=16/xm,
        amp=76/xm,
        thickness=24/xm,
        radius=12/xm,
        # flare_angle_deg=60.0,
        material=Au
    )

    sim = mp.Simulation(
        cell_size=make_cell(),
        boundary_layers=[mp.PML(p.pml)],
        geometry=antenna.build_geometry(),
        sources=make_source(),
        resolution = p.resolution,
        k_point = mp.Vector3(),
        symmetries=p.symmetries,
        dimensions=p.sim_dimensions
    )

    draw_dielectric_constant(sampling_wavelength=200, sim=sim)
    draw_dielectric_constant(sim=sim)
    
    
if __name__ == "__main__":
    run()
