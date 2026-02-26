import sys, os
import meep as mp
from meep.materials import Au, Ti, SiO2, Pd

from utils.sys_utils import *
from utils.meep_utils import *
from utils.geometry_utils import make_cell
from utils.logger import save_and_show_config
from src.antenna_geometries import *
from src.config import SimulationConfig
from src.sources import *
from src.volumes import *
from visualization.plotter import *

xm = 1000
mp.Simulation.eps_averaging = False

def experiment_bow_tie_test():
    # =====================================================
    config = SimulationConfig()

    for gap in [5, 10, 20, 40, 60]:
        # =====================================================
        SIM_NAME = f"split-bar-bigtest_{gap}nm"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
        # =====================================================
        antenna = BowTieEquilateral(
            gap=gap/xm,
            amp=76/xm,
            thickness=24/xm,
            radius=12/xm,
            material=Au,
            z_offset=0.0
        )
        cell = make_cell(config=config)
        antenna_vols = VolumeSet(cell, antenna=antenna, top_z=antenna.thickness)

        save_and_show_config(config, antenna)

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=antenna.build_geometry(),
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        sim_empty = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=[],
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        # =====================================================
        for plane in ["XY", "XZ"]:
            Name2D = f"antenna_gap_{gap}nm_{plane}.png"
            save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
        draw_dielectric_constant(sim, config, antenna_vols, sampling_wavelength=200)
        draw_dielectric_constant(sim, config, antenna_vols)
        # =====================================================
        print_task(7, "3D calculations.")
        compute_fields(sim, sim_empty, antenna_vols, config)
        # =====================================================
        print_task(8, "Postprocesing - animations and plots.")
        animate_raw_fields(config=config, mode="BOTH")
        # =====================================================
        print_task(9, "Postprocesing - animations and plots.")
        animate_enhancement_fields(config=config)
    return 0