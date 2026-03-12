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

    for gap in [10]:
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
        print_task(1, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_gap_{gap}nm_{plane}.png"
            save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
        # # =====================================================
        # print_task(2, "Dielectric const. plots.")
        # draw_dielectric_constant(sim, config, antenna_vols, sampling_wavelength=200)
        # draw_dielectric_constant(sim, config, antenna_vols)
        # # =====================================================
        # print_task(3, "3D calculations.")
        # compute_fields(sim, sim_empty, antenna_vols, config)
        # # =====================================================
        # print_task(4, "Postprocesing - raw animations.")
        # animate_raw_fields(config=config, mode="BOTH")
        # # =====================================================
        # print_task(5, "Postprocesing - animations and plots.")
        # animate_enhancement_fields(config=config, antenna=antenna)
    return 0

def split_bar_AuTiSiO2():
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500

    for gap in [30, 50, 70, 90, 110]:
        SIM_NAME = f"split_bar_antenna_gap_{gap}nm_AuTiSiO2_test"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
        # =====================================================
        AuTop = SplitBar(
            gap=gap/xm,
            length=1800/xm,
            width=240/xm,
            thickness=30/xm,
            material=Au,
            z_offset=0.0/xm,
            radius=0/xm,
        )
        TiBetween = SplitBar(
            gap=gap/xm,
            length=1800/xm,
            width=240/xm,
            thickness=5/xm,
            material=Ti,
            z_offset=-(30+5)/2.0/xm,
            radius=0/xm,
        )
        substrate = Bar(
            length=4000/xm,
            width=320/xm,
            thickness=70/xm,
            material=SiO2,
            z_offset=-(30/2.0+5+70/2.0)/xm,
            radius=12/xm,
        )

        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + substrate.build_geometry()

        config.pad = 100/xm
        config.pml = 100/xm
        config.cell_size = [
            substrate.length + 2*config.pad + 2*config.pml,   # x
            substrate.width + 2*config.pad + 2*config.pml,   # y
            substrate.thickness+AuTop.thickness+TiBetween.thickness + 2*config.pad + 2*config.pml    # z
        ]
        cell = make_cell(config=config)

        config.src_size = [
            substrate.length,  # x
            substrate.width,  # y
            0.0 / xm    # z
        ]
        config.src_center = [
            0.0,    # x
            0.0,    # y
            config.cell_size[2]/2.0-1.15*config.pml  # z
        ]

        antenna_vols = VolumeSet(cell, antenna=AuTop, top_z=AuTop.thickness)

        save_and_show_config(config, [AuTop, TiBetween, substrate])

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=geometry,
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
        print_task(1, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_{plane}.png"
            save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
        # =====================================================
        print_task(3, "3D calculations.")
        compute_fields(sim, sim_empty, antenna_vols, config)
        # =====================================================
        print_task(4, "Postprocesing - raw animations.")
        animate_raw_fields(config=config, mode="BOTH")
        # =====================================================
        draw_params = {
            "XY": {"x_zoom": 0.10,
                   "y_zoom": 0.6,
                   "roi": {
                        "center": (0, 0),
                        "width": AuTop.gap * 1e3,
                        "height": AuTop.width * 1e3,
                    },
            },
            "XZ": {"x_zoom": 0.1,
                   "y_zoom": 0.2,
                   "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": AuTop.gap * 1e3,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
            "YZ": {"x_zoom": 0.4,
                   "y_zoom": 0.2,
                   "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": AuTop.width * 1e3,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
        }
        print_task(5, "Postprocesing - animations and plots.")
        animate_enhancement_fields(config=config, draw_params=draw_params)

    return 0

def field_shape():
    # =====================================================
    config = SimulationConfig()

    config.sim_time = 15000 / xm
    config.sim_time_step = 100 / xm
    config.src_width = 1000 / xm

    gap = 50
    
    for res in [800]:
        # =====================================================
        SIM_NAME = f"SOURCE_SHAPE_{res}"
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

        config.resolution = res

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
        # # =====================================================
        # print_task(3, "3D calculations.")
        # compute_fields(sim, sim_empty, antenna_vols, config, mode="EMPTY")
        # # =====================================================
        # print_task(4, "Postprocesing - raw animations.")
        # animate_raw_fields(config=config, mode="EMPTY")
        # =====================================================
        plot_signal_amplitude_vs_time_from_h5(
            "xyplanar-empty_ex.h5",
            load_h5data_path=config.path_to_save,
            xzeros=40,
            time_step=config.sim_time_step,
            save_name=f"source_prof_res{res}"
        )
        # =====================================================
    return 0
