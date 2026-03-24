import sys, os
import meep as mp
from meep.materials import Au, Ti, SiO2, Pd, Al

from utils.sys_utils import *
from utils.meep_utils import *
from utils.geometry_utils import make_cell
from utils.logger import save_and_show_config, append_time_to_file
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
    config.resolution = 250

    gap=20
    # =====================================================
    SIM_NAME = f"bowtie-1000-new-parallel"
    # config.path_to_save="results/bowtie-1000-new-parallel"
    # config.animations_folder_path = "results/bowtie-1000-new-parallel/animations"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    # antenna = BowTieEquilateral(
    #     gap=gap/xm,
    #     amp=76/xm,
    #     thickness=24/xm,
    #     radius=0/xm,
    #     material=Au,
    #     z_offset=0.0
    # )
    antenna = BowTie(
        gap=gap/xm,
        length=150/xm,
        width=100/xm,
        thickness=24/xm,
        radius=5/xm,
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
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # # =====================================================
    # print_task(2, "Dielectric const. plots.")
    # draw_dielectric_constant(sim, config, antenna_vols, sampling_wavelength=200)
    # draw_dielectric_constant(sim, config, antenna_vols)
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(sim, sim_empty, antenna_vols, config)
    # =====================================================
    print_task(4, "Postprocesing - raw animations.")
    animate_raw_fields(config=config, mode="BOTH")
    # =====================================================
    draw_params = {
        "XY": {"x_zoom": 1.0,
                "y_zoom": 1.0,
                "roi": {
                    "center": (0, 0),
                    "width": antenna.gap*1.05 * 1e3,
                    "height": antenna.radius*2.5 * 1e3,
                },
        },
        "XZ": {"x_zoom": 1.0,
                "y_zoom": 1.0,
                "roi": {
                    "center": (0, 0),
                    "width": antenna.gap*1.05 * 1e3,
                    "height": antenna.thickness * 1e3,
                },
        },
        "YZ": {"x_zoom": 1.0,
                "y_zoom": 1.0,
                "roi": {
                    "center": (0, 0),
                    "width": antenna.radius*2.5 * 1e3,
                    "height": antenna.thickness * 1e3,
                },
        },
    }
    print_task(5, "Postprocesing - animations and plots.")
    animate_enhancement_fields(config=config, draw_params=draw_params)
    # =====================================================
    append_time_to_file(config, prefix="Finish: ")
    return 0

def split_bar_AuTiSiO2():
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500

    for gap in [10, 30]:
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
        # # =====================================================
        # print_task(1, "2D projections.")
        # for plane in ["XY", "XZ", "YZ"]:
        #     Name2D = f"antenna_{plane}.png"
        #     save_2D_plot(
        #         sim,
        #         antenna_vols.vis_volume[plane],
        #         save_name=Name2D,
        #         path_to_save=config.path_to_save,
        #         IMG_CLOSE=config.IMG_CLOSE
        #     )
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
        # =====================================================
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar-empty_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(100),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_empty_res{res}"
        # )
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(100),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_antenna_res{res}"
        # )
        
    return 0

def TRA_TEST():
    # =====================================================
    config = SimulationConfig()

    config.sim_time = 5000 / xm
    config.sim_time_step = 100 / xm

    gap = 20
    
    for res in [500]:
        # =====================================================
        SIM_NAME = f"TRA_SHAPE_res{res}"
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
        # =====================================================
        print_task(1, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_{plane}.png"
            save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
        # =====================================================
        print_task(3, "3D calculations.")
        compute_fields(sim, sim_empty, antenna_vols, config, mode="BOTH")
        # =====================================================
        print_task(4, "Postprocesing - raw animations.")
        animate_raw_fields(config=config, mode="EMPTY")
        # =====================================================
        draw_params = {
            "XY": {"x_zoom": 1.,
                   "y_zoom": 1.,
                   "roi": {
                        "center": (0, 0),
                        "width": antenna.gap * 1e3,
                        "height": antenna.gap * 1e3,
                    },
            },
            "XZ": {"x_zoom": 1.,
                   "y_zoom": 1.,
                   "roi": {
                        "center": (0, 0),
                        "width": antenna.gap * 1e3,
                        "height": antenna.thickness * 1e3,
                    },
            },
            "YZ": {"x_zoom": 1.,
                   "y_zoom": 1.,
                   "roi": {
                        "center": (0, 0),
                        "width": antenna.gap * 1e3,
                        "height": antenna.thickness * 1e3,
                    },
            },
        }
        print_task(5, "Postprocesing - animations and plots.")
        animate_enhancement_fields(config=config, draw_params=draw_params)
        # =====================================================
        plot_signal_amplitude_vs_time_from_h5(
            "xyplanar-empty_ex.h5",
            load_h5data_path=config.path_to_save,
            xzeros=int(40*res/800),
            time_step=config.sim_time_step,
            save_name=f"source_prof_empty_res{res}"
        )
        plot_signal_amplitude_vs_time_from_h5(
            "xyplanar_ex.h5",
            load_h5data_path=config.path_to_save,
            xzeros=int(40*res/800),
            time_step=config.sim_time_step,
            save_name=f"source_prof_antenna_res{res}"
        )
        # =====================================================
    return 0

def TRA_Novotn():
    # =====================================================
    config = SimulationConfig()

    config.resolution = 250
    config.sim_time = 25000 / xm
    config.sim_time_step = 100 / xm
    config.lambda0 = 1000 / xm
    config.frequency_width = 0.6
    
    # gap = 10
    
    # =====================================================
    SIM_NAME = f"TRA_Novotny_new"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    antenna = Bar(
        length=110/xm,
        width=20/xm,
        thickness=20/xm,
        material=Au,
        z_offset=0.0,
        radius=10.0/xm,
    )

    config.pml = 500/xm
    config.pad = 100/xm
    config.cell_size = [
        antenna.length + 2*config.pad + 2*config.pml,   # x
        antenna.width + 2*config.pad + 2*config.pml,   # y
        antenna.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        antenna.length + config.pad,  # x
        antenna.width + config.pad,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.05*config.pml  # z
    ]

    antenna_vols = VolumeSet(cell, antenna=antenna, top_z=antenna.thickness/2.0)

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
        Name2D = f"antenna_{plane}.png"
        save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(sim, sim_empty, antenna_vols, config, mode="BOTH")
    # =====================================================
    print_task(4, "Postprocesing - raw animations.")
    animate_raw_fields(config=config, mode="EMPTY")
    # =====================================================
    draw_params = {
        "XY": {"x_zoom": 1.,
               "y_zoom": 1.,
               "roi": {
                    "center": (0, 0),
                    "width": 0,
                    "height": 0,
                },
        },
        "XZ": {"x_zoom": 1.,
               "y_zoom": 1.,
               "roi": {
                    "center": (0, 0),
                    "width": 0,
                    "height": 0,
                },
        },
        "YZ": {"x_zoom": 1.,
               "y_zoom": 1.,
               "roi": {
                    "center": (0, 0),
                    "width": 0,
                    "height": 0,
                },
        },
    }
    print_task(5, "Postprocesing - animations and plots.")
    animate_enhancement_fields(config=config, draw_params=draw_params)
    # =====================================================
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar-empty_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=int(100),
        time_step=config.sim_time_step,
        save_name=f"source_prof_empty_res{res}"
    )
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=int(100),
        time_step=config.sim_time_step,
        save_name=f"source_prof_antenna_res{res}"
    )
    # =====================================================
    return 0

def wave_shape():
    # =====================================================
    config = SimulationConfig()

    mp.is_single_precision() # !!!!!!!

    config.sim_time = 5000 / xm
    config.sim_time_step = 20 / xm
    config.resolution = 350


    for wav in [800, 1200, 1600, 2000, 2400, 2800, 3200]:
        config.lambda0 = wav / xm
        for fwidth in [0.0005, 0.01, 0.2, 0.5, 1.0]:
            config.frequency_width = fwidth

            # =====================================================
            SIM_NAME = f"SHAPE_wav{wav}_fwidth{fwidth}"
            config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
            # =====================================================
            antenna = Bar(
                length=110/xm,
                width=20/xm,
                thickness=20/xm,
                material=Au,
                z_offset=0.0,
                radius=0.0/xm,
            )

            config.pml = 500/xm
            config.pad = 100/xm
            config.cell_size = [
                antenna.length + 2*config.pad + 2*config.pml,   # x
                antenna.width + 2*config.pad + 2*config.pml,   # y
                antenna.thickness + 2*config.pad + 2*config.pml    # z
            ]
            cell = make_cell(config=config)

            config.src_size = [
                antenna.length + config.pad,  # x
                antenna.width + config.pad,  # y
                0.0 / xm    # z
            ]
            config.src_center = [
                0.0,    # x
                0.0,    # y
                config.cell_size[2]/2.0-1.05*config.pml  # z
            ]

            antenna_vols = VolumeSet(cell, antenna=antenna, top_z=antenna.thickness/2.0)

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
            # # =====================================================
            # print_task(1, "2D projections.")
            # for plane in ["XY", "XZ", "YZ"]:
            #     Name2D = f"antenna_{plane}.png"
            #     save_2D_plot(sim, antenna_vols.vis_volume[plane], save_name=Name2D, path_to_save=config.path_to_save, IMG_CLOSE=config.IMG_CLOSE)
            
            # =====================================================
            print_task(3, "3D calculations.")
            compute_fields(sim, sim_empty, antenna_vols, config, mode="EMPTY")
            # =====================================================
            plot_signal_amplitude_vs_time_from_h5(
                "xyplanar-empty_ex.h5",
                load_h5data_path=config.path_to_save,
                xzeros=int(20),
                time_step=config.sim_time_step,
                save_name=f"source_prof_empty"
            )
            # =====================================================
            print_task(4, "Postprocesing - raw animations.")
            animate_raw_fields(config=config, mode="EMPTY")


        # =====================================================
        # print_task(3, "3D calculations.")
        # compute_fields(sim, sim_empty, antenna_vols, config, mode="BOTH")
        # # =====================================================
        # print_task(4, "Postprocesing - raw animations.")
        # animate_raw_fields(config=config, mode="BOTH")
        # # =====================================================
        # draw_params = {
        #     "XY": {"x_zoom": 1.,
        #            "y_zoom": 1.,
        #            "roi": {
        #                 "center": (0, 0),
        #                 "width": antenna.gap * 1e3,
        #                 "height": antenna.gap * 1e3,
        #             },
        #     },
        #     "XZ": {"x_zoom": 1.,
        #            "y_zoom": 1.,
        #            "roi": {
        #                 "center": (0, 0),
        #                 "width": antenna.gap * 1e3,
        #                 "height": antenna.thickness * 1e3,
        #             },
        #     },
        #     "YZ": {"x_zoom": 1.,
        #            "y_zoom": 1.,
        #            "roi": {
        #                 "center": (0, 0),
        #                 "width": antenna.gap * 1e3,
        #                 "height": antenna.thickness * 1e3,
        #             },
        #     },
        # }
        # print_task(5, "Postprocesing - animations and plots.")
        # animate_enhancement_fields(config=config, draw_params=draw_params)
        # # =====================================================
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar-empty_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(40*res/800),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_empty_res{res}"
        # )
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(40*res/800),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_antenna_res{res}"
        # )
        # # =====================================================
    return 0

def wave_shape_theo():
 
    def gaussian_wave_meep(t, f0, fwidth, distance,
                      start_time=0.0, cutoff=3.0, amplitude=1.0):
        w = 1.0 / fwidth
        t0 = start_time + cutoff * w
        t_eff = t - distance
        x = t_eff - t0
        envelope = np.exp(-(x**2) / (2 * w**2))
        carrier = np.cos(-2 * np.pi * f0 * t_eff)
        A = amplitude * fwidth**2

        return A * envelope * carrier

    # parametry
    cutoff = 5
    wavelength = 1.
    f0 = 1 / wavelength

    fw = 0.6

    wstart = 1/(f0 + fw)
    wstop = 1/(f0 - fw)
    
    distance = (420/2.0 - 100*1.05)/xm

    t = np.linspace(0, 25, 2000)

    plt.figure(figsize=(10, 6))

    E_meep = gaussian_wave_meep(t, f0, fw, distance, start_time=0, cutoff=cutoff)
    plt.plot(t, E_meep, label=f"Meep fwidth={fw}")
    plt.xlabel("Time")
    plt.ylabel("E(t)")
    plt.title(f"Gaussian-modulated wave={wavelength:.3f} ({wstart:.3f}-{wstop:.3f})")
    plt.legend()
    plt.grid()
    plt.show()
    return 0

def split_bar_AuTiX():
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 25000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 1000 / xm
    config.frequency_width = 0.6

    gap = 30

    X_materials = [mp.air, SiO2, Au, Al, Pd]
    X_material_names = ["Air", "SiO2", "Au", "Al", "Pd"]
    for X_material, X_material_name in zip(X_materials, X_material_names):
        SIM_NAME = f"split_bar_antenna_AuTi{X_material_name}"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
        # =====================================================
        AuTop = SplitBar(
            gap=gap/xm,
            length=300/xm,
            width=50/xm,
            thickness=30/xm,
            material=Au,
            z_offset=0.0/xm,
            radius=0/xm,
        )
        TiBetween = SplitBar(
            gap=gap/xm,
            length=300/xm,
            width=50/xm,
            thickness=5/xm,
            material=Ti,
            z_offset=-(30+5)/2.0/xm,
            radius=0/xm,
        )
        substrate = Bar(
            length=1000/xm,
            width=200/xm,
            thickness=70/xm,
            material=X_material,
            z_offset=-(30/2.0+5+70/2.0)/xm,
            radius=12/xm,
        )

        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + substrate.build_geometry()

        config.pad = 100/xm
        config.pml = 500/xm
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

        antenna_vols = VolumeSet(cell, antenna=AuTop, top_z=AuTop.thickness, extra_vols_in_gap=True)

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
            save_2D_plot(
                sim,
                antenna_vols.vis_volume[plane],
                save_name=Name2D,
                path_to_save=config.path_to_save,
                IMG_CLOSE=config.IMG_CLOSE
            )
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
        # =====================================================
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar-empty_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(100),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_empty_res{res}"
        # )
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=int(100),
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_antenna_res{res}"
        # ) 
    return 0