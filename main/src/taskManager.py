from . import params
from . import containters
from . import simulation

import meep as mp
from visualization.plotter import *
import numpy as np
import os

from utils.meep_utils import *

# inicialize singleton of all parameters
p = params.SimParams()
con = containters.SimContainers()

# TASK 0 -------------------------------
# Triggering calculations and saving the most general results.

def task_0():
    
    p.showParams()

    if not os.path.exists(p.path_to_save):
        os.makedirs(p.path_to_save)
    if not os.path.exists(p.animations_folder_path):
        os.makedirs(p.animations_folder_path)
    
    p.saveParams(filename=os.path.join(p.path_to_save, "simulation_params.txt"))

    simulation.start_empty_cell_calc() # MUST BE CALLED FIRST
    sim = simulation.make_sim()
    simulation.start_calc(sim)

    # np.savez(
    #     os.path.join(p.path_to_save, "data_general.npz"),
    #     Ey = con.E_comp_data_container,
    #     Ey_empty = con.empty_cell_E_comp_data_container,
    #     eps = con.eps_data_container
    #     )
    
    return sim

# TASK 1 -------------------------------
# Make medium - a split bar antenna.

def task_1():
    p.showParams()
    sim = simulation.make_sim()
    sim.plot2D()
    plt.savefig(os.path.join(p.path_to_save, "2Dplot.png"), dpi=300, bbox_inches="tight", format="png")
    if p.IMG_CLOSE:
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    else:
        plt.show()

    return 0

# TASK 2 -------------------------------
# Plotting the dielectric constant of a system.

def task_2(plot=False, recalculate=False):
    
    p.showParams()

    if recalculate:    
        sim = simulation.make_sim()
        simulation.start_calc(sim)
      
    if plot:
        show_data_img(datas_arr =   [con.eps_data_container],
                      norm_bool =   [True],
                      abs_bool  =   [True],
                      cmap_arr  =   ["binary"],
                      alphas    =   [1.0],
                      IMG_CLOSE =   p.IMG_CLOSE)
    
    return 0

# TASK 3 -------------------------------
# Calculations of the scalar electric field E component and optional plotting and animation.

def task_3(plot=False, animation=False, animation_name="animation",
           plot_3D=False, sim=None,
           recalculate=False):
    p.showParams()
    
    if recalculate:
        sim = simulation.make_sim()
        simulation.start_calc(sim)
    
    if plot:
        show_data_img(datas_arr =   [con.eps_data_container, con.E_comp_data_container],
                      norm_bool =   [True, False],
                      abs_bool  =   [True, False],
                      cmap_arr  =   ["binary", "RdBu"],
                      alphas    =   [1.0, 0.9],
                      name_to_save = os.path.join(p.path_to_save, "E_component_with_antennas"),
                      IMG_CLOSE =   p.IMG_CLOSE)
        
    if plot:
        show_data_img(datas_arr =   [con.empty_cell_E_comp_data_container],
                      norm_bool =   [False],
                      abs_bool  =   [False],
                      cmap_arr  =   ["RdBu"],
                      alphas    =   [0.9],
                      name_to_save = os.path.join(p.path_to_save, "E_component_empty_cell"),
                      IMG_CLOSE =   p.IMG_CLOSE)
        
    if animation:
        make_animation(p, sim, animation_name)
        
    collected_data, time_steps, x_coords = collect_e_line(p, sim, delta_t=p.animations_step, width=5, plot_3d=plot_3D, name=animation_name)
    np.savez(
        os.path.join(p.path_to_save, f"data_E_line_{animation_name}.npz"),
        collected_data=collected_data,
        time_steps=time_steps,
        x_coords=x_coords
        )

    return 0
           
# TASK 4 -------------------------------

def task_4(skip_fraction=0.15, E_plot=False):
    p.reset_to_defaults()
    
    # --- With antennas ---
    sim = simulation.make_sim()
    E_max_with = collect_max_field(p, sim, delta_t=p.animations_step, skip_fraction=skip_fraction, optional_name="with_antennas")
    if E_plot:
        show_data_img(datas_arr =   [E_max_with],
                        norm_bool =   [False],
                        abs_bool  =   [False],
                        cmap_arr  =   ["inferno"],
                        alphas    =   [1.0],
                        name_to_save = os.path.join(p.path_to_save, "Max_E_with_antennas"),
                        IMG_CLOSE =   p.IMG_CLOSE)
    
    # --- Without antennas ---
    p.center = [mp.Vector3(0, 0, -10.), 
                mp.Vector3(0, 0, -10.)]
    p.bowtie_center = [-9999, -9999]
    sim = simulation.make_sim()
    E_max_without = collect_max_field(p, sim, delta_t=p.animations_step, skip_fraction=skip_fraction, optional_name="without_antennas")
    if E_plot:
        show_data_img(datas_arr =   [E_max_without],
                        norm_bool =   [False],
                        abs_bool  =   [False],
                        cmap_arr  =   ["inferno"],
                        alphas    =   [1.0],
                        name_to_save = os.path.join(p.path_to_save, "Max_E_without_antennas"),
                        IMG_CLOSE =   p.IMG_CLOSE)
    
    # --- Gain ---   
    gain = E_max_with / E_max_without

    # --- Outliers clipping ---
    vmin = np.nanpercentile(gain, 1)
    vmax = np.nanpercentile(gain, 99)
    gain_clipped = np.clip(gain, vmin, vmax)
    show_data_img(datas_arr =   [gain_clipped],
                    norm_bool =   [False],
                    abs_bool  =   [False],
                    cmap_arr  =   ["inferno"],
                    alphas    =   [1.0],
                    name_to_save = os.path.join(p.path_to_save, "Gain_linear_scale"),
                    IMG_CLOSE =   p.IMG_CLOSE)
    
    # --- Gain in dB ---
    gain_db = 20.0 * np.log10(gain + 1e-12)
    gain_db_clipped = np.clip(gain_db, vmin, vmax)
    show_data_img(datas_arr =   [gain_db_clipped],
                    norm_bool =   [False],
                    abs_bool  =   [False],
                    cmap_arr  =   ["inferno"],
                    alphas    =   [1.0],
                    name_to_save = os.path.join(p.path_to_save, "Gain_dB_scale"),
                    IMG_CLOSE =   p.IMG_CLOSE)
    
    np.savez(
        os.path.join(p.path_to_save, "data_enhancement.npz"),
        E_max_with=np.array(E_max_with),
        E_max_without=np.array(E_max_without),
        gain_clipped=gain_clipped,
        gain_db_clipped=gain_db_clipped
        )

    return gain_db_clipped

# TASK 5 -------------------------------

def task_5():
    sim = simulation.make_sim()
    collected_data = collect_data_in_time(singleton_params=p, sim=sim,
                                          delta_t=p.animations_step, clear_pml=True,
                                          Ex=[], Ey=[],
                                          E2=[], H2=[])
    make_field_animation(collected_data, field_name='Ex', singleton_params=p,
                         animation_name='Ex', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)
    
    make_field_animation(collected_data, field_name='Ey', singleton_params=p,
                         animation_name='Ey', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)
    
    make_field_animation(collected_data, field_name='E2', singleton_params=p,
                         animation_name='E2', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)
    
    make_field_animation(collected_data, field_name='H2', singleton_params=p,
                         animation_name='H2', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

# TASK 6 -------------------------------

def task_6():
    # --- With antennas ---
    sim = simulation.make_sim()
    collected_data = collect_data_in_time(singleton_params=p, sim=sim,
                                          delta_t=p.animations_step, clear_pml=True,
                                          Ex=[], Ey=[],
                                          E2=[], H2=[])
    # --- Without antennas ---
    p.center = [mp.Vector3(0, 0, -10.), 
                mp.Vector3(0, 0, -10.)]
    p.bowtie_center = [-9999, -9999]
    sim = simulation.make_sim()
    empty_collected_data = collect_data_in_time(singleton_params=p, sim=sim,
                                        delta_t=p.animations_step, clear_pml=True,
                                        Ex=[], Ey=[],
                                        E2=[], H2=[])

    make_field_animation(empty_collected_data, field_name='Ex', singleton_params=p,
                         animation_name='empty_Ex', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(empty_collected_data, field_name='Ey', singleton_params=p,
                         animation_name='empty_Ey', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(empty_collected_data, field_name='E2', singleton_params=p,
                         animation_name='empty_E2', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    Ex_enhancement = calculate_field_enhancement(collected_data['Ex'], empty_collected_data['Ex'], singleton_params=p)
    collected_data["Ex_enhancement"] = Ex_enhancement
    E2_enhancement = calculate_field_enhancement(collected_data['E2'], empty_collected_data['E2'], singleton_params=p)
    collected_data["E2_enhancement"] = E2_enhancement
    H2_enhancement = calculate_field_enhancement(collected_data['H2'], empty_collected_data['H2'], singleton_params=p)
    collected_data["H2_enhancement"] = H2_enhancement

    make_field_animation(collected_data, field_name='Ex_enhancement', singleton_params=p,
                         animation_name='Ex_enhancement', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(collected_data, field_name='E2_enhancement', singleton_params=p,
                         animation_name='E2_enhancement', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(collected_data, field_name='H2_enhancement', singleton_params=p,
                         animation_name='H2_enhancement', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)
    
    Ex_ratio = calculate_field_ratio(collected_data['Ex'], empty_collected_data['Ex'])
    collected_data["Ex_ratio"] = Ex_ratio
    E2_ratio = calculate_field_ratio(collected_data['E2'], empty_collected_data['E2'])
    collected_data["E2_ratio"] = E2_ratio
    H2_ratio = calculate_field_ratio(collected_data['H2'], empty_collected_data['H2'])
    collected_data["H2_ratio"] = H2_ratio

    make_field_animation(collected_data, field_name='Ex_ratio', singleton_params=p,
                         animation_name='Ex_ratio', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(collected_data, field_name='E2_ratio', singleton_params=p,
                         animation_name='E2_ratio', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)

    make_field_animation(collected_data, field_name='H2_ratio', singleton_params=p,
                         animation_name='H2_ratio', cmap=cm_rdbu,
                         structure=con.eps_data_container, crop_pml=True)
    
# TASK 7 -------------------------------

def task_7():
    
    sim = simulation.make_sim()   
    
    #####################
    ### With antennas ###
    #####################

    collect_fields_with_output(
        sim,
        volumes={
        "xyplanar-bowtie": p.xy_plane,
        "xzplanar-bowtie": p.xz_plane,
        },
        delta_t=p.animations_step,
        until=p.sim_time,
        start_time=0
    )
    
    animate_field_from_h5(
        h5_filename="run-xyplanar-bowtie_ex.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="xyplanar-bowtie_ex.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )
    animate_field_from_h5(
        h5_filename="run-xzplanar-bowtie_ex.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="xzplanar-bowtie_ex.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )

    ########################
    ### Without antennas ###
    ########################
    
    sim.reset_meep()
    p.bowtie_center = [-9999.9, -9999.9]

    sim = simulation.make_sim()

    collect_fields_with_output(
        sim,
        volumes={
        "xyplanar-empty": p.xy_plane,
        "xzplanar-empty": p.xz_plane,
        },
        delta_t=p.animations_step,
        until=p.sim_time,
        start_time=0
    )
    
    animate_field_from_h5(
        h5_filename="run-xyplanar-empty_ex.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="xyplanar-empty_ex.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )
    animate_field_from_h5(
        h5_filename="run-xzplanar-empty_ex.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="xzplanar-empty_ex.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )
    
    ####################
    ### Calculations ###
    ####################
    
    ### XY plane
    enhancement_divided_by_max(
        ["run-xyplanar-bowtie_ex.h5", "run-xyplanar-bowtie_ey.h5", "run-xyplanar-bowtie_ez.h5"],
        ["run-xyplanar-empty_ex.h5", "run-xyplanar-empty_ey.h5", "run-xyplanar-empty_ez.h5"],
        save_to="enhancement_xy_exyz.h5",
        out_dataset_name="enhancement"
    )

    animate_field_from_h5(
        h5_filename="enhancement_xy_exyz.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="enhancement_xy_exyz.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )
    
    ### XZ plane
    enhancement_divided_by_max(
        ["run-xzplanar-bowtie_ex.h5", "run-xzplanar-bowtie_ey.h5", "run-xzplanar-bowtie_ez.h5"],
        ["run-xzplanar-empty_ex.h5", "run-xzplanar-empty_ey.h5", "run-xzplanar-empty_ez.h5"],
        save_to="enhancement_xz_exyz.h5",
        out_dataset_name="enhancement"
    )

    animate_field_from_h5(
        h5_filename="enhancement_xz_exyz.h5",
        transpose_xy=True,
        cmap="RdBu",
        save_path="enhancement_xz_exyz.mp4",
        IMG_CLOSE=p.IMG_CLOSE
    )