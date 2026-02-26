from utils.meep_utils import *
import meep as mp
from visualization.plotter import *
import numpy as np
import os

# TASK -------------------------------
def compute_fields(
    sim_antenna,
    sim_empty,
    volumes,
    config,
    mode="BOTH",
    calc_E=True,
    calc_H=False,
    calc_DPWR=False
):
    """
    Run field simulations and compute enhancement maps.

    Parameters
    ----------
    mode : str
        "WITH_ANTENNA", "EMPTY", or "BOTH"

    calc_E : bool
        Whether to calculate E-field enhancement.

    calc_H : bool
        Whether to calculate H-field enhancement.

    calc_DPWR : bool
        Whether to calculate power density fields.
    """

    valid_modes = ["WITH_ANTENNA", "EMPTY", "BOTH"]

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    # ============================================================
    # Plane configuration
    # ============================================================

    planes = {
        "xyplanar": volumes.volume["XY"],
        "xyplanarTOP": volumes.volume["XY_TOP"],
        "xzplanar": volumes.volume["XZ"],
        "yzplanar": volumes.volume["YZ"],
    }

    # ============================================================
    # WITH ANTENNA
    # ============================================================

    if mode in ["WITH_ANTENNA", "BOTH"]:
        print("Running simulation WITH antenna")

        collect_fields_with_output(
            sim_antenna,
            volumes=planes,
            delta_t=config.sim_time_step,
            until=config.sim_time,
            start_time=0,
            path=config.path_to_save,
            calc_E_fields=calc_E,
            calc_H_fields=calc_H,
            calc_Dpwr=calc_DPWR,
        )

        sim_antenna.reset_meep()

    # ============================================================
    # EMPTY STRUCTURE
    # ============================================================

    if mode in ["EMPTY", "BOTH"]:
        print("Running simulation WITHOUT antenna")

        empty_planes = {f"{k}-empty": v for k, v in planes.items()}

        collect_fields_with_output(
            sim_empty,
            volumes=empty_planes,
            delta_t=config.sim_time_step,
            until=config.sim_time,
            start_time=0,
            path=config.path_to_save,
            calc_E_fields=calc_E,
            calc_H_fields=calc_H,
            calc_Dpwr=calc_DPWR,
        )

        sim_empty.reset_meep()

    # ============================================================
    # ENHANCEMENT CALCULATION
    # ============================================================

    if mode == "BOTH":

        print("Computing enhancement maps")

        enhancement_planes = [
            "xyplanar",
            "xyplanarTOP",
            "xzplanar",
            "yzplanar",
        ]

        # ---------- E FIELD ENHANCEMENT ----------
        if calc_E:
            for base_name in enhancement_planes:

                enhancement_divided_by_maxes_arr(
                    [f"{base_name}_ex.h5", f"{base_name}_ey.h5", f"{base_name}_ez.h5"],
                    [f"{base_name}-empty_ex.h5", f"{base_name}-empty_ey.h5", f"{base_name}-empty_ez.h5"],
                    save_to=f"enhancement_{base_name}_e2.h5",
                    path=config.path_to_save,
                    out_dataset_name="enhancement",
                )

        # ---------- H FIELD ENHANCEMENT ----------
        if calc_H:
            for base_name in enhancement_planes:

                enhancement_divided_by_maxes_arr(
                    [f"{base_name}_hx.h5", f"{base_name}_hy.h5", f"{base_name}_hz.h5"],
                    [f"{base_name}-empty_hx.h5", f"{base_name}-empty_hy.h5", f"{base_name}-empty_hz.h5"],
                    save_to=f"enhancement_{base_name}_h2.h5",
                    path=config.path_to_save,
                    out_dataset_name="enhancement",
                )
    return 0

# TASK -------------------------------
def animate_raw_fields(
    config,
    mode="BOTH",
    animate_E=True,
    animate_H=False,
    animate_DPWR=False,
    component="X",
):
    """
    Generate animations for fields map.

    Parameters
    ----------
    mode : str
        "WITH_ANTENNA", "EMPTY", or "BOTH"

    animate_E : bool
        Animate E-field component.

    animate_H : bool
        Animate H-field component.

    animate_DPWR : bool
        Animate power density field.

    component : str
        Field component: "X", "Y", or "Z".
    """

    valid_modes = ["WITH_ANTENNA", "EMPTY", "BOTH"]
    valid_components = ["X", "Y", "Z"]

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    if component not in valid_components:
        raise ValueError(f"component must be one of {valid_components}")

    comp = component.lower()

    planes = [
        "xyplanar",
        "xyplanarTOP",
        "xzplanar",
        "yzplanar",
    ]

    # ============================================================
    # FUNCTION TO ANIMATE SINGLE FILE
    # ============================================================

    def animate_file(filename):
        animate_field_from_h5(
            h5_filename=filename,
            save_name=filename.replace(".h5", ".mp4"),
            load_h5data_path=config.path_to_save,
            save_path=config.animations_folder_path,
            transpose_xy=True,
            cmap="RdBu",
            IMG_CLOSE=config.IMG_CLOSE,
        )

    # ============================================================
    # WITH ANTENNA
    # ============================================================

    if mode in ["WITH_ANTENNA", "BOTH"]:
        print("Animating WITH antenna")

        for plane in planes:

            if animate_E:
                animate_file(f"{plane}_e{comp}.h5")

            if animate_H:
                animate_file(f"{plane}_h{comp}.h5")

            if animate_DPWR:
                animate_file(f"{plane}_dpwr.h5")

    # ============================================================
    # EMPTY
    # ============================================================

    if mode in ["EMPTY", "BOTH"]:
        print("Animating EMPTY structure")

        for plane in planes:

            if animate_E:
                animate_file(f"{plane}-empty_e{comp}.h5")

            if animate_H:
                animate_file(f"{plane}-empty_h{comp}.h5")

            if animate_DPWR:
                animate_file(f"{plane}-empty_dpwr.h5")
    return 0

# # TASK -------------------------------
# def animate_enhancement_fields(field='E'):
#     """
#     Task 9:
#     - Animate field enhancement for XY / XZ / YZ planes
#     - Plot max-frame field maps with structure + ROI
#     - Collect mean |E|^2 enchancement in gap vs time for each plane
#     - Plot all mean curves on a single axes using multi_line_plotter_same_axes
#     """
    
#     valid_field = ["E", "H"]
    
#     if field not in valid_field:
#         raise ValueError(f"field must be one of {valid_field}")
    
#     field=field.lower()

#     # ============================================================
#     # Plane configuration
#     # ============================================================

#     planes = {
#         "XY": {
#             "filename": f"enhancement_xyplanar_{field}2.h5",
#             "save_anim": f"enh_xy_{field}2_with_struct.mp4",
#             "x_phys_range": [-p.xyz_cell[0] / 2 * 1e3, p.xyz_cell[0] / 2 * 1e3],
#             "y_phys_range": [-p.xyz_cell[1] / 2 * 1e3, p.xyz_cell[1] / 2 * 1e3],
#             "x_zoom": 0.15,
#             "y_zoom": 1.0,
#             "xlabel": "X [nm]",
#             "ylabel": "Y [nm]",
#             "structure": {
#                 "type": "splitbar",
#                 "bars": [
#                     {
#                         "center": ((p.x_width / 2 + p.gap_size / 2) * 1e3, 0),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": p.First_layer[1] * 1e3,
#                     },
#                     {
#                         "center": (-(p.x_width / 2 + p.gap_size / 2) * 1e3, 0),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": p.First_layer[1] * 1e3,
#                     },
#                 ],
#             },
#             "roi": {
#                 "type": "rectangle",
#                 "center": (0, 0),
#                 "width": p.gap_size * 1e3,
#                 "height": p.First_layer[1] * 1e3,
#             },
#         },

#         "XYTop": {
#             "filename": f"enhancement_xyplanarTOP_{field}2.h5",
#             "save_anim": f"enh_xy_TOP_{field}2_with_struct.mp4",
#             "x_phys_range": [-p.xyz_cell[0] / 2 * 1e3, p.xyz_cell[0] / 2 * 1e3],
#             "y_phys_range": [-p.xyz_cell[1] / 2 * 1e3, p.xyz_cell[1] / 2 * 1e3],
#             "x_zoom": 0.15,
#             "y_zoom": 1.0,
#             "xlabel": "X [nm]",
#             "ylabel": "Y [nm]",
#             "structure": {
#                 "type": "splitbar",
#                 "bars": [
#                     {
#                         "center": ((p.x_width / 2 + p.gap_size / 2) * 1e3, 0),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": p.First_layer[1] * 1e3,
#                     },
#                     {
#                         "center": (-(p.x_width / 2 + p.gap_size / 2) * 1e3, 0),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": p.First_layer[1] * 1e3,
#                     },
#                 ],
#             },
#             "roi": {
#                 "type": "rectangle",
#                 "center": (0, 0),
#                 "width": p.gap_size * 1e3,
#                 "height": p.First_layer[1] * 1e3,
#             },
#         },

#         "XZ": {
#             "filename": f"enhancement_xzplanar_{field}2.h5",
#             "save_anim": f"enh_xz_{field}2_with_struct.mp4",
#             "x_phys_range": [-p.xyz_cell[0] / 2 * 1e3, p.xyz_cell[0] / 2 * 1e3],
#             "y_phys_range": [-p.xyz_cell[2] / 2 * 1e3, p.xyz_cell[2] / 2 * 1e3],
#             "x_zoom": 0.15,
#             "y_zoom": 0.5,
#             "xlabel": "X [nm]",
#             "ylabel": "Z [nm]",
#             "structure": {
#                 "type": "splitbar",
#                 "bars": [
#                     {
#                         "center": ((p.x_width / 2 + p.gap_size / 2) * 1e3, -2.5),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#                     },
#                     {
#                         "center": (-(p.x_width / 2 + p.gap_size / 2) * 1e3, -2.5),
#                         "width": p.First_layer[0] * 1e3,
#                         "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#                     },
#                 ],
#             },
#             "roi": {
#                 "type": "rectangle",
#                 "center": (0, -2.5),
#                 "width": p.gap_size * 1e3,
#                 "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#             },
#         },

#         "YZ": {
#             "filename": f"enhancement_yzplanar_{field}2.h5",
#             "save_anim": f"enh_yz_{field}2_with_struct.mp4",
#             "x_phys_range": [-p.xyz_cell[1] / 2 * 1e3, p.xyz_cell[1] / 2 * 1e3],
#             "y_phys_range": [-p.xyz_cell[2] / 2 * 1e3, p.xyz_cell[2] / 2 * 1e3],
#             "x_zoom": 1.0,
#             "y_zoom": 0.5,
#             "xlabel": "Y [nm]",
#             "ylabel": "Z [nm]",
#             "structure": {
#                 "type": "splitbar",
#                 "bars": [
#                     {
#                         "center": (0, -2.5),
#                         "width": p.First_layer[1] * 1e3,
#                         "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#                     },
#                     {
#                         "center": (0, -2.5),
#                         "width": p.First_layer[1] * 1e3,
#                         "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#                     },
#                 ],
#             },
#             "roi": {
#                 "type": "rectangle",
#                 "center": (0, -2.5),
#                 "width": p.First_layer[1] * 1e3,
#                 "height": (p.First_layer[2]+p.Second_layer[2]) * 1e3,
#             },
#         },
#     }

#     # ============================================================
#     # Containers for line plots
#     # ============================================================

#     line_xdata = []
#     line_ydata = []
#     line_labels = []

#     # ============================================================
#     # Main loop over planes
#     # ============================================================

#     for plane, cfg in planes.items():
#         print(f"Processing {plane} plane")

#         # ---------- Animation ----------
#         animate_field_from_h5_physical(
#             h5_filename=cfg["filename"],
#             load_h5data_path=p.path_to_save,
#             save_name=cfg["save_anim"],
#             save_path=p.animations_folder_path,
#             interval=50,
#             cmap="inferno",
#             transpose_xy=True,
#             IMG_CLOSE=p.IMG_CLOSE,
#             x_phys_range=cfg["x_phys_range"],
#             y_phys_range=cfg["y_phys_range"],
#             x_zoom=cfg["x_zoom"],
#             y_zoom=cfg["y_zoom"],
#             mask_left=0,
#             mask_right=0,
#             mask_bottom=5,
#             mask_top=5,
#             structure=cfg["structure"],
#             title=f"Field enhancement |E|² ({plane})",
#             xlabel=cfg["xlabel"],
#             ylabel=cfg["ylabel"],
#         )

#         # ---------- ROI analysis ----------
#         frame_mean, frame_max = analyze_roi_from_h5_physical(
#             h5_filename=cfg["filename"],
#             load_h5data_path=p.path_to_save,
#             roi=cfg["roi"],
#             x_phys_range=cfg["x_phys_range"],
#             y_phys_range=cfg["y_phys_range"],
#         )

#         # ---------- Max-frame plot ----------
#         plot_field_frame_from_h5_physical(
#             frame_index=int(frame_max[0]),
#             h5_filename=cfg["filename"],
#             load_h5data_path=p.path_to_save,
#             cmap="inferno",
#             transpose_xy=True,
#             IMG_CLOSE=p.IMG_CLOSE,
#             x_phys_range=cfg["x_phys_range"],
#             y_phys_range=cfg["y_phys_range"],
#             x_zoom=cfg["x_zoom"],
#             y_zoom=cfg["y_zoom"],
#             mask_left=0,
#             mask_right=0,
#             mask_bottom=5,
#             mask_top=5,
#             roi=cfg["roi"],
#             structure=cfg["structure"],
#             title=f"Field enhancement |E|² ({plane})",
#             xlabel=cfg["xlabel"],
#             ylabel=cfg["ylabel"],
#             save_path=p.animations_folder_path,
#             save_name=f"MAP_{plane}.png",
#         )

#         # ---------- Collect data for joint line plot ----------
#         line_xdata.append(frame_mean[:, 0])
#         line_ydata.append(frame_mean[:, 1])
#         line_labels.append(f"{plane}")

#     # ============================================================
#     # Joint line plot
#     # ============================================================
#     colors = cm2c(cm_inferno, 14)
#     multi_line_plotter_same_axes(
#         xdata_list=line_xdata,
#         ydata_list=line_ydata,
#         labels=line_labels,
#         colors=[colors[0], colors[5], colors[7], colors[9]],
#         linestyles=["-", "--", "-.", ":"],
#         grid=True,
#         xlabel="Time step",
#         ylabel="|E|² enchancement",
#         title="Mean |E|² enchancement in gap vs time",
#         legend=True,
#         save_path=p.animations_folder_path,
#         save_name="MEAN_ENHANCEMENT_ALL_PLANES.png",
#     )
#     return 0