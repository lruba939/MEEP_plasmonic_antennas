import meep as mp
import numpy as np
import os
from utils.logger import append_time_to_file

from visualization.plotter import *
# !!!!!!!!! ---> from main.src.simulation import * # CANT IMPORT DUE TO CIRCULAR DEPENDENCY

def collect_fields_with_output(
    sim,
    volumes,          # dict: {name: mp.Volume}
    delta_t,
    until,
    start_time=0.0,
    path=None,
    calc_E_fields=True,
    calc_H_fields=False,
    calc_Dpwr=False,
):
    """
    Collect selected field data for multiple volumes in ONE sim.run().

    Field components to be recorded are controlled by boolean flags.

    FIXED BEHAVIOR
    --------------
    * Data are written as HDF5 files using mp.to_appended(...)
    * All outputs follow the naming scheme:
          <prefix>_<field>.h5
      where <prefix> comes from the `volumes` dict.
    * Output directory is controlled at the Simulation level via
      sim.use_output_directory().

    PARAMETERS
    ----------
    sim : mp.Simulation
        Initialized Meep simulation object.

    volumes : dict[str, mp.Volume]
        Mapping of output prefixes to Meep volumes, e.g.:
            {
                "planar": planar_vol,
                "volume": volume_vol
            }

    delta_t : float
        Time interval between successive field outputs.

    until : float
        End time of the simulation.

    start_time : float, optional
        Time after which field recording starts.
        If 0.0, recording starts immediately.

    path : str or None, optional
        Directory where all HDF5 output files are written.
        If None, default Meep behavior is used.

    calc_E_fields : bool, optional
        If True, record electric field components:
            Ex, Ey, Ez

    calc_H_fields : bool, optional
        If True, record magnetic field components:
            Hx, Hy, Hz

    calc_Dpwr : bool, optional
        If True, record electric field energy density (Dpwr).

    NOTES
    -----
    * At least one of (calc_E_fields, calc_H_fields, calc_Dpwr)
      should be True, otherwise no data will be collected.
    * Default configuration records only E-field components.
    """

    # --------------------------------------------------
    # Disable default filename prefix (clean filenames)
    # --------------------------------------------------
    sim.filename_prefix = ""

    # --------------------------------------------------
    # Set output directory (Simulation-level, per docs)
    # --------------------------------------------------
    if path is not None:
        sim.use_output_directory(path)

    run_actions = []

    for prefix, volume in volumes.items():

        actions = []

        # -------------------------
        # Electric field components
        # -------------------------
        if calc_E_fields:
            actions.extend([
                mp.to_appended(f"{prefix}_ex", mp.at_every(delta_t, mp.output_efield_x)),
                mp.to_appended(f"{prefix}_ey", mp.at_every(delta_t, mp.output_efield_y)),
                mp.to_appended(f"{prefix}_ez", mp.at_every(delta_t, mp.output_efield_z)),
            ])

        # -------------------------
        # Magnetic field components
        # -------------------------
        if calc_H_fields:
            actions.extend([
                mp.to_appended(f"{prefix}_hx", mp.at_every(delta_t, mp.output_hfield_x)),
                mp.to_appended(f"{prefix}_hy", mp.at_every(delta_t, mp.output_hfield_y)),
                mp.to_appended(f"{prefix}_hz", mp.at_every(delta_t, mp.output_hfield_z)),
            ])

        # -------------------------
        # Energy density
        # -------------------------
        if calc_Dpwr:
            actions.append(
                mp.to_appended(f"{prefix}_dpwr", mp.at_every(delta_t, mp.output_dpwr))
            )

        # -------------------------
        # Skip volumes with no actions
        # -------------------------
        if not actions:
            continue

        # -------------------------
        # Volume action
        # -------------------------
        if start_time > 0:
            vol_action = mp.in_volume(
                volume,
                mp.after_time(start_time, *actions)
            )
        else:
            vol_action = mp.in_volume(volume, *actions)

        run_actions.append(vol_action)

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------
    if not run_actions:
        raise ValueError(
            "No field outputs selected. "
            "Set at least one of calc_E_fields, calc_H_fields, calc_Dpwr to True."
        )

    # --------------------------------------------------
    # Single sim.run()
    # --------------------------------------------------
    sim.run(*run_actions, until=until)
    return sim

def enhancement_divided_by_maxes_arr(
    h5_target,
    h5_reference,
    path=None,
    save_to=None,
    dataset_target=None,
    dataset_reference=None,
    z_index=None,
    xzeros=0,
    yzeros=None,
    eps=1e-12,
    out_dataset_name="enhancement",
):
    """
    Compute time-dependent enhancement normalized by the time-maximum
    of a reference field.

    MATHEMATICAL DEFINITION
    -----------------------
    For each spatial point (x, y) and time t:

        enhancement[x, y, t] = A[x, y, t] / max_t(B[x, y, t])

    where:
        A[x,y,t] = sum_i (A_i[x,y,t]^2)
        B[x,y,t] = sum_i (B_i[x,y,t]^2)

    The summation index i runs over the provided field components
    (e.g. Ex, Ey, Ez). Any subset of components is allowed, but
    h5_target and h5_reference MUST have identical structure.

    FIXED DATA ASSUMPTIONS
    ---------------------
    * All datasets are stored as:
          data[x, y, time]
    * No axis reordering or transposition is performed.
    * Any Z slicing is applied BEFORE computations.

    PARAMETERS
    ----------
    h5_target : str or list[str]
        HDF5 filename(s) of the target field components.
        Can be:
            - single string (one field component)
            - list/tuple of strings (multiple components)

    h5_reference : str or list[str]
        HDF5 filename(s) of the reference field components.
        MUST mirror the structure of h5_target exactly
        (same type and same list length).

    dataset_target : str or None, optional
        Dataset name inside the target HDF5 files.
        If None, the first dataset in each file is used.

    dataset_reference : str or None, optional
        Dataset name inside the reference HDF5 files.
        If None, the first dataset in each file is used.

    z_index : int or None, optional
        Index of the Z slice to extract if the dataset has shape:
            data[x, y, z, time]
        If such a Z axis exists and z_index is None, an error is raised.

    xzeros : int, optional
        Number of grid points to overwrite at left/right boundaries
        of the reference field (PML cleanup).

    yzeros : int or None, optional
        Number of grid points to overwrite at bottom/top boundaries.
        If None, defaults to xzeros.

    eps : float, optional
        Small regularization constant added to the denominator to
        avoid division by zero.

    path : str or None, optional
        Directory in which all input HDF5 files are searched AND
        where the output file is written.
        If None, filenames are interpreted as given.

    save_to : str or None, optional
        Name of the output HDF5 file (written inside `path`).
        If None, no file is saved.

    out_dataset_name : str, optional
        Dataset name under which the enhancement array is stored
        in the output HDF5 file.

    RETURNS
    -------
    enhancement : np.ndarray
        Array of shape (x, y, time) containing the enhancement field.

    B_max : np.ndarray
        Array of shape (x, y) containing max_t(B[x,y,t]).
    """
    # --------------------------------------------------
    # Helper: load one or many fields and sum squares
    # --------------------------------------------------
    def _load_and_sum_fields(h5_input, dataset_name, z_index, base_path):

        if isinstance(h5_input, (list, tuple)):
            fields = [
                _load_and_sum_fields(h5, dataset_name, z_index, base_path)
                for h5 in h5_input
            ]
            return np.sum(fields, axis=0)

        # --- Resolve file path ---
        h5_path = os.path.join(base_path, h5_input) if base_path else h5_input

        # --- Load single dataset ---
        with h5py.File(h5_path, "r") as f:
            if dataset_name is None:
                dataset_name = list(f.keys())[0]
            data = np.array(f[dataset_name])

        # --- Optional Z slicing ---
        if data.ndim == 4:
            if z_index is None:
                raise ValueError("Z axis detected but z_index not provided")
            data = data[:, :, z_index, :]

        if data.ndim != 3:
            raise ValueError(f"Expected data[x,y,time], got shape {data.shape}")

        return data**2

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------
    if isinstance(h5_target, (list, tuple)) != isinstance(h5_reference, (list, tuple)):
        raise ValueError("h5_target and h5_reference must have the same structure")

    if isinstance(h5_target, (list, tuple)):
        if len(h5_target) != len(h5_reference):
            raise ValueError("h5_target and h5_reference lists must have equal length")

    # --------------------------------------------------
    # Load and combine fields
    # --------------------------------------------------
    A = _load_and_sum_fields(h5_target, dataset_target, z_index, path)
    B = _load_and_sum_fields(h5_reference, dataset_reference, z_index, path)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    Nx, Ny, Nt = A.shape

    # --------------------------------------------------
    # Default yzeros
    # --------------------------------------------------
    if yzeros is None:
        yzeros = xzeros

    xzeros = max(0, min(xzeros, Nx // 2))
    yzeros = max(0, min(yzeros, Ny // 2))

    # --------------------------------------------------
    # Boundary cleanup (PML)
    # --------------------------------------------------
    if xzeros > 0 or yzeros > 0:
        B[:xzeros, :, :] = 1.0
        B[-xzeros:, :, :] = 1.0
        B[:, :yzeros, :] = 1.0
        B[:, -yzeros:, :] = 1.0

    # --------------------------------------------------
    # Time maximum of reference
    # --------------------------------------------------
    B_max = np.max(B, axis=2)  # (x, y)

    # --------------------------------------------------
    # Enhancement
    # --------------------------------------------------
    enhancement = A / (B_max[:, :, None] + eps)

    # --------------------------------------------------
    # Optional save (to the SAME path)
    # --------------------------------------------------
    if save_to is not None:
        save_path = os.path.join(path, save_to) if path else save_to
        with h5py.File(save_path, "w") as f:
            f.create_dataset(out_dataset_name, data=enhancement)
            f.create_dataset("reference_max", data=B_max)
    return enhancement, B_max

def analyze_roi_from_h5_physical(
    h5_filename,
    roi,
    load_h5data_path=None,
    dataset_name=None,

    # --- physical axis definition ---
    x_phys_range=None,   # (xmin, xmax)
    y_phys_range=None,   # (ymin, ymax)

    # --- PML / border crop ---
    xzeros=0,
    yzeros=None,
):
    """
    Analyze mean field value inside a physical ROI over time
    using the FULL simulation domain (no zoom).

    Parameters
    ----------
    h5_filename : str
        Name of HDF5 file with data[x,y,time].

    roi : dict
        ROI definition:
        {
            "type": "rectangle",
            "center": (x, y),
            "width": w,
            "height": h,
        }

    Returns
    -------
    frame_mean : ndarray (Nt, 2)
        [[frame_index, mean_value], ...]

    frame_max : ndarray (2,)
        [frame_index_of_max, max_mean_value]
    """
    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    if x_phys_range is None or y_phys_range is None:
        raise ValueError("You must provide x_phys_range and y_phys_range")

    if roi["type"] != "rectangle":
        raise NotImplementedError("Only rectangular ROI is supported")

    # --------------------------------------------------
    # Resolve HDF5 path
    # --------------------------------------------------
    h5_path = (
        os.path.join(load_h5data_path, h5_filename)
        if load_h5data_path is not None
        else h5_filename
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = list(f.keys())[0]
        data = np.array(f[dataset_name])

    if data.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {data.shape}")

    Nx0, Ny0, Nt = data.shape

    if yzeros is None:
        yzeros = xzeros

    # --------------------------------------------------
    # Crop PML
    # --------------------------------------------------
    xzeros = min(xzeros, Nx0 // 2)
    yzeros = min(yzeros, Ny0 // 2)

    data = data[
        xzeros : Nx0 - xzeros,
        yzeros : Ny0 - yzeros,
        :
    ]

    Nx, Ny, Nt = data.shape

    # --------------------------------------------------
    # Physical axes (FULL domain)
    # --------------------------------------------------
    x_min0, x_max0 = x_phys_range
    y_min0, y_max0 = y_phys_range

    x_phys = np.linspace(x_min0, x_max0, Nx)
    y_phys = np.linspace(y_min0, y_max0, Ny)

    # --------------------------------------------------
    # ROI mask (ONCE)
    # --------------------------------------------------
    roi_mask = roi_mask_from_rectangle(
        x_phys,
        y_phys,
        center=roi["center"],
        width=roi["width"],
        height=roi["height"],
    )

    if roi_mask.shape != (Nx, Ny):
        raise ValueError(
            f"ROI mask shape {roi_mask.shape} does not match data shape {(Nx, Ny)}"
        )

    # --------------------------------------------------
    # Mean value per frame
    # --------------------------------------------------
    mean_vals = np.zeros(Nt)

    for t in range(Nt):
        frame_raw = data[:, :, t]   # NEVER transpose
        mean_vals[t] = np.mean(frame_raw[roi_mask])

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------
    frames = np.arange(Nt)
    frame_mean = np.column_stack((frames, mean_vals))

    t_max = int(np.argmax(mean_vals))
    max_val = mean_vals[t_max]
    frame_max = np.array([t_max, max_val])
    return frame_mean, frame_max

def compute_fields(
    sim_antenna,
    sim_empty,
    volumes,
    config,
    mode="BOTH",
    calc_E=True,
    calc_H=False,
    calc_DPWR=False,
    fluxes=True,
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
    # Flux monitors
    # ============================================================
    if fluxes:
        fcen = config.frequency
        df = config.frequency_width
        nfreq = config.nfreq

        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, config.z_reflection),
            size=mp.Vector3(
                config.cell_size[0]-2*config.pad-2*config.pml,
                config.cell_size[1]-2*config.pad-2*config.pml,
                0)
        )

        tran_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, config.z_transmission),
            size=mp.Vector3(
                config.cell_size[0]-2*config.pad-2*config.pml,
                config.cell_size[1]-2*config.pad-2*config.pml,
                0)
        )

        refl_empty = sim_empty.add_flux(fcen, df, nfreq, refl_fr)
        tran_empty = sim_empty.add_flux(fcen, df, nfreq, tran_fr)
        refl = sim_antenna.add_flux(fcen, df, nfreq, refl_fr)
        tran = sim_antenna.add_flux(fcen, df, nfreq, tran_fr)
    
    # ============================================================
    # EMPTY STRUCTURE
    # ============================================================

    if mode in ["EMPTY", "BOTH"]:
        if mp.am_master():
            print("Running simulation WITHOUT antenna")
            append_time_to_file(config, prefix="Running simulation WITHOUT antenna: ")

        empty_planes = {f"{k}-empty": v for k, v in planes.items()}

        sim_empty = collect_fields_with_output(
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
        if fluxes:
            incident_flux = mp.get_fluxes(tran_empty)
            refl_data = sim_empty.get_flux_data(refl_empty)
            sim_antenna.load_minus_flux_data(refl, refl_data)

        sim_empty.reset_meep()

    # ============================================================
    # WITH ANTENNA
    # ============================================================

    if mode in ["WITH_ANTENNA", "BOTH"]:
        if mp.am_master():
            print("Running simulation WITH antenna")
            append_time_to_file(config, prefix="Running simulation WITH antenna: ")
        
        sim_antenna = collect_fields_with_output(
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
        if fluxes:
            refl_flux = mp.get_fluxes(refl)
            tran_flux = mp.get_fluxes(tran)
            flux_freqs = mp.get_flux_freqs(tran)

        sim_antenna.reset_meep()

    # ============================================================
    # TRAN AND REFL CALCULATION
    # ============================================================
    if fluxes and mode == "BOTH":
        compute_T_R_A(incident_flux, tran_flux, refl_flux, flux_freqs, config.path_to_save)

    # ============================================================
    # ENHANCEMENT CALCULATION
    # ============================================================

    if mode == "BOTH" and mp.am_master():
        if mp.am_master():
            print("Computing enhancement maps")
            append_time_to_file(config, prefix="Computing enhancement maps: ")
        
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

def animate_enhancement_fields(config, draw_params, field='E', animate=True):
    """
    Task 9:
    - Animate field enhancement for XY / XZ / YZ planes
    - Plot max-frame field maps with structure + ROI
    - Collect mean |E|^2 enchancement in gap vs time for each plane
    - Plot all mean curves on a single axes using multi_line_plotter_same_axes
    """
    if mp.am_master():
        valid_field = ["E", "H"]
        
        if field not in valid_field:
            raise ValueError(f"field must be one of {valid_field}")
        
        field=field.lower()

        # ============================================================
        # Plane configuration
        # ============================================================
        planes = {
            "XY": {
                "filename": f"enhancement_xyplanar_{field}2.h5",
                "save_anim": f"enh_xy_{field}2.mp4",
                "x_phys_range": [-config.cell_size[0] / 2 * 1e3, config.cell_size[0] / 2 * 1e3],
                "y_phys_range": [-config.cell_size[1] / 2 * 1e3, config.cell_size[1] / 2 * 1e3],
                "x_zoom": draw_params["XY"]["x_zoom"],
                "y_zoom": draw_params["XY"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Y [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XY"]["roi"]["center"],
                    "width": draw_params["XY"]["roi"]["width"],
                    "height": draw_params["XY"]["roi"]["height"],
                },
            },

            "XYTop": {
                "filename": f"enhancement_xyplanarTOP_{field}2.h5",
                "save_anim": f"enh_xy_TOP_{field}2.mp4",
                "x_phys_range": [-config.cell_size[0] / 2 * 1e3, config.cell_size[0] / 2 * 1e3],
                "y_phys_range": [-config.cell_size[1] / 2 * 1e3, config.cell_size[1] / 2 * 1e3],
                "x_zoom": draw_params["XY"]["x_zoom"],
                "y_zoom": draw_params["XY"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Y [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XY"]["roi"]["center"],
                    "width": draw_params["XY"]["roi"]["width"],
                    "height": draw_params["XY"]["roi"]["height"],
                },
            },

            "XZ": {
                "filename": f"enhancement_xzplanar_{field}2.h5",
                "save_anim": f"enh_xz_{field}2.mp4",
                "x_phys_range": [-config.cell_size[0] / 2 * 1e3, config.cell_size[0] / 2 * 1e3],
                "y_phys_range": [-config.cell_size[2] / 2 * 1e3, config.cell_size[2] / 2 * 1e3],
                "x_zoom": draw_params["XZ"]["x_zoom"],
                "y_zoom": draw_params["XZ"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Z [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XZ"]["roi"]["center"],
                    "width": draw_params["XZ"]["roi"]["width"],
                    "height": draw_params["XZ"]["roi"]["height"],
                },
            },

            "YZ": {
                "filename": f"enhancement_yzplanar_{field}2.h5",
                "save_anim": f"enh_yz_{field}2.mp4",
                "x_phys_range": [-config.cell_size[1] / 2 * 1e3, config.cell_size[1] / 2 * 1e3],
                "y_phys_range": [-config.cell_size[2] / 2 * 1e3, config.cell_size[2] / 2 * 1e3],
                "x_zoom": draw_params["YZ"]["x_zoom"],
                "y_zoom": draw_params["YZ"]["y_zoom"],
                "xlabel": "Y [nm]",
                "ylabel": "Z [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["YZ"]["roi"]["center"],
                    "width": draw_params["YZ"]["roi"]["width"],
                    "height": draw_params["YZ"]["roi"]["height"],
                },
            },
        }

        # ============================================================
        # Containers for line plots
        # ============================================================

        line_xdata = []
        line_ydata = []
        line_labels = []

        # ============================================================
        # Main loop over planes
        # ============================================================

        for plane, cfg in planes.items():
            print(f"Processing {plane} plane")

            if animate:
                # ---------- Animation ----------
                animate_field_from_h5_physical(
                    h5_filename=cfg["filename"],
                    load_h5data_path=config.path_to_save,
                    save_name=cfg["save_anim"],
                    save_path=config.animations_folder_path,
                    interval=50,
                    cmap="inferno",
                    transpose_xy=True,
                    IMG_CLOSE=config.IMG_CLOSE,
                    x_phys_range=cfg["x_phys_range"],
                    y_phys_range=cfg["y_phys_range"],
                    x_zoom=cfg["x_zoom"],
                    y_zoom=cfg["y_zoom"],
                    mask_left=0,
                    mask_right=0,
                    mask_bottom=5,
                    mask_top=5,
                    title=f"Field enhancement |E|²/|E0|² ({plane})",
                    xlabel=cfg["xlabel"],
                    ylabel=cfg["ylabel"],
                )

            # ---------- ROI analysis ----------
            frame_mean, frame_max = analyze_roi_from_h5_physical(
                h5_filename=cfg["filename"],
                load_h5data_path=config.path_to_save,
                roi=cfg["roi"],
                x_phys_range=cfg["x_phys_range"],
                y_phys_range=cfg["y_phys_range"],
            )

            print(f"Max mean enhancement in ROI for {plane}: {frame_max[1]:.2f} at frame {frame_max[0]}")

            # ---------- Max-frame plot ----------
            plot_field_frame_from_h5_physical(
                frame_index=int(frame_max[0]),
                h5_filename=cfg["filename"],
                load_h5data_path=config.path_to_save,
                cmap="inferno",
                transpose_xy=True,
                IMG_CLOSE=config.IMG_CLOSE,
                x_phys_range=cfg["x_phys_range"],
                y_phys_range=cfg["y_phys_range"],
                x_zoom=cfg["x_zoom"],
                y_zoom=cfg["y_zoom"],
                mask_left=0,
                mask_right=0,
                mask_bottom=5,
                mask_top=5,
                roi=cfg["roi"],
                title=f"Field enhancement |E|²/|E0|² ({plane})",
                xlabel=cfg["xlabel"],
                ylabel=cfg["ylabel"],
                save_path=config.animations_folder_path,
                save_name=f"MAP_{plane}.png",
            )

            # ---------- Collect data for joint line plot ----------
            line_xdata.append(frame_mean[:, 0])
            line_ydata.append(frame_mean[:, 1])
            line_labels.append(f"{plane}")

        # ============================================================
        # Joint line plot
        # ============================================================
        colors = cm2c(cm_inferno, 14)
        multi_line_plotter_same_axes(
            xdata_list=line_xdata,
            ydata_list=line_ydata,
            labels=line_labels,
            colors=[colors[0], colors[5], colors[7], colors[9]],
            linestyles=["-", "--", "-.", ":"],
            grid=True,
            xlabel="Time step",
            ylabel="|E|²/|E0|²",
            title="Mean |E|²/|E0|² in gap vs time",
            legend=True,
            save_path=config.animations_folder_path,
            save_name="MEAN_ENHANCEMENT_ALL_PLANES.png",
            IMG_CLOSE=config.IMG_CLOSE,
        )
    return 0

def compute_T_R_A(
    incident_flux,
    tran_flux,
    refl_flux,
    flux_freqs,
    save_path=None,
    save_name="spectra_TRA.txt"
):
    """
    Compute reflection (R), transmission (T), absorption (A)
    and wavelength from Meep flux monitors.

    Parameters
    ----------
    incident_flux : list or array
        Flux through transmission monitor in empty simulation.

    tran_flux : list or array
        Flux through transmission monitor with structure.

    refl_flux : list or array
        Flux through reflection monitor with structure.

    flux_freqs : list or array
        Frequencies returned by mp.get_flux_freqs().

    save_path : str or None
        Directory where results will be saved.

    save_name : str
        Name of the output file.

    Returns
    -------
    wavelength, R, T, A : numpy arrays
    """
    if mp.am_master():
        incident_flux = np.array(incident_flux)
        tran_flux = np.array(tran_flux)
        refl_flux = np.array(refl_flux)
        flux_freqs = np.array(flux_freqs)

        # -----------------------------------------
        # wavelength
        # -----------------------------------------
        wavelength = 1.0 / flux_freqs

        # -----------------------------------------
        # R T A
        # -----------------------------------------
        R = -refl_flux / incident_flux
        T = tran_flux / incident_flux
        A = 1.0 - R - T

        # -----------------------------------------
        # save
        # -----------------------------------------
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            data = np.column_stack((wavelength, R, T, A))

            header = "lambda  R  T  A"

            np.savetxt(
                os.path.join(save_path, save_name),
                data,
                header=header
            )
        
        # -----------------------------------------
        # plot
        # -----------------------------------------
        multi_line_plotter_same_axes(
            xdata_list=[wavelength, wavelength, wavelength],
            ydata_list=[R, T, A],
            labels=["R", "T", "A"],
            colors=["blue", "red", "green"],
            linestyles=["-", "--", "-."],
            xlabel="Wavelength [μm]",
            ylabel="Fraction",
            title="Reflection, Transmission, Absorption Spectra",
            legend=True,
            save_path=save_path,
            save_name="spectra_T_R_A.png",
        )
    return 0
