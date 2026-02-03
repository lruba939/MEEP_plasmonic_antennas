import meep as mp
import numpy as np
import os

from visualization.plotter import *
# !!!!!!!!! ---> from main.src.simulation import * # CANT IMPORT DUE TO CIRCULAR DEPENDENCY

def show_data_img(datas_arr, abs_bool, norm_bool, cmap_arr, alphas, name_to_save=None, IMG_CLOSE=False, Title=None, disable_ticks=True):
    """
    Displays a series of images from a given array of data.

    Parameters:
    datas_arr (list of np.ndarray): A list of 2D arrays containing the data to be visualized.
    norm_bool (list of bool): A list of boolean values indicating whether to normalize each corresponding data array.
    cmap_arr (list of str): A list of colormap names to be used for each corresponding data array.
    alphas (list of float): A list of alpha values for transparency for each corresponding data array.

    The function iterates through the provided data arrays, normalizes them if specified, 
    and displays each image using matplotlib's imshow function with the specified colormap 
    and transparency settings. The x and y axis ticks are turned off for a cleaner visualization.
    """
    for idx, data in enumerate(datas_arr):
        if abs_bool[idx]:
            data = np.abs(data) # complex -> real
        if norm_bool[idx]:
            max_data = np.max(data)
            data = data / max_data # complex -> real
        plt.imshow(data.transpose(), interpolation="spline36", cmap=cmap_arr[idx], alpha=alphas[idx])
        plt.colorbar(shrink=0.6)  # Show color scale
    if disable_ticks:
        plt.xticks([])  # Turn off x-axis numbers
        plt.yticks([])  # Turn off y-axis numbers
    if Title is not None:
        plt.title(Title)
    if name_to_save is not None:
        plt.savefig(f"{name_to_save}.png", dpi=300, bbox_inches="tight", format="png")
    if IMG_CLOSE:
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    else:
        plt.show()
    
def make_animation(singleton_params, sim, animation_name):
    """
    Generates an animation of the simulation fields and saves it as an MP4 file.

    Parameters:
        singleton_params (object): An object containing parameters for the animation, 
                                    including component, animations_step, animations_until, 
                                    animations_folder_path, and animations_fps.
        sim (object): The simulation object that is being animated.
        animation_name (str): The name of the animation file (without extension).

    Returns:
        None: This function does not return a value. It saves the animation to the specified path.
    """
    animation_name = animation_name + ".mp4"
    sim.reset_meep()
    animate = mp.Animate2D(sim, fields=singleton_params.component, normalize = True)
    sim.run(mp.at_every(singleton_params.animations_step, animate), until=singleton_params.animations_until)
    animate.to_mp4(filename = os.path.join(singleton_params.animations_folder_path, animation_name), fps = singleton_params.animations_fps)
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")

def collect_e_line(singleton_params, sim, delta_t, width=1, plot_3d=False, name=None):
    """
    Collect E component along center line (0, y_0:y_end, 0) at intervals of delta_t.
    The returned ey_line is the mean across a horizontal "width":
      - width=1 -> only the center column
      - width=2 -> center column plus columns at +/-1 (3 columns total)
      - width=N -> columns at offsets - (N-1) .. + (N-1)
    Args:
        delta_t: Time interval between data collections
        width: integer >=1 controlling how many columns (orders) to include
        plot_3d: Whether to plot the collected data in 3D
    Returns:
        collected_data: List (time) of 1D arrays (y) with mean Ey
        time_steps: List of time values
        y_coords: Array of y coordinates along the line
    """
    collected_data = []
    time_steps = []
    y_coords = None

    def collect_data(sim):
        nonlocal y_coords
        E_data = sim.get_array(center=mp.Vector3(), size=singleton_params.xyz_cell, component=singleton_params.component)
        # E_data shape: (nx, ny)  (may be 2D)
        nx = E_data.shape[0]
        ny = E_data.shape[1]
        center_i = ny // 2

        # compute offsets: for width=1 -> [0], width=2 -> [-1,0,1], etc.
        max_order = max(0, width - 1)
        offsets = [o for o in range(-max_order, max_order + 1)
                   if 0 <= center_i + o < ny]

        # select the columns and average across them (axis=0 -> per-y mean)
        cols = E_data[[center_i + o for o in offsets], :]
        if cols.ndim == 1:
            e_line = cols.copy()
        else:
            e_line = np.mean(cols, axis=0)

        collected_data.append(e_line)
        time_steps.append(sim.meep_time())

        if y_coords is None:
            # use actual simulation cell y-extent
            y_extent = singleton_params.xyz_cell[1]
            y_coords = np.linspace(-y_extent/2, y_extent/2, e_line.shape[0])
    
    sim.reset_meep()
    sim.run(mp.at_every(delta_t, collect_data), until=singleton_params.animations_until)

    if len(collected_data) == 0:
        return collected_data, time_steps, None

    if plot_3d: 
        save_name = os.path.join(singleton_params.path_to_save, f"3Dplot_profile_{name}.png")
        plot_e_3d(collected_data, y_coords, time_steps, name=save_name, IMG_CLOSE=singleton_params.IMG_CLOSE)

    return collected_data, time_steps, y_coords

def plot_e_3d(collected_data, x_coords, time_steps, name=None, IMG_CLOSE=False):
    """
    Plot E component in 3D: x axis, time axis, z axis (E magnitude)
    
    Args:
        collected_data: List of E field arrays at each time step
        x_coords: Array of x coordinates
        time_steps: List of time values
    """   
    # Create meshgrid for 3D plot
    X, T = np.meshgrid(x_coords, time_steps)
    Z = np.abs(np.array(collected_data))
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, T, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    
    ax.set_xlabel('y coordinate')
    ax.set_ylabel('time')
    ax.set_zlabel('|E|')
    ax.set_title('E Component vs Time')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Adjust viewpoint: elev controls vertical angle, azim controls horizontal angle
    ax.view_init(elev=20, azim=45)
    plt.savefig(name, dpi=300, bbox_inches="tight", format="png")
    if IMG_CLOSE:
        plt.show(block=False)
        plt.pause(2)
        plt.close("all")
    else:
        plt.show()
    
def collect_max_field(singleton_params, sim, delta_t, skip_fraction=0.5, optional_name="NAME"):
    """
    Collects the maximum value of the component field at each spatial point 
    across the simulation duration, skipping the first skip_fraction of time.

    Parameters:
        singleton_params (object): Singleton with component, xyz_cell, animations_until
        sim (object): MEEP simulation object
        skip_fraction (float): Fraction of simulation time to skip (default 0.25 = 25%)
        delta_t (float): Time interval between data collections

    Returns:
        E_max (np.ndarray): 2D array with maximum field magnitude at each point
    """
    collected_data = []
    skip_time = singleton_params.animations_until * skip_fraction

    def collect_data(sim):
        current_time = sim.meep_time()
        
        if current_time >= skip_time:
            E_data = sim.get_array(center=mp.Vector3(), 
                                   size=singleton_params.xyz_cell, 
                                   component=singleton_params.component)
            collected_data.append(np.abs(E_data))

    sim.reset_meep()
    sim.run(mp.at_every(delta_t, collect_data), until=singleton_params.animations_until)

    if len(collected_data) == 0:
        print("Warning: No data collected after skipping initial time!")
        return None

    # Initialize E_maxes as a zero array with the same shape as collected_data[0]
    E_maxes = np.zeros_like(collected_data[0], dtype=float)

    np.savez(
        os.path.join(singleton_params.path_to_save, "anim_collected_data_f{optional_name}.npz"),
        current_data = collected_data
        )
    
    # For each time step, update E_maxes if the current value is greater
    for i in range(len(collected_data)):
        current_data = collected_data[i]
        # print(f"CURRENT DATA: {current_data[100,100]}")
        # print(f"CURRENT MAXES: {E_maxes[100,100]}")
        E_maxes = np.maximum(E_maxes, current_data)
    
    # Zero the frame of width `frame_width` from each edge
    frame_width = 20
    if frame_width > 0:
        # Top and bottom edges
        E_maxes[:frame_width, :] = 0
        E_maxes[-frame_width:, :] = 0
        # Left and right edges
        E_maxes[:, :frame_width] = 0
        E_maxes[:, -frame_width:] = 0

    return E_maxes

def collect_data_in_time(singleton_params, sim, delta_t, clear_pml=False, **field_arrays):
    """
    Collect field data from simulation at intervals of delta_t.
    Only collects data for the field arrays passed as keyword arguments.
    
    Args:
        singleton_params: Simulation parameters object
        sim: MEEP simulation object
        delta_t: Time interval between data collections
        clear_pml: If True, zeroes out PML regions in collected data
        **field_arrays: Keyword arguments with field names as keys and empty lists as values.
                       Example: collect_data_in_time(params, sim, 0.1, Ex=[], Ey=[], Ez=[])
                       Supported fields: Ex, Ey, Ez, Hx, Hy, Hz, Dpwr, Hpwr, E2, H2
                       Special fields:
                       - E2: Calculates |Ex|^2 + |Ey|^2 + |Ez|^2
                       - H2: Calculates |Hx|^2 + |Hy|^2 + |Hz|^2
    
    Returns:
        Dictionary with collected data for each field and time_steps list
        Example: {'Ex': [data_t0, data_t1, ...], 'Ey': [...], 'time_steps': [t0, t1, ...]}
    """
    # Validate field names and create mapping to MEEP getter functions
    field_getters = {
        'Ex': lambda s: s.get_efield_x(),
        'Ey': lambda s: s.get_efield_y(),
        'Ez': lambda s: s.get_efield_z(),
        'Hx': lambda s: s.get_hfield_x(),
        'Hy': lambda s: s.get_hfield_y(),
        'Hz': lambda s: s.get_hfield_z(),
        'Dpwr': lambda s: s.get_dpwr(),
        'Hpwr': lambda s: s.get_hpwr(),
    }
    
    # Check if user requested E2 or H2 (special fields)
    needs_E_components = 'E2' in field_arrays
    needs_H_components = 'H2' in field_arrays
    
    # Add required components to collect
    if needs_E_components:
        if 'Ex' not in field_arrays:
            field_arrays['Ex'] = []
        if 'Ey' not in field_arrays:
            field_arrays['Ey'] = []
        if 'Ez' not in field_arrays:
            field_arrays['Ez'] = []
    
    if needs_H_components:
        if 'Hx' not in field_arrays:
            field_arrays['Hx'] = []
        if 'Hy' not in field_arrays:
            field_arrays['Hy'] = []
        if 'Hz' not in field_arrays:
            field_arrays['Hz'] = []
    
    # Validate requested fields
    for field_name in field_arrays.keys():
        if field_name not in field_getters and field_name not in ['E2', 'H2']:
            raise ValueError(f"Unknown field: {field_name}. Supported fields: {list(field_getters.keys()) + ['E2', 'H2']}")
    
    # Initialize collected data dictionaries
    collected_data = {field_name: [] for field_name in field_arrays.keys() if field_name not in ['E2', 'H2']}
    if needs_E_components:
        collected_data['E2'] = []
    if needs_H_components:
        collected_data['H2'] = []
    
    time_steps = []
    
    def process_array(arr):
        """Process array by clearing PML regions if requested"""
        if clear_pml:
            pmlx = singleton_params.pml / (singleton_params.xyz_cell[0] / 2)
            pmly = singleton_params.pml / (singleton_params.xyz_cell[1] / 2)
            
            cx = int(np.ceil(pmlx * arr.shape[1] / 2))
            cy = int(np.ceil(pmly * arr.shape[0] / 2))
            
            # Zero out PML regions
            arr[:cy, :] = 0
            arr[-cy:, :] = 0
            arr[:, :cx] = 0
            arr[:, -cx:] = 0
        
        return arr
    
    def collect_data_callback(sim):
        """Callback function to collect data at each time step"""
        for field_name in field_arrays.keys():
            if field_name not in ['E2', 'H2']:
                data = field_getters[field_name](sim)
                data = process_array(data.copy())
                collected_data[field_name].append(data)
        
        # Calculate E2 if requested
        if needs_E_components:
            Ex = field_getters['Ex'](sim)
            Ey = field_getters['Ey'](sim)
            Ez = field_getters['Ez'](sim)
            
            E2 = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
            E2 = process_array(E2)
            collected_data['E2'].append(E2)
        
        # Calculate H2 if requested
        if needs_H_components:
            Hx = field_getters['Hx'](sim)
            Hy = field_getters['Hy'](sim)
            Hz = field_getters['Hz'](sim)
            
            H2 = np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2
            H2 = process_array(H2)
            collected_data['H2'].append(H2)
        
        time_steps.append(sim.meep_time())
    
    # Run simulation with data collection
    sim.reset_meep()
    sim.run(mp.at_every(delta_t, collect_data_callback), until=singleton_params.animations_until)
    
    # Add time_steps to output dictionary
    collected_data['time_steps'] = time_steps
    
    return collected_data

def calculate_field_ratio(data_numerator, data_denominator, epsilon=1e-10):
    """
    Calculate the ratio between two field arrays element-wise.
    
    Args:
        data_numerator: List of 2D arrays (numerator) from collect_data_in_time()
        data_denominator: List of 2D arrays (denominator) from collect_data_in_time()
        epsilon: Small value to avoid division by zero (default: 1e-10)
    
    Returns:
        ratio_data: List of 2D arrays containing the ratios for each time step
                   Each element: numerator[i] / (denominator[i] + epsilon)
    
    Example:
        collected = collect_data_in_time(params, sim, 0.1, Ex=[], Ey=[])
        ex_ey_ratio = calculate_field_ratio(collected['Ex'], collected['Ey'])
        # Now ex_ey_ratio[t] contains Ex[t] / (Ey[t] + 1e-10) for each time step t
    """
    if len(data_numerator) != len(data_denominator):
        raise ValueError(f"Arrays have different lengths: numerator={len(data_numerator)}, denominator={len(data_denominator)}")
    
    ratio_data = []
    
    for num_data, denom_data in zip(data_numerator, data_denominator):
        # Calculate ratio with epsilon to avoid division by zero
        ratio = num_data / (np.abs(denom_data) + epsilon)
        ratio_data.append(ratio)
    
    return ratio_data

def calculate_field_enhancement(data_field, data_ref, singleton_params, region_fraction=0.25):
    """
    Calculate field enhancement as the ratio of field to maximum reference field
    in a specified region around the center (0, 0, 0).
    
    Args:
        data_field: List of 2D arrays (field to enhance) from collect_data_in_time()
        data_ref: List of 2D arrays (reference field) from collect_data_in_time()
        singleton_params: Simulation parameters object with xyz_cell, resolution
        region_fraction: Fraction of cell size to define region of interest around center
                        (default: 0.25 = 25% of cell size)
    
    Returns:
        enhancement_data: List of 2D arrays containing enhancement for each time step
                         Each element: field[i] / maximum_ref_in_region
    
    Example:
        collected = collect_data_in_time(params, sim, 0.1, Ex=[], Ex_ref=[])
        enhancement = calculate_field_enhancement(
            collected['Ex'], 
            collected['Ex_ref'],
            params,
            region_fraction=0.25  # 25% of cell size around (0,0,0)
        )
        # enhancement[t] = Ex[t] / max(Ex_ref[t] in central 25% region)
    """
    if len(data_field) != len(data_ref):
        raise ValueError(f"Arrays have different lengths: field={len(data_field)}, ref={len(data_ref)}")
    
    # Calculate region dimensions based on cell size and fraction
    region_x = singleton_params.xyz_cell[0] * region_fraction / 2  # half-width
    region_y = singleton_params.xyz_cell[1] * region_fraction / 2  # half-height
    
    # Get array shape to map to pixel coordinates
    # Assuming data shape is (nx, ny)
    nx = data_ref[0].shape[0]
    ny = data_ref[0].shape[1]
    
    cell_x = singleton_params.xyz_cell[0]
    cell_y = singleton_params.xyz_cell[1]
    
    # Calculate pixel indices for the region around center (0, 0, 0)
    # Center is at (nx/2, ny/2) in pixel coordinates
    center_i = nx // 2
    center_j = ny // 2
    
    # Convert physical coordinates to pixel offsets
    pixels_per_unit_x = nx / cell_x
    pixels_per_unit_y = ny / cell_y
    
    region_pixels_x = int(np.ceil(region_x * pixels_per_unit_x))
    region_pixels_y = int(np.ceil(region_y * pixels_per_unit_y))
    
    # Define region boundaries
    i_min = max(0, center_i - region_pixels_y)
    i_max = min(nx, center_i + region_pixels_y)
    j_min = max(0, center_j - region_pixels_x)
    j_max = min(ny, center_j + region_pixels_x)
    
    # Find maximum reference value in the region across all time steps
    max_ref_in_region = 0.0
    for ref_data in data_ref:
        region_data = np.abs(ref_data[i_min:i_max, j_min:j_max])
        max_val = np.max(region_data)
        if max_val > max_ref_in_region:
            max_ref_in_region = max_val
    
    # Prevent division by zero
    if max_ref_in_region == 0:
        max_ref_in_region = 1e-10
    print("\n\n!!!!!!!!!!!!! Max field for empty cell: ", max_ref_in_region, "\n\n")
    # Calculate enhancement for each time step
    enhancement_data = []
    for field_data in data_field:
        enhancement = np.abs(field_data) / max_ref_in_region
        enhancement_data.append(enhancement)
    
    return enhancement_data

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

def collect_time_max_from_h5(
    h5_file,
    dataset_name=None,
    z_index=None,
    skip_fraction=0.5,
    frame_width=0,
    take_abs=True,
):
    """
    Collects time-maximum field map from HDF5 data.

    FIXED ASSUMPTION:
    -----------------
    Data layout is ALWAYS:
        data[x, y, time]

    Returns
    -------
    max_map : np.ndarray
        2D array (x, y)
    """

    # --- Load data ---
    with h5py.File(h5_file, "r") as f:
        if dataset_name is None:
            dataset_name = list(f.keys())[0]
        data = np.array(f[dataset_name])

    # --- Optional Z axis ---
    # Allowed shapes:
    #   (x, y, t)
    #   (x, y, z, t)
    if data.ndim == 4:
        if z_index is None:
            raise ValueError("Z axis detected but z_index not provided")
        data = data[:, :, z_index, :]

    if data.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got shape {data.shape}")

    Nx, Ny, Nt = data.shape

    # --- Skip transient ---
    start_idx = int(skip_fraction * Nt)
    if start_idx >= Nt:
        raise ValueError("skip_fraction too large")

    # --- Prepare first frame ---
    frame0 = data[:, :, start_idx]
    if take_abs:
        frame0 = np.abs(frame0)

    max_map = frame0.astype(float, copy=True)

    # --- Incremental max over time ---
    for t in range(start_idx + 1, Nt):
        frame = data[:, :, t]
        if take_abs:
            frame = np.abs(frame)
        np.maximum(max_map, frame, out=max_map)

    # --- Zero frame (PML-like) ---
    if frame_width > 0:
        fw = min(frame_width, Nx // 2, Ny // 2)
        max_map[:fw, :] = 0
        max_map[-fw:, :] = 0
        max_map[:, :fw] = 0
        max_map[:, -fw:] = 0

    return max_map

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
