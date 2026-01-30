import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from matplotlib import animation
import numpy as np
import os
import h5py
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Global settings for plotting

## Font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14

## Lines
rcParams['lines.solid_joinstyle'] = 'miter'  # other options: 'round' or 'bevel'
rcParams['lines.antialiased'] = True  # turning on/off of antialiasing for sharper edges
rcParams['lines.linewidth'] = 1.25

## Legend
rcParams['legend.loc'] = 'upper left'
rcParams['legend.frameon'] = False

## Ticks
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True

rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True

## Resolution
rcParams['figure.dpi'] = 150

## Global color
rcParams['image.cmap'] = "viridis"

## Colors
### cmaps
cm_inferno = get_cmap("inferno")
cm_viridis = get_cmap("viridis")
cm_seismic = get_cmap("seismic")
cm_jet = get_cmap("jet")
cm_tab10 = get_cmap("tab10")
cm_rdbu = get_cmap("RdBu")
### Palettes from color-hex.com/
c_google = ['#008744', '#0057e7', '#d62d20', '#ffa700'] # G, B, R, Y # https://www.color-hex.com/color-palette/1872
c_twilight = ['#363b74', '#673888', '#ef4f91', '#c79dd7', '#4d1b7b'] # https://www.color-hex.com/color-palette/809


# Get array of colors from cmap
def cm2c(cmap, c_numb, step=6):
    """
    Convert a colormap to a list of discrete colors.
    This function samples colors from a given colormap at regular intervals
    and returns them as a list of color values.
    Args:
        cmap: A matplotlib colormap object to sample colors from.
        c_numb (int): The number of colors to generate from the colormap.
        step (int, optional): The step size for sampling the colormap. 
            If c_numb is greater than step, step is adjusted to c_numb.
            Defaults to 6.
    Returns:
        list: A list of color tuples sampled from the colormap at regular intervals.
    """
    if c_numb > step:
        step = c_numb
    
    colors_arr = []
    for i in range(c_numb):
        colors_arr.append(cmap(i / step))
    
    return colors_arr

def map_plotter(data, ax=None, cm=cm_inferno, xlabel=r"x [nm]", ylabel=r"y [nm]", 
                xborder=None, yborder=None, ticks_step=2, vmin=None, vmax=None, 
                equal_aspect=True, title=None, show_colorbar=True):
    """
    Plot a 2D data map with customizable axes, colormap, and formatting options.
    Parameters
    ----------
    data : array-like
        2D array of data to be plotted.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
        Default is None.
    cm : matplotlib.colors.Colormap, optional
        Colormap to use for the plot. Default is cm_inferno.
    xlabel : str, optional
        Label for the x-axis. Default is r"x [nm]".
    ylabel : str, optional
        Label for the y-axis. Default is r"y [nm]".
    xborder : float, optional
        Half-width of the x-axis limits (symmetric around origin).
        Must be provided together with yborder. Default is None.
    yborder : float, optional
        Half-width of the y-axis limits (symmetric around origin).
        Must be provided together with xborder. Default is None.
    ticks_step : int, optional
        Step size for tick placement. Automatically adjusted if it doesn't
        divide evenly into borders. Default is 2.
    vmin : float, optional
        Minimum value for colormap normalization. Default is None.
    vmax : float, optional
        Maximum value for colormap normalization. Default is None.
    equal_aspect : bool, optional
        If True, set equal aspect ratio for the plot. Default is True.
    title : str, optional
        Title for the plot. Default is None.
    show_colorbar : bool, optional
        If True, display a colorbar. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plotted data.
    Raises
    ------
    ValueError
        If only one of xborder or yborder is provided (both must be given together).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    
    if equal_aspect:
        ax.set_aspect('equal')

    extent = None
    if xborder is not None and yborder is not None:
        ax.set_xlim(-xborder, xborder)
        ax.set_ylim(-yborder, yborder)

        while (xborder % ticks_step != 0 and yborder % ticks_step != 0):
            ticks_step += 1
            if ticks_step > 5:
                ticks_step = 1
                break

        ax.set_xticks(np.linspace(-xborder, xborder, round(ticks_step * 2) + 1))
        ax.set_yticks(np.linspace(-yborder, yborder, round(ticks_step * 2) + 1))

        extent = [-xborder, xborder, -yborder, yborder]
    elif xborder is not None or yborder is not None:
        print("\n\nPlotting error!\nBoth 'xborder' and 'yborder' must be provided.\n")

    im = ax.imshow(data, interpolation='none', origin='lower', extent=extent, cmap=cm, vmin=vmin, vmax=vmax)
    ax.tick_params(direction="out", which="both")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, orientation='vertical')

    return ax

def map_grid_plotter(data_list, n, m, **kwargs):
    """
    Displays a grid of map plots from a list of data arrays.
    This function creates a subplot grid of size n x m and plots each data array
    from data_list using the map_plotter function. If there are fewer data arrays
    than subplots, the empty subplots are hidden. The colorbar is disabled for
    individual plots to save space in the grid layout.
    Parameters
    ----------
    data_list : list
        List of data arrays to be plotted. Each element will be plotted in a
        separate subplot using map_plotter.
    n : int
        Number of rows in the subplot grid.
    m : int
        Number of columns in the subplot grid.
    **kwargs : dict
        Additional keyword arguments to pass to map_plotter function.
    Returns
    -------
    None
        Displays the figure using plt.show().
    Notes
    -----
    - If len(data_list) < n*m, empty subplots will have their axes turned off.
    - Colorbars are disabled for individual subplots to maintain clean layout.
    - The figure is automatically adjusted using tight_layout.
    Examples
    --------
    >>> data1 = np.random.rand(10, 10)
    >>> data2 = np.random.rand(10, 10)
    >>> map_grid_plotter([data1, data2], n=1, m=2)
    """
    fig, axes = plt.subplots(n, m, figsize=(4*m, 4*n))
    
    axes = np.atleast_2d(axes).reshape(-1)

    for i, data in enumerate(data_list):
        if i >= len(axes):
            break
        map_plotter(data, ax=axes[i], show_colorbar=False, **kwargs)

    for j in range(len(data_list), len(axes)): # empty subplots if no data
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    
def line_plotter(xdata, ydata, ax=None, xlabel=r"x [-]", ylabel=r"y [-]", color="black",
                    linestyle="-", xlim=None, ylim=None, equal_aspect=False, title=None, label=None):
    """
    Plot a line graph with customizable axes, limits, and styling.
    Parameters
    ----------
    xdata : array-like
        X-axis data points.
    ydata : array-like
        Y-axis data points.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, a new figure and axes are created.
        Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is "x [-]".
    ylabel : str, optional
        Label for the y-axis. Default is "y [-]".
    color : str, optional
        Line color. Default is "black".
    linestyle : str, optional
        Line style (e.g., "-", "--", "-.", ":"). Default is "-".
    xlim : tuple, list, or array-like, optional
        X-axis limits as [min, max]. If None, limits are set to data range.
        Default is None.
    ylim : tuple, list, or array-like, optional
        Y-axis limits as [min, max]. If None, axis auto-scales.
        Default is None.
    equal_aspect : bool, optional
        If True, sets equal aspect ratio for the plot. Default is False.
    title : str, optional
        Title for the plot. If None, no title is set. Default is None.
    label : str, optional
        Label for the line (used in legend). Default is None.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Raises
    ------
    ValueError
        If xlim or ylim are not in the correct format (list, dict, tuple, or numpy array).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    
    if equal_aspect:
        ax.set_aspect('equal')

    if xlim is not None:
        if isinstance(xlim, (list, dict, tuple, np.ndarray)):
            ax.set_xlim(xlim[0], xlim[1])
        else:
            print("\n\nWrong format of 'xlim'!\n")
    else:
        ax.set_xlim(min(xdata), max(xdata))

    if ylim is not None:
        if isinstance(ylim, (list, dict, tuple, np.ndarray)):
            ax.set_ylim(ylim[0], ylim[1])
        else:
            print("\n\nWrong format of 'ylim'!\n")

    ax.plot(xdata, ydata, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return ax

def multi_line_plotter_same_axes(xdata_list, ydata_list, colors=None, linestyles=None, labels=None, 
                                  xlabel=r"x [-]", ylabel=r"y [-]",
                                  xlim=None, ylim=None, equal_aspect=False, title=None,
                                  legend=True):
    """
    Plot multiple lines on the same axes with customizable styling.
    This function creates a single matplotlib figure with multiple line plots overlaid
    on the same axes. It supports customization of colors, line styles, labels, axis
    limits, and other plot properties.
    Parameters
    ----------
    xdata_list : list of array-like
        List of x-axis data arrays for each curve.
    xdata_list : list of array-like
        List of y-axis data arrays for each curve. Must have the same length as xdata_list.
    colors : list of str, optional
        List of color specifications for each line. If None, defaults to "black".
    linestyles : list of str, optional
        List of line style specifications for each line (e.g., "-", "--", "-.").
        If None, defaults to "-" (solid line).
    labels : list of str, optional
        List of labels for each line to be displayed in the legend. If None, no labels
        are shown.
    xlabel : str, optional
        Label for the x-axis. Default is "x [-]".
    ylabel : str, optional
        Label for the y-axis. Default is "y [-]".
    xlim : tuple of float, optional
        Limits for the x-axis as (min, max). If None, limits are determined automatically.
    ylim : tuple of float, optional
        Limits for the y-axis as (min, max). If None, limits are determined automatically.
    equal_aspect : bool, optional
        If True, sets equal aspect ratio for x and y axes. Default is False.
    title : str, optional
        Title for the plot. If None, no title is displayed. Default is None.
    legend : bool, optional
        If True and labels are provided, display a legend. Default is True.
    Returns
    -------
    None
        Displays the plot using plt.show().
    """    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    num_curves = len(xdata_list)
    
    for i in range(num_curves):
        color = colors[i] if colors is not None and i < len(colors) else "black"
        linestyle = linestyles[i] if linestyles is not None and i < len(linestyles) else "-"
        label = labels[i] if labels is not None and i < len(labels) else None

        line_plotter(xdata_list[i], ydata_list[i], ax=ax,
                       xlabel=xlabel, ylabel=ylabel,
                       color=color, linestyle=linestyle,
                       xlim=xlim, ylim=ylim,
                       equal_aspect=equal_aspect, title=title)
        
        # Dodaj label tylko, jeśli jest
        if label is not None:
            ax.plot(xdata_list[i], ydata_list[i], label=label, color=color, linestyle=linestyle)

    if legend and labels is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

def make_field_animation(
    collected_data,
    field_name,
    singleton_params,
    animation_name,
    structure=[],
    cmap='RdBu',
    absolute=False,
    crop_pml=True,
    interval=100,
    percentile=99.5,
    structure_alpha=0.9,
    field_alpha=0.8
):

    if field_name not in collected_data or field_name == 'time_steps':
        raise ValueError(f"Field '{field_name}' not found")

    raw_fields = collected_data[field_name]
    if len(raw_fields) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    structure = np.abs(structure)

    # --------------------------------------------------
    # Field + structure preprocessing (ONCE)
    # --------------------------------------------------
    def process_array(arr):
        if crop_pml:
            pmlx = singleton_params.pml / (singleton_params.xyz_cell[0] / 2)
            pmly = singleton_params.pml / (singleton_params.xyz_cell[1] / 2)

            cx = int(np.ceil(pmlx * arr.shape[1] / 2)*1.15)
            cy = int(np.ceil(pmly * arr.shape[0] / 2)*1.15)

            arr = arr[cy:-cy, cx:-cx]

        return arr.T

    # Process structure
    if len(structure) != 0:
        structure_proc = process_array(structure)
    else:
        structure_proc = np.zeros_like(process_array(raw_fields[0]))

    # Process fields
    fields = []
    for f in raw_fields:
        fld = np.abs(f) if absolute else f
        fields.append(process_array(fld))
    fields = np.array(fields)

    # --------------------------------------------------
    # FIXED global scale (KEY PART)
    # --------------------------------------------------
    if absolute:
        vmin = 0.0
        vmax = np.percentile(fields, percentile)
    else:
        vmax = np.percentile(np.abs(fields), percentile)
        vmin = -vmax

    # --------------------------------------------------
    # Background: gray
    # --------------------------------------------------
    ax.set_facecolor((0.5, 0.5, 0.5))  # neutral gray

    # --------------------------------------------------
    # STRUCTURE overlay (WHITE)
    # --------------------------------------------------
    structure_img = ax.imshow(
        structure_proc,
        cmap='gray',
        origin='lower',
        interpolation='none',
        vmin=np.min(structure_proc),
        vmax=np.max(structure_proc),
        alpha=structure_alpha
    )

    # --------------------------------------------------
    # FIELD overlay (COLOR)
    # --------------------------------------------------
    field_img = ax.imshow(
        fields[0],
        cmap=cmap,
        origin='lower',
        interpolation='none',
        vmin=vmin,
        vmax=vmax,
        alpha=field_alpha
    )

    # Colorbar only for field
    cbar = plt.colorbar(field_img, ax=ax, fraction=0.046, pad=0.04)
    label = f"|{field_name}|" if absolute else field_name
    cbar.set_label(label)

    ax.set_title(f"Field: {field_name}")

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------
    def update(i):
        field_img.set_data(fields[i])
        return (field_img,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(fields),
        interval=interval,
        blit=True
    )

    path = os.path.join(singleton_params.path_to_save, animation_name + ".mp4")
    ani.save(path, writer='ffmpeg')

    print(f"Saved animation: {path}")

    plt.close(fig) if singleton_params.IMG_CLOSE else plt.show()


def animate_field_from_h5(
    h5_filename,
    load_h5data_path=None,
    save_name=None,
    save_path=None,
    dataset_name=None,
    interval=50,
    cmap="inferno",
    vmin=None,
    vmax=None,
    transpose_xy=False,
    IMG_CLOSE=False,
    xzeros=0,
    yzeros=None,
):
    """
    Animate time-dependent field data stored in an HDF5 file.

    FIXED DATA FORMAT
    -----------------
    The HDF5 dataset is assumed to have the layout:
        data[x, y, time]

    No axis reordering or transposition of the data array is performed.
    Any transpose is applied ONLY at the visualization level.

    Parameters
    ----------
    h5_filename : str
        Name of the HDF5 file containing field data.
        If `load_h5data_path` is provided, this is treated as a filename
        relative to that directory.

    dataset_name : str or None, optional
        Name of the dataset inside the HDF5 file.
        If None, the first dataset found in the file is used.

    interval : int, optional
        Delay between animation frames in milliseconds.

    cmap : str, optional
        Matplotlib colormap used for rendering the field.

    vmin, vmax : float or None, optional
        Color scale limits.
        If None, they are computed from the cropped data.

    transpose_xy : bool, optional
        If True, transpose X and Y axes for display purposes ONLY.
        This does NOT affect the underlying data array.

    save_path : str or None, optional
        Directory where the animation file should be saved.
        If None, the animation is not saved to disk.

    save_name : str or None, optional
        Name of the output animation file (e.g. "field.mp4").
        Used only if `save_path` is not None.
        If None and `save_path` is given, defaults to "<h5_filename>.mp4".

    load_h5data_path : str or None, optional
        Directory in which the HDF5 file is searched.
        If None, `h5_filename` is interpreted as a full or relative path.

    IMG_CLOSE : bool, optional
        If True, the matplotlib window is not displayed (useful for batch runs).

    xzeros : int, optional
        Number of grid points to crop from left and right boundaries (PML removal).

    yzeros : int or None, optional
        Number of grid points to crop from bottom and top boundaries.
        If None, defaults to `xzeros`.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The generated animation object.
    """

    # --------------------------------------------------
    # Resolve HDF5 file path
    # --------------------------------------------------
    if load_h5data_path is not None:
        h5_path = os.path.join(load_h5data_path, h5_filename)
    else:
        h5_path = h5_filename

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = list(f.keys())[0]
        data = np.array(f[dataset_name])

    if data.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {data.shape}")

    Nx, Ny, Nt = data.shape

    # --------------------------------------------------
    # Default yzeros
    # --------------------------------------------------
    if yzeros is None:
        yzeros = xzeros

    xzeros = max(0, min(xzeros, Nx // 2))
    yzeros = max(0, min(yzeros, Ny // 2))

    # --------------------------------------------------
    # Crop PML regions (x,y only)
    # --------------------------------------------------
    data = data[
        xzeros : Nx - xzeros,
        yzeros : Ny - yzeros,
        :
    ]

    Nx, Ny, Nt = data.shape

    # --------------------------------------------------
    # Color scale
    # --------------------------------------------------
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # --------------------------------------------------
    # Plot setup
    # --------------------------------------------------
    fig, ax = plt.subplots()

    frame0 = data[:, :, 0]
    if transpose_xy:
        frame0 = frame0.T

    img = ax.imshow(
        frame0,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    plt.colorbar(img, ax=ax)

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------
    def update(t):
        frame = data[:, :, t]
        if transpose_xy:
            frame = frame.T
        img.set_array(frame)
        ax.set_title(f"frame {t}/{Nt-1}")
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval,
        blit=True,
    )

    # --------------------------------------------------
    # Save animation
    # --------------------------------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        if save_name is None:
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            save_name = f"{base}.mp4"

        save_fullpath = os.path.join(save_path, save_name)
        anim.save(save_fullpath, writer="ffmpeg")

    # --------------------------------------------------
    # Show / close
    # --------------------------------------------------
    if not IMG_CLOSE:
        plt.show()

    return anim

def animate_field_from_h5_physical(
    h5_filename,
    load_h5data_path=None,
    save_name=None,
    save_path=None,
    dataset_name=None,
    interval=50,
    cmap="inferno",
    vmin=None,
    vmax=None,
    transpose_xy=False,
    IMG_CLOSE=False,

    # --- physical axis definition ---
    x_phys_range=None,   # (xmin, xmax) e.g. (-2000, 2000)
    y_phys_range=None,   # (ymin, ymax)

    # --- PML / border crop ---
    xzeros=0,
    yzeros=None,

    # --- zoom (fraction of current size) ---
    x_zoom=1.0,
    y_zoom=1.0,

    # --- artifact killing (set value to 1) ---
    mask_left=0,
    mask_right=0,
    mask_bottom=0,
    mask_top=0,
):
    """
    Animate 2D field data stored as data[x, y, time] with:
    - user-defined physical axes
    - centered zoom
    - independent edge masking
    """
    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    if x_phys_range is None or y_phys_range is None:
        raise ValueError(
            "You must provide x_phys_range and y_phys_range, "
            "e.g. x_phys_range=(-2000,2000)"
        )

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
    # Crop PML regions (ONCE)
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
    # Artifact masking (ONCE, BEFORE scaling)
    # --------------------------------------------------
    if mask_left > 0:
        data[:mask_left, :, :] = 1.0
    if mask_right > 0:
        data[-mask_right:, :, :] = 1.0
    if mask_bottom > 0:
        data[:, :mask_bottom, :] = 1.0
    if mask_top > 0:
        data[:, -mask_top:, :] = 1.0

    # --------------------------------------------------
    # Zoom (fractional, centered)
    # --------------------------------------------------
    cx, cy = Nx // 2, Ny // 2

    hx = int(0.5 * x_zoom * Nx)
    hy = int(0.5 * y_zoom * Ny)

    x1, x2 = cx - hx, cx + hx
    y1, y2 = cy - hy, cy + hy

    data = data[x1:x2, y1:y2, :]

    Nx, Ny, Nt = data.shape

    # --------------------------------------------------
    # Physical axes (mapped from user-defined ranges)
    # --------------------------------------------------
    x_min0, x_max0 = x_phys_range
    y_min0, y_max0 = y_phys_range

    x_full = np.linspace(x_min0, x_max0, Nx0 - 2 * xzeros)
    y_full = np.linspace(y_min0, y_max0, Ny0 - 2 * yzeros)

    x_phys = x_full[x1:x2]
    y_phys = y_full[y1:y2]

    extent = [
        x_phys[0], x_phys[-1],
        y_phys[0], y_phys[-1]
    ]

    # --------------------------------------------------
    # Color scale (AFTER everything)
    # --------------------------------------------------
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # --------------------------------------------------
    # Plot setup
    # --------------------------------------------------
    fig, ax = plt.subplots()

    frame0 = data[:, :, 0]
    if transpose_xy:
        frame0 = frame0.T

    img = ax.imshow(
        frame0,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )

    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")

    plt.colorbar(img, ax=ax)

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------
    def update(t):
        frame = data[:, :, t]
        if transpose_xy:
            frame = frame.T
        img.set_array(frame)
        ax.set_title(f"frame {t}/{Nt-1}")
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval,
        blit=True,
    )

    # --------------------------------------------------
    # Save animation
    # --------------------------------------------------
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        if save_name is None:
            base = os.path.splitext(os.path.basename(h5_filename))[0]
            save_name = f"{base}.mp4"

        anim.save(os.path.join(save_path, save_name), writer="ffmpeg")

    # --------------------------------------------------
    # Show / close
    # --------------------------------------------------
    if not IMG_CLOSE:
        plt.show()
    else:
        plt.close(fig)

    return anim