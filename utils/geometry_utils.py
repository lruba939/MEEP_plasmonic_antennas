# from ..main.src import params
import meep as mp
import numpy as np

# inicialize singleton of all parameters
# p = params.SimParams()

def make_cell(x=None, y=None, z=None, config=None):
    if config is not None:
        cell = mp.Vector3(config.cell_size[0],
            config.cell_size[1],
            config.cell_size[2])
    else:
        cell = mp.Vector3(x, y, z)
    return cell

# # geometry
# def make_medium():
#     if p.antenna_type == "split-bar":
#         geometry = make_split_bar()
#     elif p.antenna_type == "custom-split-bar":
#         geometry = make_custom_split_bar()
#     else:
#         raise ValueError(f"Unknown antenna type: {p.antenna_type}")
#     return geometry

# def make_split_bar():
#     split_bar = [
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.z_height),
#             center = p.splitbar_center[0], # upper bar
#             material = p.material,
#         ),
        
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.z_height),
#             center = p.splitbar_center[1], # lower bar
#             material = p.material,
#         )
#     ]
#     return split_bar

# def make_custom_split_bar():
#     custom_split_bar = [
#         # RIGHT BAR
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.First_layer[2]),
#             center = p.custom_center[0], # right bar
#             material = p.material_1,
#         ),
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.Second_layer[2]),
#             center = p.custom_center[0] - mp.Vector3(0, 0, p.First_layer[2]/2.0 + p.Second_layer[2]/2.0) , # right bar
#             material = p.material_2,
#         ),
#         # LEFT BAR
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.First_layer[2]),
#             center = p.custom_center[1], # left bar
#             material = p.material_1,
#         ),
#         mp.Block(
#             mp.Vector3(p.x_width, p.y_length, p.Second_layer[2]),
#             center = p.custom_center[1] - mp.Vector3(0, 0, p.First_layer[2]/2.0 + p.Second_layer[2]/2.0) , # left bar
#             material = p.material_2,
#         ),
#         # substrate
#         mp.Block(
#             mp.Vector3(p.Third_layer[0], p.Third_layer[1], p.Third_layer[2]),
#             center = mp.Vector3(0, 0, (-1)*(p.First_layer[2]/2.0 + p.Second_layer[2]/2.0 + p.Third_layer[2]/2.0)) , # left bar
#             material = p.material_3,
#         )
#     ]
#     return custom_split_bar

# rounded edges !!!

def unit(v):
    return v / np.linalg.norm(v)

def flare_angle(P_center, P_left, P_right):
    v1 = P_left  - P_center
    v2 = P_right - P_center

    v1 = unit(v1)
    v2 = unit(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)

def angle_bisector(P_center, P_left, P_right):
    v1 = unit(P_left  - P_center)
    v2 = unit(P_right - P_center)

    return unit(v1 + v2)

def inscribed_circle_center(P_center, P_left, P_right, R):
    theta = flare_angle(P_center, P_left, P_right)
    bis   = angle_bisector(P_center, P_left, P_right)

    L = R / np.sin(theta / 2.0)

    C = P_center + L * bis
    return C

def tangent_points(P_center, P_left, P_right, R):
    theta = flare_angle(P_center, P_left, P_right)
    d = R / np.tan(theta / 2.0)

    v1 = unit(P_left  - P_center)
    v2 = unit(P_right - P_center)

    T1 = P_center + d * v1
    T2 = P_center + d * v2
    return T1, T2

def safe_cut_point(P_center, P_left, P_right, R, overshoot=0.01):
    theta = flare_angle(P_center, P_left, P_right)
    bis   = angle_bisector(P_center, P_left, P_right)

    L = R / np.sin(theta / 2.0)

    # move slightly beyond the tangent point to avoid artifacts
    d = L - R * (1.0 - overshoot)

    return P_center - d * bis

def triangle_centroid(A, B, C):
    """
    Compute centroid (x, y) of triangle defined by points A, B, C.

    Parameters
    ----------
    A, B, C : np.ndarray shape (2,)
        Triangle vertices

    Returns
    -------
    np.ndarray shape (2,)
        Centroid coordinates
    """
    return (A + B + C) / 3.0

def clear_edges(
    points,
    radius,
    height,
    z_offset,
    overshoot=0.01
):
    """
    Clear all corners of a 2D polygon using air-cut prisms.

    Parameters
    ----------
    points : list of np.ndarray
        Polygon vertices in cyclic order, shape (N, 2)
    radius : float
        Fillet radius
    height : float
        Extrusion height
    overshoot : float
        Fraction of radius used to extend the air cut beyond tangency

    Returns
    -------
    geometry : list
        List of mp.GeometricObject (Prisms)
    """

    geometry = []
    N = len(points)

    for i in range(N):
        P_center = points[i]
        P_left   = points[(i - 1) % N]
        P_right  = points[(i + 1) % N]

        # --- fillet geometry ---
        T1, T2 = tangent_points(P_center, P_left, P_right, radius)
        P_cut  = safe_cut_point(P_center, P_left, P_right, radius, overshoot)

        # --- air cut (corner removal) ---
        air_triangle = [
            mp.Vector3(T1[0], T1[1], 0),
            mp.Vector3(T2[0], T2[1], 0),
            mp.Vector3(P_cut[0], P_cut[1], 0)
        ]

        centroid = triangle_centroid(T1, T2, P_cut)

        geometry.append(
            mp.Prism(
                air_triangle,
                height=height,
                material=mp.air,
                center=mp.Vector3(centroid[0], centroid[1], z_offset)
            )
        )

    return geometry

def fillet_polygon(
    points,
    radius,
    height,
    material,
    z_offset,
    axis=np.array([0, 0, 1])
):
    """
    Fillets all corners of a 2D polygon using material cylinders.

    Parameters
    ----------
    points : list of np.ndarray
        Polygon vertices in cyclic order, shape (N, 2)
    radius : float
        Fillet radius
    height : float
        Extrusion height
    material : mp.Medium
        Material of the solid polygon
    axis : array-like
        Extrusion axis (default z)

    Returns
    -------
    geometry : list
        List of mp.GeometricObject (Cylinders)
    """

    geometry = []
    N = len(points)

    for i in range(N):
        P_center = points[i]
        P_left   = points[(i - 1) % N]
        P_right  = points[(i + 1) % N]

        # --- fillet geometry ---
        C = inscribed_circle_center(P_center, P_left, P_right, radius)

        # --- fillet cylinder ---
        geometry.append(
            mp.Cylinder(
                center=mp.Vector3(C[0], C[1], z_offset),
                radius=radius,
                height=height,
                axis=mp.Vector3(*axis),
                material=material
            )
        )

    return geometry

def corrected_gap(g_target, R, theta):
        """
        Compute the nominal gap that must be used in geometry so that,
        after corner rounding with radius R, the effective gap equals g_target.

        Parameters
        ----------
        g_target : float
            Desired physical gap after rounding
        R : float
            Fillet radius
        theta : float
            Opening angle of the corner (radians)

        Returns
        -------
        g_input : float
            Gap to use in the sharp geometry
        """
        delta = R / np.sin(theta / 2.0) - R
        # print(f"Gap correction: For target gap {g_target*1e3:.2f} nm, radius {R*1e3:.1f} nm, and angle {np.rad2deg(theta):.1f} deg, the correction is {delta*1e3:.2f} nm.")
        return g_target - 2.0 * delta