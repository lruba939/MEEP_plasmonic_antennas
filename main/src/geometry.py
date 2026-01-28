from . import params
import meep as mp
import numpy as np

# inicialize singleton of all parameters
p = params.SimParams()

# cell is the whole sim box
def make_cell():
    cell = mp.Vector3(p.xyz_cell[0], p.xyz_cell[1], p.xyz_cell[2])
    return cell

# geometry
def make_medium():
    if p.antenna_type == "split-bar":
        geometry = make_split_bar()
    elif p.antenna_type == "bow-tie":
        geometry = make_bow_tie()
    elif p.antenna_type == "custom-split-bar":
        geometry = make_custom_split_bar()
    else:
        raise ValueError(f"Unknown antenna type: {p.antenna_type}")
    return geometry

def make_split_bar():
    split_bar = [
        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.z_height),
            center = p.splitbar_center[0], # upper bar
            material = p.material,
        ),
        
        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.z_height),
            center = p.splitbar_center[1], # lower bar
            material = p.material,
        )
    ]
    return split_bar

def make_bow_tie():

    P1 = np.array([p.bowtie_center[0] + p.gap_size/2.0, p.bowtie_center[1]]) # right tips' point on the x-axis
    # Rn we assume the bow-tie is a equilateral triangle !!!
    P2 = P1 + p.bowtie_amp * np.array([1.0, np.tan(np.deg2rad(30))])
    P3 = P2 * np.array([1.0, -1.0]) # mirror on the x-axis

    tip_right = [mp.Vector3(*P1),
                 mp.Vector3(*P2),
                 mp.Vector3(*P3)]
    
    mirror = np.array([-1.0, 1.0]) # mirror on the y-axis
    tip_left = [mp.Vector3(*(P1*mirror)),
                mp.Vector3(*(P2*mirror)),
                mp.Vector3(*(P3*mirror))]

    bow_tie = [
        mp.Prism(tip_right, height=p.bowtie_thickness, material=p.material),
        mp.Prism(tip_left, height=p.bowtie_thickness, material=p.material)
        ]

    if p.bowtie_radius >= 0 + 1e-12: # + to avoid floating point errors
        # right tip clearing
        bow_tie += clear_edges(
            points=[P1, P2, P3],
            radius=p.bowtie_radius,
            height=p.bowtie_thickness
        )
        # left tip fillet (after mirroring)
        bow_tie += clear_edges(
            points=[P1*mirror, P2*mirror, P3*mirror],
            radius=p.bowtie_radius,
            height=p.bowtie_thickness
        )
        # right tip fillet
        bow_tie += fillet_polygon(
            points=[P1, P2, P3],
            radius=p.bowtie_radius,
            height=p.bowtie_thickness
        )
        # left tip fillet (after mirroring)
        bow_tie += fillet_polygon(
            points=[P1*mirror, P2*mirror, P3*mirror],
            radius=p.bowtie_radius,
            height=p.bowtie_thickness
        )

    return bow_tie

def make_custom_split_bar():
    custom_split_bar = [
        # RIGHT BAR
        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.Au_part[2]),
            center = p.custom_center[0], # right bar
            material = p.material_1,
        ),
        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.Ti_part[2]),
            center = p.custom_center[0] - mp.Vector3(0, 0, p.Au_part[2]/2.0 + p.Ti_part[2]/2.0) , # right bar
            material = p.material_2,
        ),
        
        # LEFT BAR
        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.Au_part[2]),
            center = p.custom_center[1], # left bar
            material = p.material_1,
        ),

        mp.Block(
            mp.Vector3(p.x_width, p.y_length, p.Ti_part[2]),
            center = p.custom_center[1] - mp.Vector3(0, 0, p.Au_part[2]/2.0 + p.Ti_part[2]/2.0) , # left bar
            material = p.material_2,
        )
    ]
    return custom_split_bar

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

def clear_edges(
    points,
    radius,
    height,
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

        geometry.append(
            mp.Prism(
                air_triangle,
                height=height,
                material=mp.air
            )
        )

    return geometry

def fillet_polygon(
    points,
    radius,
    height,
    material=p.material,
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
                center=mp.Vector3(C[0], C[1], height/2.0),
                radius=radius,
                height=height,
                axis=mp.Vector3(*axis),
                material=material
            )
        )

    return geometry
