import meep as mp

class VolumeSet:
    """
    Containers for all user volumes in stores.

    They are divided into:
        - volume (computational)
        - vis_volume (visualization)
    """

    def __init__(self, cell_size, antenna=None, top_z=None):

        self.volume = {}
        self.vis_volume = {}

        cx = cell_size.x
        cy = cell_size.y
        cz = cell_size.z

        # =====================================================
        # PLANES FOR CALCULATIONS
        # =====================================================

        self.volume["XY"] = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cx, cy, 0.0)
        )

        self.volume["XZ"] = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cx, 0.0, cz)
        )

        self.volume["YZ"] = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(0.0, cy, cz)
        )

        if top_z is not None:
            top_shift_z = top_z
        elif antenna is not None and hasattr(antenna, "thickness"):
            top_shift_z = antenna.thickness / 2.0
        else:
            top_shift_z = 0.0

        self.volume["XY_TOP"] = mp.Volume(
            center=mp.Vector3(0, 0, top_shift_z),
            size=mp.Vector3(cx, cy, 0.0)
        )

        # =====================================================
        # VISUALIZATION PLANES
        # =====================================================

        self.vis_volume["XY"] = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cx, cy, 0.0)
        )

        self.vis_volume["XZ"] = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cx, 0.0, cz)
        )

        if antenna is not None and hasattr(antenna, "gap"):
            vis_shift_x = antenna.gap * 1.5
        else:
            vis_shift_x = 0.0

        self.vis_volume["YZ"] = mp.Volume(
            center=mp.Vector3(vis_shift_x, 0, 0),
            size=mp.Vector3(0.0, cy, cz)
        )