import meep as mp

class VolumeSet:
    """
    Containers for all user volumes in stores.

    They are divided into:
        - volume (computational)
        - vis_volume (visualization)
    """

    def __init__(self, cell_size, antenna=None, top_z=None, extra_vols_in_gap=False):

        extra = extra_vols_in_gap
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

        if extra_vols_in_gap:
            # XY
            self.volume["XY_5"] = mp.Volume(
                center=mp.Vector3(0, 0, top_shift_z),
                size=mp.Vector3(antenna.gap*2, antenna.width*2, 0.0)
            )
            self.volume["XY_4"] = mp.Volume(
                center=mp.Vector3(0, 0, top_shift_z/4.0*3.0),
                size=mp.Vector3(antenna.gap*2, antenna.width*2, 0.0)
            )
            self.volume["XY_3"] = mp.Volume(
                center=mp.Vector3(0, 0, top_shift_z/4.0*2.0),
                size=mp.Vector3(antenna.gap*2, antenna.width*2, 0.0)
            )
            self.volume["XY_2"] = mp.Volume(
                center=mp.Vector3(0, 0, top_shift_z/4.0),
                size=mp.Vector3(antenna.gap*2, antenna.width*2, 0.0)
            )
            self.volume["XY_1"] = mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(antenna.gap*2, antenna.width*2, 0.0)
            )

            # XZ
            self.volume["XZ_5"] = mp.Volume(
                center=mp.Vector3(0, antenna.width, 0),
                size=mp.Vector3(antenna.gap*2, 0.0, antenna.thickness*2)
            )
            self.volume["XZ_4"] = mp.Volume(
                center=mp.Vector3(0, antenna.width/4.0*3.0, 0),
                size=mp.Vector3(antenna.gap*2, 0.0, antenna.thickness*2)
            )
            self.volume["XZ_3"] = mp.Volume(
                center=mp.Vector3(0, antenna.width/4.0*2.0, 0),
                size=mp.Vector3(antenna.gap*2, 0.0, antenna.thickness*2)
            )
            self.volume["XZ_2"] = mp.Volume(
                center=mp.Vector3(0, antenna.width/4.0, 0),
                size=mp.Vector3(antenna.gap*2, 0.0, antenna.thickness*2)
            )
            self.volume["XZ_1"] = mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(antenna.gap*2, 0.0, antenna.thickness*2)
            )

            # YZ
            self.volume["YZ_5"] = mp.Volume(
                center=mp.Vector3(antenna.gap, 0, 0),
                size=mp.Vector3(0.0, antenna.width*2, antenna.thickness*2)
            )
            self.volume["YZ_4"] = mp.Volume(
                center=mp.Vector3(antenna.gap/4.0*3.0, 0, 0),
                size=mp.Vector3(0.0, antenna.width*2, antenna.thickness*2)
            )
            self.volume["YZ_3"] = mp.Volume(
                center=mp.Vector3(antenna.gap/4.0*2.0, 0, 0),
                size=mp.Vector3(0.0, antenna.width*2, antenna.thickness*2)
            )
            self.volume["YZ_2"] = mp.Volume(
                center=mp.Vector3(antenna.gap/4.0, 0, 0),
                size=mp.Vector3(0.0, antenna.width*2, antenna.thickness*2)
            )
            self.volume["YZ_1"] = mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(0.0, antenna.width*2, antenna.thickness*2)
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