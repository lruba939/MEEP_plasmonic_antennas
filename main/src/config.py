import meep as mp
import os

xm = 1000
class SimulationConfig:
    """
    Globalne parametry symulacji.
    NIE zawiera geometrii.
    NIE zawiera typów anten.
    """

    def __init__(self):

        # =====================================================
        # SYMULATION
        # =====================================================
        self.resolution = 2000
        self.courant = 0.5
        self.sim_time = 8000 / xm  # w jednostkach Meep (µm)
        self.sim_time_step = 22 / xm

        self.symmetries = [
            mp.Mirror(direction=mp.X, phase=-1),
            mp.Mirror(direction=mp.Y, phase=+1)
        ]

        # =====================================================
        # CELL
        # =====================================================

        self.pml = 30/xm
        self.pad = 40/xm

        self.cell_size = [
            180.0 / xm + 2*self.pad + 2*self.pml,   # x
            100.0 / xm + 2*self.pad + 2*self.pml,   # y
            50.0 / xm + 2*self.pad + 2*self.pml    # z
        ]

        # =====================================================
        # SOURCE
        # =====================================================

        self.src_type = "gaussian"  # "continuous" or "gaussian"
        self.src_is_integrated = False # if source overlaps with PML regions use True
        self.lambda0 = 8100 / xm
        self.src_width = 1000 / xm
        self.src_amp = 1.0
        self.src_cutoff = 3 # number of widths used to smoothly turn on/off the source; reduces high-frequency artifacts
        self.component = mp.Ex
        self.src_center = [
            0.0,    # x
            0.0,    # y
            48.0 / xm  # z
        ]
        self.src_size = [
            200.0 / xm,  # x
            100.0 / xm,  # y
            0.0 / xm    # z
        ]

        # =====================================================
        # ANIMATIONS
        # =====================================================
        self.IMG_CLOSE = True
        self.animations_fps         =   15
        self.path_to_save           =   "results/"
        self.animations_folder_path =   os.path.join(self.path_to_save, "animations")

    # --------------------------------------------------------

    @property
    def frequency(self):
        return 1.0 / self.lambda0

    @property
    def frequency_width(self):
        return 1.0 / self.src_width