import meep as mp

xm = 1000
class SimulationConfig:
    """
    Globalne parametry symulacji.
    NIE zawiera geometrii.
    NIE zawiera typów anten.
    """

    def __init__(self):

        # =====================================================
        # SYMULACJA
        # =====================================================

        self.resolution = 150
        self.courant = 0.5
        self.sim_time = 5000 / xm  # w jednostkach Meep (µm)

        self.sim_dimensions = 3

        self.symmetries = [
            mp.Mirror(direction=mp.X, phase=-1),
            mp.Mirror(direction=mp.Y, phase=+1)
        ]

        # =====================================================
        # DOMENA (STAŁA – DO PORÓWNAŃ!)
        # =====================================================

        self.cell_size = mp.Vector3(
            80.0,   # x
            60.0,   # y
            40.0    # z
        )

        self.pml_thickness = 0.5

        # =====================================================
        # SOURCE
        # =====================================================

        self.lambda0 = 8100 / 1000
        self.src_width = 1000 / 1000
        self.src_amp = 1.0

        self.component = mp.Ex

        # Źródło ustawione nad strukturą
        self.source_z_offset = 10.0

    # --------------------------------------------------------

    @property
    def frequency(self):
        return 1.0 / self.lambda0

    @property
    def frequency_width(self):
        return 1.0 / self.src_width