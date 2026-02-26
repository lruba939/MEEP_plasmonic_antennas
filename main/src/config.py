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
        mp.Simulation.eps_averaging = False
        self.resolution = 1500
        self.courant = 0.5
        self.sim_time = 5000 / xm  # w jednostkach Meep (µm)
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
            200.0 / xm + 2*self.pad + 2*self.pml,   # x
            100.0 / xm + 2*self.pad + 2*self.pml,   # y
            50.0 / xm + 2*self.pad + 2*self.pml    # z
        ]

        # =====================================================
        # SOURCE
        # =====================================================

        self.lambda0 = 8100 / 1000
        self.src_width = 1000 / 1000
        self.src_amp = 1.0
        self.src_cutoff = 3 # number of widths used to smoothly turn on/off the source; reduces high-frequency artifacts


        self.component = mp.Ex

        # Źródło ustawione nad strukturą
        self.source_z_offset = 10.0

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
    
    # --------------------------------------------------------
    # PRINT CONFIG
    # --------------------------------------------------------

    def showConfig(self):

        print("\n\n#################################")
        print("Simulation Configuration:")
        print("#################################\n")

        for key, value in self.__dict__.items():

            if key.startswith("_"):
                continue

            clean_value = self._format_value(value)
            print(f"{key} = {clean_value}")

        print("\nDerived values:")
        print(f"frequency = {self.frequency}")
        print(f"frequency_width = {self.frequency_width}")

        print("\n#################################\n")


    # --------------------------------------------------------
    # SAVE CONFIG TO FILE
    # --------------------------------------------------------

    def saveConfig(self, filename=None):

        if filename is None:
            filename = os.path.join(self.path_to_save, "simulation_params.txt")

        os.makedirs(self.path_to_save, exist_ok=True)
        os.makedirs(self.animations_folder_path, exist_ok=True)

        with open(filename, "w") as f:

            f.write("\n\n#################################\n")
            f.write("Simulation Configuration:\n")
            f.write("#################################\n\n")

            for key, value in self.__dict__.items():

                if key.startswith("_"):
                    continue

                clean_value = self._format_value(value)
                f.write(f"{key} = {clean_value}\n")

            f.write("\nDerived values:\n")
            f.write(f"frequency = {self.frequency}\n")
            f.write(f"frequency_width = {self.frequency_width}\n")

            f.write("\n#################################\n\n")

    # --------------------------------------------------------
    # VALUE FORMATTER (clean Meep objects)
    # --------------------------------------------------------

    def _format_value(self, value):

        # ---- Symmetries (Mirror objects)
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], mp.Mirror):
                formatted = []
                for m in value:
                    direction = (
                        "X" if m.direction == mp.X else
                        "Y" if m.direction == mp.Y else
                        "Z"
                    )
                    formatted.append(f"Mirror({direction}, phase={m.phase})")
                return formatted
            return value

        # ---- Field component (exact type match)
        if value is mp.Ex:
            return "Ex"
        if value is mp.Ey:
            return "Ey"
        if value is mp.Ez:
            return "Ez"
        if value is mp.Hx:
            return "Hx"
        if value is mp.Hy:
            return "Hy"
        if value is mp.Hz:
            return "Hz"

        return value