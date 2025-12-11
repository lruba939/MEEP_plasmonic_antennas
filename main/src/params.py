## Singleton of parameters
import numpy as np
import meep as mp
from meep.materials import Au, Cr, W, SiO2, Ag

class SimParams:
    _instance=None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance=object.__new__(cls)
            cls._instance._init_parameters()
        return cls._instance
    
    def _init_parameters(self):
        # SYSTEM
        self.IMG_CLOSE =  True

        # Geometry
        self.material   =   Ag
        
        self.x_width    =   0.7
        self.y_length   =   0.19
        self.z_height   =   0.0
        self.gap_size   =   0.05
        self.pad        =   2.0
        
        self.xyz_cell   =   [self.x_width+2*self.pad,
                            2*self.y_length + self.gap_size + 2*self.pad,
                            0]
        self.center     =   [mp.Vector3(0, self.y_length/2.0 + self.gap_size/2.0, 0), # upper bar
                            mp.Vector3(0, (-1)*(self.y_length/2.0 + self.gap_size/2.0), 0)] # lower bar
        # self.center     =   [mp.Vector3(0, 0, -10.), # upper bar
                            # mp.Vector3(0, 0, -10.)] # lower bar
        
        # Source
        self.lambda0    =   1.0 #um
        self.freq       =   1.0 / self.lambda0
        self.freq_width =   self.freq * 0.5
        self.component  =   mp.Ey
        self.xyz_src    =   [2.0, 0.0, 0.0]
        self.src_size   =   [0.0, 3.6, 0.0]
        
        ## Simulation settings
        self.pml                    =   0.25
        self.resolution             =   200
        self.sim_time               =   20
        self.animations_folder_path =   "results"
        self.animations_until       =   20
        self.animations_step        =   0.1377
        self.animations_fps         =   10

        self.eps_data_container     =   []
        self.E_comp_data_container  =   []
        self.empty_cell_E_comp_data_container =   []

    def reset_to_defaults(self):
        self._init_parameters()
        
    def showParams(self):
        print("\n\n#################################\nSimulation and System Parameters:\n")
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if not isinstance(v, (list, dict, tuple, np.ndarray)): #
                    print(f"{k}={v}")
                else:
                    print(f"{k}={v[:5]}")
        print("#################################\n\n")

    def saveParams(self, filename="results/simulation_params.txt"):
        """
        Saves simulation/system parameters to a file.

        Args:
            filename (str): Nazwa pliku, do którego zostaną zapisane parametry.
        """
        # Otwarcie pliku do zapisywania
        with open(filename, "w") as f:
            # Nagłówek
            header = "\n\n#################################\nSimulation and System Parameters:\n"
            f.write(header)

            # Iteracja po atrybutach
            for k, v in self.__dict__.items():
                if not k.startswith('_'):
                    if not isinstance(v, (list, dict, tuple, np.ndarray)):
                        line = f"{k}={v}\n"
                        f.write(line)
                    else:
                        line = f"{k}={str(v[:5])}\n"
                        f.write(line)

            # Stopka
            footer = "#################################\n\n"
            f.write(footer)