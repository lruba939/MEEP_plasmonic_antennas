## Singleton of parameters
import os
import numpy as np
import meep as mp
from meep.materials import Au, Cr, W, SiO2, Ag
# !!! Fitting parameters for all materials are defined for a unit distance of 1 µm.
xm = 1000 # nm to um conversion factor

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
        mp.Simulation.eps_averaging = True

        ###### Geometry ######
        self.xyz_cell   =   [(264)/xm, (181)/xm, 0.0]  # [nm] z = 0.169
        self.material   =   Au
        self.gap_size   =   16/xm
        self.pad        =   40/xm
        
        ### Diferent antenna types parameters ###
        # Antenna type is a string defining the type of antenna to be used in the simulation
        self.antenna_type = "bow-tie"  # options: "bow-tie", "split-bar"
        # self.antenna_type = "split-bar"

        # Bow tie antenna dimensions
        self.bowtie_amp         =   76/xm
        self.bowtie_radius      =   12/xm
        self.bowtie_thickness   =   24/xm # thickness value CANT be zero !!!
        self.bowtie_flare_angle = 60.0 # we need to know the opening angle to compute the corrected gap size, sorry but im lazy...
        if self.bowtie_radius > 0 + 1e-12: # + to avoid floating point errors
            self.gap_size = self.corrected_gap(self.gap_size, self.bowtie_radius, np.deg2rad(self.bowtie_flare_angle))
        self.bowtie_center     =   [0.0, 0.0]

        # Split bar antenna dimensions
        self.x_width    =   5/xm
        self.y_length   =   11/xm
        self.z_height   =   24/xm
        self.center     =   [mp.Vector3(self.x_width/2.0 + self.gap_size/2.0, 0.0, 0.0), # left bar
                            mp.Vector3((-1)*(self.x_width/2.0 + self.gap_size/2.0), 0.0, 0.0)] # right bar
        
        ###### Source ######
        self.lambda0    =   5000/xm # nm
        self.src_width  =   600/xm # temporal width (sigma) of the Gaussian envelope; controls spectral bandwidth
        self.freq       =   1.0 / self.lambda0
        self.freq_width =   1.0 / self.src_width
        self.component  =   mp.Ex
        self.src_amp    =   1.0
        self.src_cutoff =   5  # number of widths used to smoothly turn on/off the source; reduces high-frequency artifacts
        self.xyz_src    =   [0.0, 0.0, 0.0] # z , 49.5
        self.src_size   =   [160.0/xm, 100.0/xm, 0.0]
        
        ###### Simulation settings ######
        self.Courant_factor         =   0.5
        self.pml                    =   30/xm
        # self.pml                    =   (self.lambda0 + self.lambda0*0.5 ) / 2 #Should be: d_PML = lambda_max / 2
        self.resolution             =   1000
        self.sim_time               =   5000/xm
        # self.animations_step        =   self.Courant_factor * (1 / self.resolution) # From dt = S * dx / c, where c=1 in MEEP units
        self.animations_step        =   22/xm
        self.animations_until       =   5000/xm
        self.animations_fps         =   10
        self.path_to_save           =   "results/"
        self.animations_folder_path =   os.path.join(self.path_to_save, "animations")

    def corrected_gap(self, g_target, R, theta):
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
        return g_target - 2.0 * delta


    def reset_to_defaults(self):
        dir_nam_con = self.path_to_save
        dir_ani_con = self.animations_folder_path
        
        self._init_parameters()
        
        self.path_to_save = dir_nam_con
        self.animations_folder_path = dir_ani_con
        
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
        with open(filename, "w") as f:
            header = "\n\n#################################\nSimulation and System Parameters:\n"
            f.write(header)
            for k, v in self.__dict__.items():
                if not k.startswith('_'):
                    if not isinstance(v, (list, dict, tuple, np.ndarray)):
                        line = f"{k}={v}\n"
                        f.write(line)
                    else:
                        line = f"{k}={str(v[:5])}\n"
                        f.write(line)
            footer = "#################################\n\n"
            f.write(footer)