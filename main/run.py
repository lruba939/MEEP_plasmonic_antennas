import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    # =====================================================
    # TRA_Novotn()
    # split_bar_AuTiSiO2()
    # wave_shape_theo()
    # experiment_bow_tie_test()
    # split_bar_AuTiX()
    # split_bar_AuTiX_SP_new_substr_geometry_XY()
    split_bar_AuTiX_SP_new_substr_geometry_XYZ()
    
if __name__ == "__main__":
    run()
