import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    # =====================================================
    # TRA_Novotn()
    # split_bar_AuTiSiO2()
    # wave_shape_theo()
    experiment_bow_tie_test()
    
if __name__ == "__main__":
    run()
