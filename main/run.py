import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    # =====================================================
    # split_bar_AuTiSiO2()
    # split_bar_AuTiX()
    bowtie_substrate_experiment()
    
if __name__ == "__main__":
    run()
