import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    
    material_name = sys.argv[1] if len(sys.argv) > 1 else "air"
    # =====================================================
    # split_bar_AuTiSiO2()
    # split_bar_AuTiX()
    bowtie_substrate_experiment(material_name)
    
if __name__ == "__main__":
    run()
