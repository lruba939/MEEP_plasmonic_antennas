import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    
    # default values
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    material_name = sys.argv[2] if len(sys.argv) > 2 else "air"

    match mode:

        case 1:
            bowtie_substrate_experiment(material_name)

        case 2:
            bowtie_substrate_experiment_LT(material_name)

        case 3:
            bowtie_substrate_experiment_MIR(material_name)

        case 4:
            bowtie_substrate_ONLY_experiment(material_name)

        case 5:
            after_hpc_redraw(material_name)

if __name__ == "__main__":
    run()