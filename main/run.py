import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    # =====================================================
    experiment_bow_tie_test()
    
if __name__ == "__main__":
    run()
