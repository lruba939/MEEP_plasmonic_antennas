import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    # =====================================================
    new_geo()
    
if __name__ == "__main__":
    run()
