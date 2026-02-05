import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.taskManager import *
from utils.sys_utils import *

def run():

    # ### Set paths to save results
    # SIM_NAME = "splitbar-AuTi-gap-10nm-lambda-8100nm-L-1800nm"
    # #############################
    
    # p.path_to_save = os.path.join("results", SIM_NAME)
    # p.animations_folder_path = os.path.join(p.path_to_save, "animations")

    # if not os.path.exists(p.path_to_save):
    #     os.makedirs(p.path_to_save)
    # if not os.path.exists(p.animations_folder_path):
    #     os.makedirs(p.animations_folder_path)

    gaps = [10]  # nm
    for gap in gaps:
        print(f"--- Starting simulation for gap size: {gap} nm ---")
        p.gap_size = gap / 1000  # Convert nm to um

        ### Set paths to save results
        SIM_NAME = f"splitbar-AuPd-gap-{int(gap)}nm-lambda-8100nm-L-1800nm"
        #############################
        
        p.path_to_save = os.path.join("results", SIM_NAME)
        p.animations_folder_path = os.path.join(p.path_to_save, "animations")

        if not os.path.exists(p.path_to_save):
            os.makedirs(p.path_to_save)
        if not os.path.exists(p.animations_folder_path):
            os.makedirs(p.animations_folder_path)

        #--- Task 7 ---
        print_task(7, "3D calculations.")
        task_7()

        #--- Task 9 ---
        print_task(9, "Postprocesing - animations and plots.")
        task_9()

        p.reset_to_defaults()
    
if __name__ == "__main__":
    run()
