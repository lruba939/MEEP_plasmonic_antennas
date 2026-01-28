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

    # #--- Task 0 ---
    # print_task(0, "Triggering calculations and saving the most general results.")
    # sim = task_0()

    # #--- Task 1 ---
    # print_task(1, "Making medium - a split bar antenna.")
    # task_1()

    # #--- Task 2 ---
    # print_task(2, "Plotting the dielectric constant of a system.")
    # task_2(plot=True)

    # #--- Task 3 ---
    # print_task(3, "Plotting the scalar electric field E component.")
    # task_3(plot=True, animation=True, animation_name="with_antennas", plot_3D=True, sim=sim)

    # # #--- Task 3 ---
    # # print_task(3, "WITHOUT ANTENNAS; Plotting the scalar electric field E component.")
    # # p.center = [mp.Vector3(-9999, -9999, -9999), # upper bar
    # #             mp.Vector3(-9999, -9999, -9999)] # lower bar
    # # p.bowtie_center = [-9999, -9999]
    # # task_3(plot=False, animation=True, animation_name="without_antennas", plot_3D=True, recalculate=True)
    # p.reset_to_defaults()

    # #--- Task 4 ---
    # print_task(4, "Magnitude of the electric field with and without antennas.")
    # task_4(E_plot = True)

    #--- Task 5 ---
    # print_task(5, "Creating animation of Ex field component over time.")
    # task_5()

    # #--- Task 6 ---
    # print_task(6, "Enhanced field calculation in the gap region.")
    # task_6()

    gaps = [30]  # nm
    for gap in gaps:
        print(f"--- Starting simulation for gap size: {gap} nm ---")
        p.gap_size = gap / 1000  # Convert nm to um

        ### Set paths to save results
        SIM_NAME = f"splitbar-AuTi-gap-{int(gap)}nm-lambda-8100nm-L-1800nm"
        #############################
        
        p.path_to_save = os.path.join("results", SIM_NAME)
        p.animations_folder_path = os.path.join(p.path_to_save, "animations")

        if not os.path.exists(p.path_to_save):
            os.makedirs(p.path_to_save)
        if not os.path.exists(p.animations_folder_path):
            os.makedirs(p.animations_folder_path)

        #--- Task 8 ---
        print_task(8, "3D geometry check.")
        task_8()

        #--- Task 7 ---
        print_task(7, "3D.")
        task_7()

        p.reset_to_defaults()


    
if __name__ == "__main__":
    run()
