import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.taskManager import *
from utils.sys_utils import *

def main():
    #### Task 1
    print_task(1, "Making medium - a split bar antenna.")
    task_1()
    
    # #### Task 2
    # print_task(2, "Making medium - a split bar antenna.")
    # eps_Au = task_2(plot=True)

    # #### Task 3
    # print_task(3, "Calculations of the scalar electric field Ey as result of continuous source radiation.")
    # task_3(plot=True, animation=True, animation_name="antennas_test")
    
    #### Task 4
    print_task(4, "")
    task_4()   
    
if __name__ == "__main__":
    main()
