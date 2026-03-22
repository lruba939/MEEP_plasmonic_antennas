import os
import meep as mp

def print_task(task_number, description=None):
    if mp.am_master():
        title = f"## TASK {task_number} ##"
        border = "#" * len(title)
        
        print(border)
        print(title)
        print(border)
        
        if description:
            print(f"Description: {description}")
        
        print("-\n-")

def create_directory(SIM_NAME):
    if mp.am_master():
        path_to_save = os.path.join("results", SIM_NAME)
        animations_folder_path = os.path.join(path_to_save, "animations")
    
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        if not os.path.exists(animations_folder_path):
            os.makedirs(animations_folder_path)

        return path_to_save, animations_folder_path