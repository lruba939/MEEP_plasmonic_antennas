import os
import meep as mp

HPC = os.environ.get("SCRATCH") is not None

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

# def create_directory_names(SIM_NAME):
#     path_to_save = os.path.join("results", SIM_NAME)
#     animations_folder_path = os.path.join(path_to_save, "animations")
#     
#     if mp.am_master():
#         if not os.path.exists(path_to_save):
#             os.makedirs(path_to_save)
#         if not os.path.exists(animations_folder_path):
#             os.makedirs(animations_folder_path)
# 
#         return path_to_save, animations_folder_path

def create_directory(SIM_NAME):
    # ==========================================
    # Select base directory
    # ==========================================
    if HPC:
        base_dir = os.environ["SCRATCH"]
    else:
        base_dir = os.getcwd()

    # ==========================================
    # Paths
    # ==========================================
    path_to_save = os.path.join(
        base_dir,
        "results",
        SIM_NAME
    )

    animations_folder_path = os.path.join(
        path_to_save,
        "animations"
    )

    # ==========================================
    # Create directories only on master process
    # ==========================================
    if mp.am_master():

        os.makedirs(path_to_save, exist_ok=True)
        os.makedirs(animations_folder_path, exist_ok=True)

        print("=" * 60)
        print(f"HPC mode: {HPC}")
        print(f"Saving results to:")
        print(path_to_save)
        print("=" * 60)

    return path_to_save, animations_folder_path
