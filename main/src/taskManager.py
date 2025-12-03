from . import params
from . import simulation

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os

# inicialize singleton of all parameters
p = params.SimParams()

# TASK 1 -------------------------------

def task_1():
    p.showParams()
    sim = simulation.make_sim()
    sim.plot2D()
    plt.show()

# TASK 2 -------------------------------

    ###########################
    # task 2 not working .... #
    ###########################

def task_2(plot=False):
    
    p.showParams()
    
    sim = simulation.make_sim()
    simulation.start_calc(sim)
    
    # eps_data = sim.get_array(
    #         center=p.center[0],
    #         size=p.xyz_cell,
    #         component=mp.Epsilon())
    eps_data = sim.get_epsilon()
    
    if plot:
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
        plt.axis("on")
        plt.show()

    return eps_data

# TASK 3 -------------------------------

def task_3(plot=False, animation=False, animation_name="dupa"):
    p.showParams()
    
    sim = simulation.make_sim()
    simulation.start_calc(sim)
    
    E_data = sim.get_array(center=mp.Vector3(), size=p.xyz_cell, component=p.component)

    if plot:
        plt.figure()
        # plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
        plt.imshow(E_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
        plt.axis("off")
        plt.show()
        
    if animation:
        animation_name = animation_name + ".mp4"
        
        sim.reset_meep()
        animate = mp.Animate2D(sim, fields=p.component, normalize = True)
        sim.run(mp.at_every(0.1, animate), until=10)
        animate.to_mp4(filename = os.path.join(p.animations_folder_path, animation_name), fps = 10)

    return E_data

def task_4():
    size_params = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [4.0, 4.0, 0.0]        
    ]
    
    compontets = [
        mp.Ex,
        mp.Ey,
        mp.Ez
    ]
    
    comp_names = ["Ex", "Ey", "Ez"]
    size_names = ["000", "200", "400", "220", "440"]
    
    for comp_pos, comp in enumerate(compontets):
        comp_name = comp_names[comp_pos]
        for size_pos, new_size in enumerate(size_params):
            size_name = size_names[size_pos]
            print("SET: ", comp_name, "; ", size_name, "\n")        
            p.reset_to_defaults()
            p.component = comp
            p.src_size = new_size
            
            name = "antennas" + comp_name + size_name
            task_3(plot=False, animation=True, animation_name=name)

            p.center = [mp.Vector3(0, 0, -10.), # upper bar
                        mp.Vector3(0, 0, -10.)] # lower bar
            name = "without_antennas" + comp_name + size_name
            task_3(plot=False, animation=True, animation_name=name)