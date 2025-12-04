import meep as mp
import numpy as np
import os

from visualization.plotter import *

# inicialize singleton of all parameters

def show_data_img(datas_arr, norm_bool, cmap_arr, alphas):
    for idx, data in enumerate(datas_arr):
        if norm_bool[idx]:
            data = np.abs(data) # complex -> real
        plt.imshow(data.transpose(), interpolation="spline36", cmap=cmap_arr[idx], alpha=alphas[idx])
        plt.xticks([])  # Turn off x-axis numbers
        plt.yticks([])  # Turn off y-axis numbers
        plt.axis("on")
    plt.show()
    
def make_animation(singleton_params, sim, animation_name):
    animation_name = animation_name + ".mp4"
    sim.reset_meep()
    animate = mp.Animate2D(sim, fields=singleton_params.component, normalize = True)
    sim.run(mp.at_every(singleton_params.animations_step, animate), until=singleton_params.animations_until)
    animate.to_mp4(filename = os.path.join(singleton_params.animations_folder_path, animation_name), fps = singleton_params.animations_fps)