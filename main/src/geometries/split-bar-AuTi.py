import params
import geometry as geom
import meep as mp
import numpy as np
from meep.materials import Au, Ti, SiO2


# inicialize singleton of all parameters
p = params.SimParams()

xm = 1000  # nm to um

Au_part = [2000, 240, 30] / xm
Ti_part = [2000, 240, 5] / xm

p.x_width = Au_part[0]
p.y_length = Au_part[1]
p.z_height = (Au_part[2] + Ti_part[2])

p.custom_center = [mp.Vector3(p.x_width/2.0 + p.gap_size/2.0, 0.0, 0.0), # left bar
                    mp.Vector3((-1)*(p.x_width/2.0 + p.gap_size/2.0), 0.0, 0.0)] # right bar

p.xyz_cell = [p.x_width*2+p.gap_size+p.pad*2+p.pml*2,   # x
                p.y_length+p.pad*2+p.pml*2,             # y
                p.z_height+p.pad*2+p.pml*2]             # z

def make_split_bar_AuTi():
    split_bar_AuTi = [
        # RIGHT BAR
        mp.Block(
            mp.Vector3(Au_part[0], Au_part[1], Au_part[2]),
            center = p.custom_center[0], # right bar
            material = Au,
        ),
        mp.Block(
            mp.Vector3(Ti_part[0], Ti_part[1], Ti_part[2]),
            center = p.custom_center[0], # right bar
            material = Ti,
        ),
        
        # LEFT BAR
        mp.Block(
            mp.Vector3(Au_part[0], Au_part[1], Au_part[2]),
            center = p.custom_center[1], # left bar
            material = Au,
        ),

        mp.Block(
            mp.Vector3(Ti_part[0], Ti_part[1], Ti_part[2]),
            center = p.custom_center[1], # left bar
            material = Ti,
        ),
    ]
    return split_bar_AuTi