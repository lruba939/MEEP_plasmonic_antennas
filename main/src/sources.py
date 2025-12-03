from . import params
import meep as mp

# inicialize singleton of all parameters
p = params.SimParams()

def make_source():
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=p.freq),
            component=p.component,
            center=mp.Vector3(p.xyz_src[0], p.xyz_src[1], p.xyz_src[2]),
            size = mp.Vector3(p.src_size[0], p.src_size[1], p.src_size[2])
        )
    ]
    return sources