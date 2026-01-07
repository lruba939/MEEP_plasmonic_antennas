from . import params
import meep as mp
import cmath
import math

# inicialize singleton of all parameters
p = params.SimParams()

def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))
    return _pw_amp

def make_source():
    if p.src_type == "continuous":
        return [
            mp.Source(
                src=mp.ContinuousSource(frequency=p.freq, is_integrated=p.src_is_integrated),
                component=p.component,
                center=mp.Vector3(p.xyz_src[0], p.xyz_src[1], p.xyz_src[2]),
                size = mp.Vector3(p.src_size[0], p.src_size[1], p.src_size[2]),
                amplitude=p.src_amp
            )
        ]

    # k = mp.Vector3(z=1)
    # src_center=mp.Vector3(p.xyz_src[0], p.xyz_src[1], p.xyz_src[2])

    elif p.src_type == "gaussian":
        return [mp.Source(
                mp.GaussianSource(p.freq, fwidth=p.freq_width, is_integrated=p.src_is_integrated),
                component=p.component,
                center=mp.Vector3(p.xyz_src[0], p.xyz_src[1], p.xyz_src[2]),
                size = mp.Vector3(p.src_size[0], p.src_size[1], p.src_size[2]),
                amplitude=p.src_amp
            )
        ]
    else:
        raise ValueError(f"Unknown source type: {p.src_type}")