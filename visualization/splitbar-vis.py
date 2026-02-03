import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================
# Helper: cuboid definition
# =========================
def cuboid(center, size):
    cx, cy, cz = center
    dx, dy, dz = size

    x = [cx - dx/2, cx + dx/2]
    y = [cy - dy/2, cy + dy/2]
    z = [cz - dz/2, cz + dz/2]

    v = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]],
    ])

    faces = [
        [v[i] for i in [0,1,2,3]],
        [v[i] for i in [4,5,6,7]],
        [v[i] for i in [0,1,5,4]],
        [v[i] for i in [2,3,7,6]],
        [v[i] for i in [1,2,6,5]],
        [v[i] for i in [4,7,3,0]],
    ]
    return faces


# =========================
# Geometry parameters
# =========================
bar_length = 120.0
bar_width  = 30.0
bar_height = 20.0
gap        = 20.0

left_center  = (-(gap/2 + bar_length/2), 0, 0)
right_center = ( (gap/2 + bar_length/2), 0, 0)

# =========================
# Figure
# =========================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# =========================
# Split-bar antenna
# =========================
gold = (1.0, 0.75, 0.2)

for c in [left_center, right_center]:
    poly = Poly3DCollection(
        cuboid(c, (bar_length, bar_width, bar_height)),
        facecolor=gold,
        edgecolor="k",
        linewidths=0.6,
        alpha=1.00
    )
    ax.add_collection3d(poly)

# =========================
# Planes + dashed frames
# =========================
L = 120
plane_alpha = 0.075
lw = 1.5

# --- XY (z=0)
xx, yy = np.meshgrid([-L, L], [-L, L])
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, color="blue", alpha=plane_alpha)
ax.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L], [0]*5,
        color="blue", linestyle="--", linewidth=lw)
ax.text(0.9*L, 0.9*L, 0, "XY", color="blue", fontsize=14, ha="center", va="center")

# --- XZ (y=0)
xx, zz = np.meshgrid([-L, L], [-L, L])
yy = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, color="green", alpha=plane_alpha)
ax.plot([-L, L, L, -L, -L], [0]*5, [-L, -L, L, L, -L],
        color="green", linestyle="--", linewidth=lw)
ax.text(0, 0, 0.9*L, "XZ", color="green", fontsize=14, ha="center")

# --- YZ (x=0)
yy, zz = np.meshgrid([-L, L], [-L, L])
xx = np.zeros_like(yy)
ax.plot_surface(xx, yy, zz, color="red", alpha=plane_alpha)
ax.plot([0]*5, [-L, L, L, -L, -L], [-L, -L, L, L, -L],
        color="red", linestyle="--", linewidth=lw)
ax.text(0, 0.9*L, 0, "YZ", color="red", fontsize=14, va="center")

# =========================
# Clean look
# =========================
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)
ax.set_box_aspect([1, 1, 0.75])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.grid(False)

ax.view_init(elev=20, azim=40)

plt.tight_layout()
plt.show()
