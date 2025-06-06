import meshio
import numpy as np
from nemo.util.plotting import plot_3d_points

# Load mesh from file
mesh = meshio.read("../data/LandscapeMap.ply")
print("Mesh loaded successfully")

vertices = mesh.points
print("Number of vertices:", len(vertices))

np.save("../data/landscape_mountains_full.npy", vertices)

points = vertices[::50]
fig = plot_3d_points(
    x=points[:, 0], y=points[:, 1], z=points[:, 2], color=points[:, 2], markersize=1
)
fig.update_layout(width=2000, height=1200)
fig.show()
