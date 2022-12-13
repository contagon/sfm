import numpy as np
import open3d as o3d
import sys

def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

key_to_callback = {}
key_to_callback[ord("K")] = change_background_to_black

data = np.load(sys.argv[1])

# Flips everything so it's easily viewable by default
flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

# Make point cloud from our points
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(data["P"])
pc.colors = o3d.utility.Vector3dVector(data["pixels"])
pc.transform(flip)

geo = [pc]

# Insert frames
for i, T in enumerate(data["T"]):
    # Make new coordinate frame
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.transform(np.linalg.inv(T))
    mesh.transform(flip)

    geo.append(mesh)

o3d.visualization.draw_geometries_with_key_callbacks(geo, key_to_callback)
