import numpy as np
import open3d as o3d
import sys

def display_inlier_outlier(cloud, ind):
    # http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

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
pcd_og = o3d.geometry.PointCloud()
pcd_og.points = o3d.utility.Vector3dVector(data["P"].copy())
pcd_og.colors = o3d.utility.Vector3dVector(data["pixels"].copy())
pcd_og.transform(flip)

pcd_og = pcd_og.farthest_point_down_sample(30000) # downsample
# o3d.visualization.draw_geometries([pcd_og])

# Estimate normals
pcd_og.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=120))
pcd_og.orient_normals_consistent_tangent_plane(200)
# o3d.visualization.draw_geometries([pcd_og], point_show_normal=True)

# Outlier removal
pcd, ind = pcd_og.remove_radius_outlier(nb_points=4, radius=0.5)
# display_inlier_outlier(pcd_og, ind)

# Ball reconstruction
radii = np.array([0.005, 0.01, 0.02, 0.04])*25
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries_with_key_callbacks([rec_mesh], key_to_callback)
# mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(rec_mesh)
# o3d.io.write_triangle_mesh("out.stl", rec_mesh)
# o3d.visualization.draw_geometries_with_key_callbacks([mesh], key_to_callback)

# Poisson 3D reconstruction
# http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
o3d.visualization.draw_geometries_with_key_callbacks([mesh], key_to_callback)
# mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
# o3d.io.write_triangle_mesh("out.stl", mesh)