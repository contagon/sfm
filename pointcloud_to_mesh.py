import numpy as np
import open3d as o3d
import sys

def display_inlier_outlier(cloud, ind):
    # http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud,],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

data = np.load(sys.argv[1])

# Flips everything so it's easily viewable by default
flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

# Make point cloud from our points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data["P"])
pcd.colors = o3d.utility.Vector3dVector(data["pixels"])
# pcd.transform(flip)

# Statistical Removal
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
# display_inlier_outlier(pcd, ind)
# quit()

# Radius removal
pcd, ind = pcd.remove_radius_outlier(nb_points=4, radius=0.1)
# display_inlier_outlier(pcd, ind)

# Convert to mesh!
# http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html

# Estimate normals
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(25)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# # Radii method
# radii = [0.005, 0.01, 0.02, 0.04]
# radii = np.array([0.1*2**i for i in range(4)])
# print(radii)
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([rec_mesh])

# Poisson 3D reconstruction
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
# Remove low density areas?
# densities = np.asarray(densities)
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)
o3d.visualization.draw_geometries([mesh])
