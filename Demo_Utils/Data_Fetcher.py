import numpy as np

from models.graph_gen import get_graph_generate_fn
from Demo_Utils.PointCloud_Downsampling import downsample_by_average_voxel

def fetch_data(dataset, frame_idx, voxel_size, config):
    velo_points = dataset.get_velo_points(frame_idx) # Get the entire original point cloud
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx, config['downsample_by_voxel_size']) # Get only the points that are enclosed inside the image and apply downsample according to config
    downsampled_PointCloud = downsample_by_average_voxel(cam_rgb_points, voxel_size) # Downsample the point cloud using average voxel method
    
    graph_generate_fn= get_graph_generate_fn(config["graph_gen_method"])
    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(cam_rgb_points.xyz, **config["graph_gen_kwargs"])

    input_v = cam_rgb_points.attr # Generate Input vector

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    return velo_points, cam_rgb_points, downsampled_PointCloud, input_v, vertex_coord_list, keypoint_indices_list, edges_list