import numpy as np

from models.graph_gen import get_graph_generate_fn
from Demo_Utils.PointCloud_Downsampling import downsample_by_average_voxel

def fetch_data(dataset, frame_idx, voxel_size, config):
    velo_points = dataset.get_velo_points(frame_idx) # Get the entire original point cloud
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx, config['downsample_by_voxel_size']) # Get only the points that are enclosed inside the image and apply downsample according to config
    downsampled_PointCloud, downsample_indices = downsample_by_average_voxel(cam_rgb_points, voxel_size) # Downsample the point cloud using average voxel method

    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])

    input_v = cam_rgb_points.attr # Generate Input vector

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    return velo_points, cam_rgb_points, downsampled_PointCloud, input_v, vertex_coord_list, keypoint_indices_list, edges_list

def fetch_data_and_labels(dataset, frame_idx, voxel_size, config):
    velo_points = dataset.get_velo_points(frame_idx) # Get the entire original point cloud
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx) # Get only the points that are enclosed inside the image and apply downsample according to config
    downsampled_PointCloud, downsample_indices = downsample_by_average_voxel(cam_rgb_points, voxel_size) # Downsample the point cloud using average voxel method

    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])

    input_v = cam_rgb_points.attr # Generate Input vector

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    box_label_list = dataset.get_label(frame_idx)

    ## ================== New code ==========================================
    last_layer_graph_level = \
        config['model_kwargs']['layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    if config['label_method'] == 'yaw':
        cls_labels, boxes_3d, valid_boxes, label_map_1 = \
            dataset.assign_classaware_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map_1 = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        (cls_labels, boxes_3d, valid_boxes, label_map_1) =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz,
            expend_factor=(1.0, 1.0, 1.0))
    cls_labels = cls_labels.astype(np.int32)

    ## Get cls_labels of the original point cloud =====================================
    (vertex_coord_list_full, keypoint_indices_list_full, edges_list_full) = \
        graph_generate_fn(
            cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
    last_layer_points_xyz_full = vertex_coord_list_full[last_layer_graph_level+1]
    if config['label_method'] == 'yaw':
        cls_labels_full, boxes_3d, valid_boxes, label_map_1 = \
            dataset.assign_classaware_label_to_points(box_label_list,
            last_layer_points_xyz_full,
            expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Car':
        cls_labels_full, boxes_3d, valid_boxes, label_map_1 = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz_full,
            expend_factor=(1.0, 1.0, 1.0))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        (cls_labels_full, boxes_3d, valid_boxes, label_map_1) =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz_full,
            expend_factor=(1.0, 1.0, 1.0))
    cls_labels_full = cls_labels_full.astype(np.int32)

    return cam_rgb_points, downsampled_PointCloud, last_layer_points_xyz, last_layer_points_xyz_full, cls_labels_full, cls_labels