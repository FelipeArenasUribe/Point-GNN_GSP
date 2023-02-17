import numpy as np
import open3d

import sys
from os.path import join
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from dataset.kitti_dataset import KittiDataset, Points
from Demo_Utils import Graph_generation
from Demo_Utils import PointCloud_Visualization
from Demo_Utils.PointCloud_Downsampling import downsample_by_average_voxel

# Create dataset object for easier data manipulation
dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

voxel_size = 0.8

level_configs = [
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 1.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 0,
                "graph_scale": 1
            },
            {
                "graph_gen_kwargs": {
                    "num_neighbors": 256,
                    "radius": 4.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 1,
                "graph_scale": 1
            }
        ]

def fetch_data(frame_idx, voxel_size, level_configs):
    velo_points = dataset.get_velo_points(frame_idx) # Get the entire original point cloud
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx) # Get only the points that are enclosed inside the image
    downsampled_PointCloud = downsample_by_average_voxel(cam_rgb_points, voxel_size) # Downsample the point cloud using average voxel method

    #box_label_list = dataset.get_label(frame_idx)

    '''Apply graph generation functions'''

    #ML_downsampled_list = Graph_generation.multi_layer_downsampling(downsampled_PointCloud.xyz, voxel_size, levels=[2], add_rnd3d=False)
    #vertex_coord_list, keypoint_indices_list = Graph_generation.multi_layer_downsampling_select(downsampled_PointCloud.xyz, voxel_size, levels=[2], add_rnd3d=False)

    vertex_coord_list, keypoint_indices_list, edges_list = Graph_generation.gen_multi_level_local_graph_v3(downsampled_PointCloud.xyz, voxel_size, level_configs, add_rnd3d=False, downsample_method='center')
    
    input_v = downsampled_PointCloud.attr # Generate Input vector

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    return velo_points, cam_rgb_points, downsampled_PointCloud, input_v, vertex_coord_list, keypoint_indices_list, edges_list
    
    return velo_points, cam_rgb_points, downsampled_PointCloud, vertex_coord_list, keypoint_indices_list, edges_list

def show_graph(src_points, des_points, edges):
    """
    :param src_points: [N, 3] src_points.
    :param des_points: [N, 3] des_points.
    :param edges: [M, 2],M pairs of connections src_points[edges[0]] -> des_points[edges[1]]
    :return: None
    """
    points = np.concatenate([src_points, des_points])
    edges[:, 1] += src_points.shape[0]
    line_set = open3d.LineSet()
    line_set.points = open3d.Vector3dVector(points)
    line_set.lines = open3d.Vector2iVector(edges)

    # add color if wanted
    line_set.colors = open3d.Vector3dVector(np.zeros((edges.shape[0], 3), dtype=np.float32))
    
    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([line_set])

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        original_PC, calibrated_PC, downsampled_PC, input_v, vertex_coord_list, keypoint_indices_list, edges_list = fetch_data(frame_idx, voxel_size, level_configs)

        nodes = vertex_coord_list[0]
        edges = edges_list[0]
        keypoint_indices = keypoint_indices_list[0]

        src_points = np.zeros([len(edges),3])
        for i in range(0,len(edges)):
            src_points[i][:] = nodes[edges[i][0]]

        des_points = np.zeros([len(edges),3])
        for i in range(0,len(edges)):
            des_points[i][:] = nodes[edges[i][0]]

        show_graph(src_points, des_points, edges)

        
        image_file=join(dataset._image_dir,dataset._file_list[frame_idx]+'.png')
        print('Image file path: ',image_file)
        print()

        print('------------------Original Point Cloud visualization------------------')
        print()
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(original_PC.xyz)
        #pcd.colors = open3d.Vector3dVector(colors)
        PointCloud_Visualization.Visualize_Point_Cloud([pcd])
        print()

        print('------------------Calibrated Point Cloud visualization------------------')
        print()
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(calibrated_PC.xyz)
        #pcd.colors = open3d.Vector3dVector(colors)
        PointCloud_Visualization.Visualize_Point_Cloud([pcd])
        print()

        print('------------------Downsampled Point Cloud visualization------------------')
        print()
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(downsampled_PC.xyz)
        #pcd.colors = open3d.Vector3dVector(colors)
        PointCloud_Visualization.Visualize_Point_Cloud([pcd])
        print()

        holder = False
        while holder==False:
            if input('Enter q to continue: ') == 'q':
                holder=True