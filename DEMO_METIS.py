import numpy as np
import open3d
import networkx as nx
import pymetis as metis
import random as rnd

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

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        original_PC, calibrated_PC, downsampled_PC, input_v, vertex_coord_list, keypoint_indices_list, edges_list = fetch_data(frame_idx, voxel_size, level_configs)

        nodes = vertex_coord_list[0]
        edges = edges_list[0]
        keypoint_indices = keypoint_indices_list[0]

        G = nx.Graph() #Create a Graph Networkx object
        for i in range(0,len(edges)):
            G.add_edge(edges[i][0], edges[i][1])

        A = nx.to_numpy_array(G) #Get Adjacency matrix from G as a np.array

        partitions = 4 # Set number of partitions
        
        n_cuts, membership = metis.part_graph(partitions, adjacency=A)

        new_node_indices = []

        for i in range(0,partitions):
            new_node_indices.append(np.argwhere(np.array(membership) == i).ravel())

        #nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel()
        #nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel()

        new_node_xyz = []

        for i in range(0, partitions):
            new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
            for j in range(0, len(new_node_indices[i])):
                new_node_xyz[i][j] = nodes[new_node_indices[i][j]]

        #new_nodes_0 = np.zeros((len(nodes_part_0), 3))
        #new_nodes_1 = np.zeros((len(nodes_part_1), 3))

        #for i in range(0,len(nodes_part_0)):
        #    new_nodes_0[i] = nodes[nodes_part_0[i]]

        #for i in range(0,len(nodes_part_1)):
        #    new_nodes_1[i] = nodes[nodes_part_1[i]]

        pcd = []

        for i in range(0, partitions):
            pcd.append(open3d.PointCloud())
            pcd[i].points = open3d.Vector3dVector(new_node_xyz[i])
            pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

        #pcd_0 = open3d.PointCloud()
        #pcd_0.points = open3d.Vector3dVector(new_nodes_0)
        #pcd_0.paint_uniform_color([0.5, 1, 0.8])

        #pcd_1 = open3d.PointCloud()
        #pcd_1.points = open3d.Vector3dVector(new_nodes_1)
        #pcd_1.paint_uniform_color([1, 0.5, 0.8])

        #PointCloud_Visualization.Visualize_Point_Cloud([pcd_0, pcd_1])
        PointCloud_Visualization.Visualize_Point_Cloud(pcd)

        holder = False
        while holder==False:
            if input('Enter q to continue: ') == 'q':
                holder=True