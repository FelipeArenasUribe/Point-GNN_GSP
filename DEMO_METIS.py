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
from models.graph_gen import get_graph_generate_fn
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

config = {
    "downsample_by_voxel_size": None,
    "graph_gen_kwargs": {
        "add_rnd3d": True,
        "base_voxel_size": 0.8,
        "downsample_method": "random",
        "level_configs": [
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
    },
    "graph_gen_method": "multi_level_local_graph_v3",
}

def fetch_data(frame_idx, voxel_size, config):
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

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(frame_idx, voxel_size, config)

        nodes = nodes_coord_list[1]
        edges = edges_list[1]
        keypoint_indices = keypoint_indices_list[1]

        PointCloud_Visualization.Visualize_Graph(nodes, edges)
        
        G = nx.Graph() #Create a Graph Networkx object
        for i in range(0,len(edges)):
            G.add_edge(edges[i][0], edges[i][1])
            G.add_edge(edges[i][1], edges[i][0])
        A = nx.to_numpy_array(G) #Get Adjacency matrix from G as a np.array

        
        partitions = 3 # Set number of partitions
        n_cuts, membership = metis.part_graph(partitions, adjacency=A)

        new_node_indices = []
        for i in range(0,partitions):
            new_node_indices.append(np.argwhere(np.array(membership) == i).ravel())

        new_node_xyz = []
        for i in range(0, partitions):
            new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
            for j in range(0, len(new_node_indices[i])):
                new_node_xyz[i][j] = nodes[new_node_indices[i][j]]

        pcd = []
        for i in range(0, partitions):
            pcd.append(open3d.PointCloud())
            pcd[i].points = open3d.Vector3dVector(new_node_xyz[i])
            pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

        PointCloud_Visualization.Visualize_Point_Cloud(pcd)
        #PointCloud_Visualization.Visualize_Partitioned_Graph(nodes, edges, pcd)

        holder = False
        while holder==False:
            if input('Enter q to continue: ') == 'q':
                holder=True