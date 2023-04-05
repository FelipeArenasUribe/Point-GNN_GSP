import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import radius_neighbors_graph
import open3d
import random as rnd
import math

from dataset.kitti_dataset import KittiDataset
from Demo_Utils import PointCloud_Visualization
from Demo_Utils.Data_Fetcher import fetch_data
from Demo_Utils.Graph_Signal_Processing import Get_Adjacency_Matrix, Get_Affinity_Matrix, Get_Edges_from_Adj

# Create dataset object for easier data manipulation
dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

config = {
    "downsample_by_voxel_size": None,
    "graph_gen_kwargs": {
        "add_rnd3d": True,
        "base_voxel_size": 0.05,
        "downsample_method": "void_and_cluster",
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
                "graph_level": 0,
                "graph_scale": 1
            }
        ]
    },
    "graph_gen_method": "multi_level_local_graph_v3",
}

if __name__ == "__main__":
    #for frame_idx in range(0, dataset.num_files):
    voxel_size = 0.8
    for frame_idx in range(0, 30):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        print()
        print(nodes_coord_list)
        print(len(nodes_coord_list))
        for i in range(0, len(nodes_coord_list)):
            print(len(nodes_coord_list[i]))
        print()
        print(keypoint_indices_list)
        print(len(keypoint_indices_list))
        for i in range(0, len(keypoint_indices_list)):
            print(len(keypoint_indices_list[i]))
        print()
        print(edges_list)
        print(len(edges_list))
        for i in range(0, len(edges_list)):
            print(len(edges_list[i]))
        print()
        input('Press enter...')
        
        input('Press enter to continue...')