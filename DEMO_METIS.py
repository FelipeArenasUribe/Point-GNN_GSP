import numpy as np
import open3d
import networkx as nx
#import pymetis as metis
import metis
import random as rnd
from sklearn.neighbors import kneighbors_graph

from dataset.kitti_dataset import KittiDataset
from Demo_Utils import PointCloud_Visualization
from Demo_Utils.Data_Fetcher import fetch_data

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

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        print('Building graph...')
        print('')
        Adj = kneighbors_graph(X = calibrated_PC.xyz, n_neighbors = 128, mode = 'connectivity', n_jobs = 10).toarray()

        partitions = 4 # Set number of partitions

        print('METIS...')
        #n_cuts, membership = metis.part_graph(partitions, adjacency=Adj)
        G = nx.from_numpy_matrix(Adj)
        G = metis.networkx_to_metis(G)
        edgecuts, parts = metis.part_graph(G, partitions)
        
        new_node_indices = []
        for i in range(0,partitions):
            new_node_indices.append(np.argwhere(np.array(parts) == i).ravel())

        new_node_xyz = []
        for i in range(0, partitions):
            new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
            for j in range(0, len(new_node_indices[i])):
                new_node_xyz[i][j] = calibrated_PC.xyz[new_node_indices[i][j]]

        pcd = []
        for i in range(0, partitions):
            pcd.append(open3d.PointCloud())
            pcd[i].points = open3d.Vector3dVector(new_node_xyz[i])
            pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

        #for i in range(0, partitions):
        #    PointCloud_Visualization.Visualize_Point_Cloud([pcd[i]])

        PointCloud_Visualization.Visualize_Point_Cloud(pcd)
        #PointCloud_Visualization.Visualize_Partitioned_Graph(nodes, edges, pcd)
        

        input('Press enter...')