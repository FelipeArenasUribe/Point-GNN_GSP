import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
import open3d
import random as rnd
from scipy.io import savemat
from tqdm import tqdm
import pandas as pd

import pdb

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
    for frame_idx in tqdm(range(0, dataset.num_files)):
    #for frame_idx in tqdm(range(0, 30)):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        # read by default 1st sheet of an excel file
        dataframe1 = pd.read_excel('index_pc_'+str(frame_idx)+'.xls')
        index = pd.DataFrame(dataframe1).to_numpy()
        
        pdb.set_trace()
