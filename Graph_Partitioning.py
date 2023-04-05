import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
import open3d
import random as rnd
from scipy.io import savemat
from tqdm import tqdm


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
    #for frame_idx in range(0, dataset.num_files):
    for frame_idx in tqdm(range(0, 30)):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        n_clusters = 4 # Number of clusters to partition
        n_components = n_clusters # Number of eigenvectors to use for the spectral embeding
        data = calibrated_PC.xyz

        clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors', n_neighbors = 256).fit(data)

        new_node_indices = []
        for i in range(0,n_clusters):
            new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel().tolist())

        new_node_xyz = []
        for i in range(0, n_clusters):
            new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
            for j in range(0, len(new_node_indices[i])):
                new_node_xyz[i][j] = data[new_node_indices[i][j]]

        pcd = []
        for i in range(0, n_clusters):
            pcd.append(open3d.PointCloud())
            pcd[i].points = open3d.Vector3dVector(new_node_xyz[i])
            pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

        '''
        print()
        print('------------------Partitioned Point Cloud visualization------------------')
        print()
        PointCloud_Visualization.Visualize_Point_Cloud(pcd)
        '''

        Aff = clustering.affinity_matrix_.toarray()        
        new_edges = np.transpose(np.nonzero(Aff))

        new_A = []
        cluster_edges = []
        for i in range(0,n_clusters):
            #new_A.append(radius_neighbors_graph(X = new_node_xyz[i], radius = 1.0, mode = 'connectivity', n_jobs = 10).toarray())
            new_A.append(kneighbors_graph(X = new_node_xyz[i], n_neighbors = 5, mode = 'connectivity', n_jobs = 10).toarray())
            cluster_edges.append(np.transpose(np.nonzero(new_A[i])))
            #PointCloud_Visualization.Visualize_Graph(new_node_xyz[i], cluster_edges[i])

        '''
        print()
        print('------------------Spectral Clustering Generated Graph visualization------------------')
        print()
        #PointCloud_Visualization.Visualize_Graph(data, new_edges)
        '''
        
        for i in range(0,n_clusters):
            file_name = 'PointCloud_{}_XYZ_{}.mat'.format(frame_idx,i)
            print('Saving: ',file_name)
            mdic = {'PointCloud_{}_XYZ_{}'.format(frame_idx,i): new_node_xyz[i]}
            savemat(file_name, mdic)

            file_name = 'PointCloud_{}_Partition_{}.mat'.format(frame_idx,i)
            print('Saving: ',file_name)
            mdic = {'PointCloud_{}_Partition_{}'.format(frame_idx,i): cluster_edges[i]}
            savemat(file_name, mdic)
        
        #holder = False
        #while holder==False:
        #    if input('Enter q to continue: ') == 'q':
        #        holder=True