import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
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
    for frame_idx in range(0, 30):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        nodes = nodes_coord_list[1]
        edges = edges_list[1]
        keypoint_indices = keypoint_indices_list[1]
        '''
        print()
        print('------------------Point-GNN Generated Graph visualization------------------')
        print()
        PointCloud_Visualization.Visualize_Graph(nodes, edges)
        '''
        '''
        A = Get_Adjacency_Matrix(nodes, edges)
        Aff = Get_Affinity_Matrix(nodes, edges, A, 1)
        D = np.diag(A.sum(axis=1)) # Degree matrix
        L = D-A # Graph Laplacian matrix
        
        
        # find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(L)

        # Uncomment to visualize Graph Spectrum
        
        ranks = range(0,len(eigenvalues))
        
        print('Spectral graph: ', eigenvalues[0])
        print('Fiedler value: ',eigenvalues[1])
        
        plt.scatter(ranks, eigenvalues)
        plt.xlabel('ranks')
        plt.ylabel('eigenvalues')
        '''
        '''Spectral Clustering with only nodes as input:
        
            This means that a graph will be generated using the affinity parameter.
        '''
        n_clusters = 4 # Number of clusters to partition
        n_components = n_clusters # Number of eigenvectors to use for the spectral embeding
        #data = calibrated_PC.xyz
        data = nodes
        
        clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors', n_neighbors = math.floor(math.sqrt(len(data)))).fit(data)

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

        #for i in range(0, n_clusters):
        #    PointCloud_Visualization.Visualize_Point_Cloud([pcd[i]])

        print()
        print('------------------Partitioned Point Cloud visualization------------------')
        print()
        PointCloud_Visualization.Visualize_Point_Cloud(pcd)
        
        Aff = clustering.affinity_matrix_.toarray()        
        new_edges = np.transpose(np.nonzero(Aff))

        print()
        print('------------------Spectral Clustering Generated Graph visualization------------------')
        print()
        PointCloud_Visualization.Visualize_Partitioned_Graph(nodes= data, edges= new_edges, pcds=pcd)

        cluster_edges = []
        for i in range(0, n_clusters):
            A = kneighbors_graph(X = new_node_xyz[i], n_neighbors = 5, mode = 'connectivity', n_jobs = 10).toarray()
            cluster_edges.append(Get_Edges_from_Adj(A))
            print('Partitioned graph')
            PointCloud_Visualization.Visualize_Graph(new_node_xyz[i], cluster_edges[i])

        PointCloud_Visualization.Visualize_Graphs(nodes = new_node_xyz, edges=cluster_edges)
        
        input('Press enter to continue...')