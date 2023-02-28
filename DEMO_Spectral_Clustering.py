import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import open3d
import random as rnd


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

        nodes = nodes_coord_list[1]
        edges = edges_list[1]
        keypoint_indices = keypoint_indices_list[1]

        print('------------------Generated Graph visualization------------------')
        print()
        #PointCloud_Visualization.Visualize_Graph(nodes, edges)
        
        G = nx.Graph() #Create a Graph Networkx object
        for i in range(0,len(edges)):
            G.add_edge(edges[i][0], edges[i][1])
            G.add_edge(edges[i][1], edges[i][0])
        A = nx.to_numpy_array(G) #Get Adjacency matrix from G as a np.array

        D = np.diag(A.sum(axis=1)) # Degree matrix
        L = D-A # Graph Laplacian matrix

        # find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(L)
        
        ranks = range(0,len(eigenvalues))
        '''
        print('Spectral graph: ', eigenvalues[0])
        print('Fiedler value: ',eigenvalues[1])
        
        plt.scatter(ranks, eigenvalues)
        plt.xlabel('ranks')
        plt.ylabel('eigenvalues')
        plt.title('Graph Spectrum')
        plt.show()
        '''

        n_clusters = 4

        clustering = SpectralClustering(n_clusters, assign_labels='discretize', random_state=0, affinity='nearest_neighbors').fit(nodes)

        new_node_indices = []
        for i in range(0,n_clusters):
            new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel())

        new_node_xyz = []
        for i in range(0, n_clusters):
            new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
            for j in range(0, len(new_node_indices[i])):
                new_node_xyz[i][j] = nodes[new_node_indices[i][j]]

        pcd = []
        for i in range(0, n_clusters):
            pcd.append(open3d.PointCloud())
            pcd[i].points = open3d.Vector3dVector(new_node_xyz[i])
            pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])

        #for i in range(0, partitions):
        #    PointCloud_Visualization.Visualize_Point_Cloud([pcd[i]])

        PointCloud_Visualization.Visualize_Point_Cloud(pcd)    
        
        holder = False
        while holder==False:
            if input('Enter q to continue: ') == 'q':
                holder=True