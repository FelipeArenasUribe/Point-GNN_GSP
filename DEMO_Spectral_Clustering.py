import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import open3d
import random as rnd
from scipy.io import savemat
import networkx as nx


from dataset.kitti_dataset import KittiDataset
from Demo_Utils import PointCloud_Visualization
from Demo_Utils.Data_Fetcher import fetch_data
from models.graph_gen import get_graph_generate_fn

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

def Get_Adjacency_Matrix(nodes,edges):
    A = np.zeros((len(nodes), len(nodes)))

    for i in range(0, len(edges)):
        A[edges[i][0]][edges[i][1]] = 1
    
    return A

def Get_Cluster_Adjacency_Matrix(nodes, edges):
    A = np.zeros((len(nodes), len(nodes)))

    for i in range(0, len(edges)):
        A[nodes.index(edges[i][0])][nodes.index(edges[i][1])] = 1

    return A


def Get_Affinity_Matrix(nodes,edges, A, gamma):
    Aff = np.zeros((len(nodes), len(nodes)))

    for i in range(0, len(edges)):
        if A[edges[i][0]][edges[i][1]] == 1:
            Aff[edges[i][0]][edges[i][1]] = np.exp(-gamma * np.linalg.norm(nodes[edges[i][0]]-nodes[edges[i][1]]) ** 2)

    return Aff

if __name__ == "__main__":
    for frame_idx in range(0, dataset.num_files):
    #for frame_idx in range(0, 30):
        original_PC, calibrated_PC, downsampled_PC, input_v, nodes_coord_list, keypoint_indices_list, edges_list = fetch_data(dataset, frame_idx, voxel_size, config)

        print(input_v)

        nodes = nodes_coord_list[1]
        edges = edges_list[1]
        keypoint_indices = keypoint_indices_list[1]

        print()
        print('------------------Point-GNN Generated Graph visualization------------------')
        print()
        PointCloud_Visualization.Visualize_Graph(nodes, edges)

        A = Get_Adjacency_Matrix(nodes, edges)
        #Aff = Get_Affinity_Matrix(nodes, edges, A, 1)
        #D = np.diag(A.sum(axis=1)) # Degree matrix
        #L = D-A # Graph Laplacian matrix
        
        '''
        # find the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(L)

        # Uncomment to visualize Graph Spectrum
        
        ranks = range(0,len(eigenvalues))
        
        print('Spectral graph: ', eigenvalues[0])
        print('Fiedler value: ',eigenvalues[1])
        
        plt.scatter(ranks, eigenvalues)
        plt.xlabel('ranks')
        plt.ylabel('eigenvalues')
        plt.title('Graph Spectrum')
        plt.show()
        '''

        '''Spectral Clustering with only nodes as input:
        
            This means that a graph will be generated using the affinity parameter.
        '''
        n_clusters = 4 # Number of clusters to partition
        n_components = n_clusters # Number of eigenvectors to use for the spectral embeding
        clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors').fit(nodes)

        ''' Spectral Clustering with Affinity Matrix Input: '''
        '''
        n_clusters = 4 # Number of clusters to partition
        n_components = n_clusters # Number of eigenvectors to use for the spectral embeding

        clustering = SpectralClustering(n_clusters, n_components = n_components, affinity='precomputed', assign_labels='discretize')
        clustering.fit_predict(A)
        '''

        new_node_indices = []
        for i in range(0,n_clusters):
            new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel().tolist())

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
        PointCloud_Visualization.Visualize_Graph(nodes, new_edges)

        cluster_edges = []
        for i in range(0, n_clusters):
            cluster_edges.append([])
            for j in range(0,len(new_edges)):
                if (new_edges[j][0] in new_node_indices[i]) and (new_edges[j][1] in new_node_indices[i]):
                    cluster_edges[i].append(new_edges[j].tolist())
            #PointCloud_Visualization.Visualize_Graph(nodes, cluster_edges[i])
        
        
        new_A = []
        for i in range(0,n_clusters):
            new_A.append(Get_Cluster_Adjacency_Matrix(new_node_indices[i],cluster_edges[i]))

            '''
            file_name = 'PointCloud_{}_Partition_{}.mat'.format(frame_idx,i)
            print('Saving: ',file_name)
            mdic = {'PointCloud_{}_Partition_{}'.format(frame_idx,i): new_A[i]}
            savemat(file_name, mdic)
            '''
        
        #holder = False
        #while holder==False:
        #    if input('Enter q to continue: ') == 'q':
        #        holder=True