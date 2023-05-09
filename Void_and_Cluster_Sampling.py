import numpy as np
import itertools
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import radius_neighbors_graph
import random as rnd
from tqdm import tqdm

from collections import namedtuple
Points = namedtuple('Points', ['xyz', 'attr'])

#from dataset.kitti_dataset import KittiDataset
#from Demo_Utils import PointCloud_Visualization
#from Demo_Utils.Data_Fetcher import fetch_data
#from Demo_Utils.Graph_Signal_Processing import Get_Edges_from_Adj

def Void_and_Cluster(points_xyz, n_clusters=4, n_components=4):

    n_clusters = 4 # Number of clusters to partition
    n_components = n_clusters # Number of eigenvectors to use for the spectral embeding
    data = points_xyz

    # Create graph partitions using Spectral Clustering
    print('Spectral Clustering...')
    clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors', n_neighbors = 256).fit(data)
    
    # Create list of the indices for elements of each partition
    '''
    Example:
    new_node_indices = [[1 2 3]
                        [4 5 6]
                        [7 8 9]]

    '''
    new_node_indices = []
    for i in range(0,n_clusters):
        new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel().tolist())

    # Extract xyz coordinates for each partition using index list
    new_node_xyz = []
    for i in range(0, n_clusters):
        new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
        for j in range(0, len(new_node_indices[i])):
            new_node_xyz[i][j] = points_xyz[new_node_indices[i][j]]

    new_A = []
    cluster_edges = []
    for i in range(0,n_clusters):
        new_A.append(radius_neighbors_graph(X = new_node_xyz[i], radius = 4.0, mode = 'connectivity', n_jobs = 10).toarray())
        cluster_edges.append(np.transpose(np.nonzero(new_A[i])))
    

    # Implement Void and Cluster here:
    print('Voiding and Clustering...')
    Sampling_list = []
    for i in range(0, n_clusters):
        Sampling_list.append([])
        for j in tqdm(range(0, len(new_node_indices[i]))):
            Sampling_list[i].append(rnd.randint(0,9))


    # Sample points according to void and cluster list
    Sampled_PC = []
    sampled_xyz = []
    sampled_attr = []
    for i in range(0, n_clusters):
        Sampled_PC.append([])
        sampled_xyz.append([])
        sampled_attr.append([])
        for j in range(0, len(new_node_xyz[i])):
            try:
                if Sampling_list[i][j] != 0:
                    Sampled_PC[i].append(j)
                else:
                    pass
            except:
                pass

        sampled_xyz[i] = np.delete(new_node_xyz[i], Sampled_PC[i], 0).tolist()

    new_xyz = list(itertools.chain.from_iterable(sampled_xyz))
    new_xyz = np.array(new_xyz)
    new_attr = list(itertools.chain.from_iterable(sampled_attr))
    new_attr = np.array(new_attr)

    # Build lists

    vertex_coord_list = [points_xyz, new_xyz, new_xyz]

    # Fix list construction so that it looks like the output from other downsampling functions
    keypoint_indices_list = []
    keypoint_indices_list.append([])
    for i in tqdm(range(0, len(points_xyz))):
        if points_xyz[i] in new_xyz:
            keypoint_indices_list[0].append(i)
    keypoint_indices_list[0] = np.array(keypoint_indices_list[0])
    keypoint_indices_list.append(np.array(range(0,len(keypoint_indices_list[0]))))

    return vertex_coord_list, keypoint_indices_list

def Void_and_Cluster_Downsampling(points_xyz, n_clusters = 4, n_components = 4, n_neighbors = 128):
    data = points_xyz

    # Create graph partitions using Spectral Clustering
    print('Spectral Clustering...')
    clustering = SpectralClustering(n_clusters, n_components = n_components, assign_labels='discretize', random_state=0, affinity='nearest_neighbors', n_neighbors = n_neighbors).fit(data)
    
    # Create list of the indices for elements of each partition
    '''
    Example:
    new_node_indices = [[1 2 3]
                        [4 5 6]
                        [7 8 9]]

    '''
    new_node_indices = []
    for i in range(0,n_clusters):
        new_node_indices.append(np.argwhere(np.array(clustering.labels_) == i).ravel().tolist())

    # Extract xyz coordinates for each partition using index list
    new_node_xyz = []
    for i in range(0, n_clusters):
        new_node_xyz.append(np.zeros((len(new_node_indices[i]), 3)))
        for j in range(0, len(new_node_indices[i])):
            new_node_xyz[i][j] = points_xyz[new_node_indices[i][j]]

    new_A = []
    cluster_edges = []
    for i in range(0,n_clusters):
        new_A.append(radius_neighbors_graph(X = new_node_xyz[i], radius = 4.0, mode = 'connectivity', n_jobs = 10).toarray())
        cluster_edges.append(np.transpose(np.nonzero(new_A[i])))
    

    # Implement Void and Cluster here:
    print('Voiding and Clustering...')
    Sampling_list = []
    for i in range(0, n_clusters):
        Sampling_list.append([])
        for j in tqdm(range(0, len(new_node_indices[i]))):
            Sampling_list[i].append(rnd.randint(0,9))


    # Sample points according to void and cluster list
    Sampled_PC = []
    sampled_xyz = []
    sampled_attr = []
    for i in range(0, n_clusters):
        Sampled_PC.append([])
        sampled_xyz.append([])
        sampled_attr.append([])
        for j in range(0, len(new_node_xyz[i])):
            try:
                if Sampling_list[i][j] != 0:
                    Sampled_PC[i].append(j)
                else:
                    pass
            except:
                pass

        sampled_xyz[i] = np.delete(new_node_xyz[i], Sampled_PC[i], 0).tolist()

    new_xyz = list(itertools.chain.from_iterable(sampled_xyz))
    new_xyz = np.array(new_xyz)
    new_attr = list(itertools.chain.from_iterable(sampled_attr))
    new_attr = np.array(new_attr)

    # Build Point class
    Downsampled_points = Points(new_xyz, new_attr)

    return Downsampled_points
