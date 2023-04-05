from scipy.io import loadmat
import os
import numpy as np
from tqdm import tqdm
from Demo_Utils.Graph_Signal_Processing import Get_Adjacency_Matrix, Get_Edges_from_Adj
from sklearn.neighbors import radius_neighbors_graph
import random as rnd
import itertools

from Demo_Utils import PointCloud_Visualization

sample_directory = '/home/felipearur/Documents/Point-GNN_GSP/Sampling_Data/PointCloud_Samples'
partition_directory = '/home/felipearur/Documents/Point-GNN_GSP/Sampling_Data/PointCloud_Partitions'
pc_directory = partition_directory
n_clusters = 4

Edge_lists = []
PC_partitions = []
Sample_groups = []

files = 30

print('-------------------------- FETCHING DATA --------------------------')
for frame_idx in tqdm(range(0,files)):
    Edge_lists.append([])
    Sample_groups.append([])
    PC_partitions.append([])

    for i in range(0, n_clusters):
        sample_file = 'PointCloud_{}_Partition_{}_sample.mat'.format(frame_idx,i)
        f = os.path.join(sample_directory, sample_file)
        if os.path.isfile(f):
            dic = loadmat(f)
            Sample_groups[frame_idx].append(dic['PointCloud_{}_Partition_{}_sample'.format(frame_idx,i)])
        
        partition_file = 'PointCloud_{}_Partition_{}.mat'.format(frame_idx,i)
        f = os.path.join(partition_directory, partition_file)
        if os.path.isfile(f):
            dic = loadmat(f)
            Edge_lists[frame_idx].append(dic['PointCloud_{}_Partition_{}'.format(frame_idx,i)])

        pc_file = 'PointCloud_{}_XYZ_{}.mat'.format(frame_idx,i)
        f = os.path.join(pc_directory, pc_file)
        if os.path.isfile(f):
            dic = loadmat(f)
            PC_partitions[frame_idx].append(dic['PointCloud_{}_XYZ_{}'.format(frame_idx,i)])

'''
print('-------------------------- BUILDING ADJACENCY MATRICES --------------------------')
Adjacency_Matrix = []
for frame_idx in tqdm(range(0, 30)):
    Adjacency_Matrix.append([])

    for i in range(0, n_clusters):
        Adjacency_Matrix[frame_idx].append(Get_Adjacency_Matrix(PC_partitions[frame_idx][i], Edge_lists[frame_idx][i]))
'''

print('-------------------------- SAMPLING POINTS --------------------------')
Sampled_PC = []
Residual_PC = []
Sampled_Edges = []
Sampled_Adjacency = []

for frame_idx in tqdm(range(0,files)):
    Sampled_PC.append([])
    Residual_PC.append([])
    Sampled_Edges.append([])
    Sampled_Adjacency.append([])

    for i in range(0, n_clusters):
        Sampled_PC[frame_idx].append([])
        Residual_PC[frame_idx].append([])
        for j in range(0, len(PC_partitions[frame_idx][i])):
            try:
                if Sample_groups[frame_idx][i][j] == 0:
                #if rnd.randint(0,1) == 1:
                    Sampled_PC[frame_idx][i].append(j)
                else:
                    Residual_PC[frame_idx][i].append(j)
            except:
                Residual_PC[frame_idx][i].append(j)

        Sampled_PC[frame_idx][i] = np.delete(PC_partitions[frame_idx][i], Sampled_PC[frame_idx][i], 0)
        Residual_PC[frame_idx][i] = np.delete(PC_partitions[frame_idx][i], Residual_PC[frame_idx][i], 0)

        Sampled_Adjacency[frame_idx].append(radius_neighbors_graph(X = Sampled_PC[frame_idx][i], radius = 4.0, mode = 'connectivity', n_jobs = 10).toarray())
        Sampled_Edges[frame_idx].append(Get_Edges_from_Adj(Sampled_Adjacency[frame_idx][i]))

for frame_idx in range(0,files):
    print()
    print('Original point cloud size: ',(len(PC_partitions[frame_idx][0])+len(PC_partitions[frame_idx][1])+len(PC_partitions[frame_idx][2])+len(PC_partitions[frame_idx][3])))
    print('Sampled point cloud size: ',(len(Sampled_PC[frame_idx][0])+len(Sampled_PC[frame_idx][1])+len(Sampled_PC[frame_idx][2])+len(Sampled_PC[frame_idx][3])))
    print('Original graph number of edges: ',(len(Edge_lists[frame_idx][0])+len(Edge_lists[frame_idx][1])+len(Edge_lists[frame_idx][2])+len(Edge_lists[frame_idx][3])))
    print('Sampled graph number of edges: ',(len(Sampled_Edges[frame_idx][0])+len(Sampled_Edges[frame_idx][1])+len(Sampled_Edges[frame_idx][2])+len(Sampled_Edges[frame_idx][3])))
    
    Total_Sampled = list(itertools.chain.from_iterable(Sampled_PC[frame_idx]))
    Total_Residual = list(itertools.chain.from_iterable(Residual_PC[frame_idx]))

    Total_Sampled_Adjacency = radius_neighbors_graph(X = Total_Sampled, radius = 4.0, mode = 'connectivity', n_jobs = 10).toarray()
    Total_Sampled_Edges = Get_Edges_from_Adj(Total_Sampled_Adjacency)

    Points = [Total_Residual, Total_Sampled]
    
    print()
    print('------------------Void and Cluster Sampling visualization------------------')
    print()
    PointCloud_Visualization.Visualize_VaC_Point_Clouds(Points)
    
    print()
    print('------------------Original Partitioned Point Cloud visualization------------------')
    print()
    PointCloud_Visualization.Visualize_Point_Clouds(PC_partitions[frame_idx])

    print()
    print('------------------Downsampled Point Cloud visualization------------------')
    print()
    PointCloud_Visualization.Visualize_Point_Clouds(Sampled_PC[frame_idx])
    
    print()
    print('------------------Original partitioned graphs visualization------------------')
    print()
    PointCloud_Visualization.Visualize_Graphs(PC_partitions[frame_idx], Edge_lists[frame_idx])

    print()
    print('------------------Downsampled partitioned graphs visualization------------------')
    print()
    PointCloud_Visualization.Visualize_Graphs(Sampled_PC[frame_idx], Sampled_Edges[frame_idx])
    
    print()
    print('------------------Downsampled graph visualization------------------')
    print()
    PointCloud_Visualization.Visualize_Graph(Total_Sampled, Total_Sampled_Edges)

    input('Press enter to continue...')
    