import numpy as np

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

def Get_Edges_from_Adj(A):
    Edges = np.transpose(np.nonzero(A))
    return Edges