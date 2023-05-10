import numpy as np
import igraph as ig

def association_matrix(partitions):
    n_nodes, n_partition = partitions.shape

    A = np.zeros((n_nodes, n_nodes))
    for p in range(n_partition):
        A += (partitions[:, p][..., None] == partitions[:, p]).astype(float)/n_partition

    return A

def find_comms(partitions, alg, n_calls=1):
    
    # Number of nodes and community structures
    n_nodes, n_partition = partitions.shape

    # Original association matrix
    A_org = association_matrix(partitions)

    # When partitions are the same across runs, association matrix includes only two values
    # We can use this fact to stop the algorithm. 
    if len(np.unique(A_org)) == 2 or n_calls>10:
        return partitions[:, 0]

    # Randomize partitions and get a randomized association matrix
    for p in range(n_partition):
        np.random.shuffle(partitions[:, p])

    A_rnd = association_matrix(partitions)

    # Expected number of times a pair of nodes is assigned to the same community in random partition
    threshold = np.max(A_rnd[np.triu_indices(n_nodes, k=1)])

    A_org_c = A_org.copy()
    A_org[A_org<threshold] = 0
    A_org[np.diag_indices(n_nodes)] = 0

    # Generate a network from association matrix
    G = ig.Graph().Weighted_Adjacency(A_org, mode="undirected")
    
    # After thresholding some nodes may be disconnected, connect them to neighbors with high weights  
    degrees = np.count_nonzero(A_org, axis=1)
    for i in np.where(degrees==0)[0]:
        for j in np.where(A_org_c[i, :] >= np.max(A_org_c[i, :])-1e-6)[0]:
            G.add_edge(i, j, weight=A_org_c[i, j])

    return find_comms(alg(G), alg, n_calls=n_calls+1)