import numpy as np
import leidenalg as la

def find_comms(G, res=1, null_model="cm", weights=None, n_runs=1):

    if null_model == "cm":
        partition_type = la.RBConfigurationVertexPartition
    elif null_model == "er":
        partition_type = la.RBERVertexPartition

    partitions = np.zeros((G.vcount(), n_runs))

    rng = np.random.default_rng()
    for r in range(n_runs):
        # Maximize modularity: we need to set seed here to make sure consecutive 
        # calls of find_partition returns different community structure
        C = la.find_partition(
            G, partition_type, resolution_parameter=res, weights=weights, 
            seed=rng.integers(low=1, high=1e6)
        )
        partitions[:, r] = C.membership
    return partitions