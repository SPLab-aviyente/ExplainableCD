import numpy as np

def participation_coeff(C):
    is_weighted = C.graph.is_weighted()
    weights = "weight" if is_weighted else None

    d = C.graph.strength(weights=weights)
    
    coeffs = np.ones(len(d))
    for node in range(len(d)):
        for comm in C:
            if is_weighted: 
                coeffs[node] -= (sum(C.graph.es.select(_between=([node], comm))[weights])/d[node])**2
            else: 
                coeffs[node] -= (len(C.graph.es.select(_between=([node], comm)))/d[node])**2

    return coeffs

def module_degree_zscore(C):
    weights = "weight" if C.graph.is_weighted() else None
    
    zscores = np.zeros(C.graph.vcount())
    for comm_id, comm_nodes in enumerate(C):
        comm_strengths = C.subgraph(comm_id).strength(weights=weights)
        zscores[comm_nodes] = (comm_strengths - np.mean(comm_strengths))/np.std(comm_strengths)
    
    return zscores