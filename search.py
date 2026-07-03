import pandas as pd
import numpy as np
from core.mapping import poincare_distance

def find_geodesic_neighbors(query_coord, df_knowledge_base, k=5):
    """
    Performs nearest-neighbor search using strictly Hyperbolic (Geodesic) distance.
    This acts as the query engine for the Continuous Clinical Knowledge Base.
    """
    distances = []
    
    for idx, row in df_knowledge_base.iterrows():
        hist_coord = np.array([row['poincare_x'], row['poincare_y'], row['poincare_z']])
        
        # Calculate topological distance using the non-Euclidean metric
        dist = poincare_distance(query_coord, hist_coord)
        distances.append(dist)
        
    df_knowledge_base['d_P_distance'] = distances
    
    # Sort by closest geodesic distance (ascending)
    neighbors = df_knowledge_base.sort_values(by='d_P_distance', ascending=True).head(k)
    return neighbors
