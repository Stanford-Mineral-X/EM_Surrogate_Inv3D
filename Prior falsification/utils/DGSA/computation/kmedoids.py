"""
Perform k-medoid clustering on a distance matrix.
Based upon the work of Celine Scheidt.
"""

from numpy.typing import NDArray
import numpy as np

def kmedoids(
        distance_matrix: NDArray[np.float64],
        n_clusters: int = 3,
        n_rep: int = 5,
        max_iterations: int = 50,
        ) -> dict:
    """
    Perform k-medoids clustering on a distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Array of distances between samples 
          
    n_clusters : int, default = 3
        Number of clusters to construct

    n_rep : int, default = 5
        Number of k-medoid runs to perform. The best cluster configuration is returned
        
    max_iterations : int, default = 50
        Maximum number of iterations to perform
    
    Returns
    -------
    Results of the kmedoids clustering: dict
        - cluster_assignments : np.ndarray of shape (n_samples,)
            Cluster index for each sample.
        - medoid_indices : np.ndarray of shape (n_clusters,)
            Indices of cluster medoids.
        - n_points : np.ndarray of shape (n_clusters,)
            Number of points belonging to each cluster.
    """
    
    n_points = distance_matrix.shape[0]
    min_dist_best = np.inf
    
    # Variables to store best configuration
    label_best = None
    current_medoids_best = None

    # perform medoids clustering
    for iter in range(n_rep): 
        # 1. Initialize: randomly select n_clusters points as the medoids
        init_medoids = np.random.choice(n_points, size=n_clusters, replace=False)
        
        # 2. Associate each data point to the closest medoid
        min_dist_init = distance_matrix[init_medoids].min(axis=0)
        label = distance_matrix[init_medoids].argmin(axis=0)
        
        current_medoids = init_medoids.copy()
        min_dist_current = min_dist_init.copy()
        
        label_prev = np.full(n_points, np.nan)
        n_iter = 0
        
        # While cluster configuration is changing and maxIteration not reached
        while not np.array_equal(label, label_prev) and n_iter < max_iterations:
            label_prev = label.copy()
            
            # 3. For each medoid m
            for m in range(n_clusters):
                # Get non-medoid points
                no_medoid = np.setdiff1d(np.arange(n_points), current_medoids)
                new_medoids = current_medoids.copy()
                
                # For each non-medoid data point o
                for o in no_medoid:
                    # Swap m and o and compute the cost of the configuration
                    new_medoids[m] = o
                    min_dist = distance_matrix[new_medoids].min(axis=0)
                    label_temp = distance_matrix[new_medoids].argmin(axis=0)
                    cost = min_dist.sum() - min_dist_current.sum()
                    
                    # 4. Select the configuration with the lowest cost
                    if cost < 0:
                        current_medoids[m] = o
                        min_dist_current = distance_matrix[current_medoids].min(axis=0)
                        label = distance_matrix[current_medoids].argmin(axis=0)
            
            n_iter += 1
        
        # Sort medoids for consistency
        current_medoids.sort()
        min_dist = distance_matrix[current_medoids].min(axis=0)
        label = distance_matrix[current_medoids].argmin(axis=0)
        
        # Return the best clustering configuration among the n_rep tested
        if min_dist.sum() < min_dist_best:
            print(f'minDist {min_dist.sum():.2f}, iter {iter+1}')
            min_dist_best = min_dist.sum()
            label_best = label
            current_medoids_best = current_medoids

    # Once the medoids are defined, store the outputs
    weights = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        weights[i] = np.sum(label_best == i)

    # Check if there are clusters with no points
    cluster_contains_zero = np.any(weights == 0)
    if cluster_contains_zero:
        print("There are cluster(s) with no points! Clustering is NOT valid!!!")
        print("Check the data and adjust clustering parameters (e.g. n_clusters)!")
    else:
        print("Kmedoids clustering completed")

    return {
        'cluster_assignments': label_best,  # Cluster assignments
        'medoid_indices': current_medoids_best,  # Medoid indices
        'n_points': weights  # Number of points in each cluster
    } 