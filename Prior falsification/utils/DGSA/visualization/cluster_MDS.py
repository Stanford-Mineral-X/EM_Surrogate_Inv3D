from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

def cluster_MDS(
    distance_matrix: NDArray[np.float64], 
    clustering: dict,
    stress_score_output_dimensions: int,
    fig_size: tuple = (8, 6),
    font_size: int = 12,
    font: str = None
    ) -> np.ndarray:
    """
    Perform multidimensional scaling (MDS) on the distance matrix to plot (1) clusters in 2D space and (2) stress scores across dimensions.

    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Array of distances between samples 
          
    clustering : dict
        Results from clustering analysis containing:
          - cluster_assignments : np.ndarray of shape (n_samples,)
              Cluster index for each sample.
          - medoid_indices : np.ndarray of shape (n_clusters,)
              Indices of cluster medoids.
          - n_points : np.ndarray of shape (n_clusters,)
              Number of points belonging to each cluster.

    stress_score_output_dimensions : int
        Maximum number of dimensions to compute stress scores for.
    
    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
              
    Returns:
    Plots of (1) clusters in 2D space and (2) stress scores across dimensions.
    """
    dimensions = list(range(1, stress_score_output_dimensions + 1))
    stress_scores = []
    mds_coordinates_2d = None # Will store the 2D MDS coordinates for scatter plot
    
    # run MDS
    for dim in dimensions:
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        mds_coordinates = mds.fit_transform(distance_matrix)
        if dim == 2:
            mds_coordinates_2d = mds_coordinates

        # calculate stress score (Kruskal's Stress-1 formula)
        dist_low_dim = pairwise_distances(mds_coordinates)
        stress = np.sqrt(np.sum((distance_matrix - dist_low_dim) ** 2) /
                         np.sum(distance_matrix ** 2))
        stress_scores.append(stress)
    
    # prepare DataFrame for plotting
    df = pd.DataFrame(mds_coordinates_2d, columns=['Dimension 1', 'Dimension 2'])
    df['cluster'] = clustering['cluster_assignments']

    # define colormap
    unique_clusters = np.sort(df['cluster'].unique())
    cmap = plt.get_cmap('tab10', len(unique_clusters))  # Safe for <=10 clusters

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # plot MDS for each cluster
    plt.figure(figsize = fig_size)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['Dimension 1'],
            cluster_data['Dimension 2'],
            color=cmap(i),
            label=f'Cluster {cluster_id + 1}',
            s=60,  # size of points
            edgecolor='k',  # optional: black edge for visibility
            alpha=0.8
        )

    plt.xlabel('Dimension 1', fontsize=font_size)
    plt.ylabel('Dimension 2', fontsize=font_size)
    plt.title('MDS of Response Clusters', fontsize=font_size + 2)
    # plt.legend(title="Clusters", loc='upper right')
    plt.legend(fontsize=font_size - 2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # plot stress scores
    plt.figure(figsize = fig_size)
    plt.plot(dimensions, stress_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Dimensions', fontsize=font_size)
    plt.ylabel('Stress Score', fontsize=font_size)
    plt.title('Stress Scores for Different Dimensions', fontsize=font_size + 2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

