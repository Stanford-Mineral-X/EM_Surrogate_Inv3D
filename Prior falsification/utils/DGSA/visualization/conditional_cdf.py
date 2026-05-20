"""
Plot conditional CDFs of parameters based on conditioning parameters within a cluster.
Based upon the work of Celine Scheidt and Jihoon Park.
"""

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def conditional_cdf(
    parameter_names: list,
    parameter_values: NDArray[np.float64], 
    clustering: dict[str, NDArray[np.int_]],
    conditioned_parameter_name: str, 
    conditioning_parameter_name: str, 
    n_bins: int = 3, 
    which_cluster: int = 1, 
    fig_size: tuple = (6, 4),
    font_size: int = 12,
    font: str = None
    ) -> None:
    """
    Plot conditional CDFs of parameters based on conditioning parameters within a cluster
    
    Parameters:
    -----------
    conditioned_parameter_name : str
        Name of conditioned parameter (e.g. x in x|y)

    conditioning_parameter_name : str
        Name of conditioning parameter (e.g. y in x|y)

    clustering : dict
        Results from clustering analysis containing:
          - cluster_assignments : np.ndarray of shape (n_samples,)
              Cluster index for each sample.
          - medoid_indices : np.ndarray of shape (n_clusters,)
              Indices of cluster medoids.
          - n_points : np.ndarray of shape (n_clusters,)
              Number of points belonging to each cluster.

    parameter_names : list[str]
        List of parameter names

    n_bins :  int, default = 3
        Number of bins for the conditioning parameter

    which_cluster : int, default = 1
        Index of the cluster (class) for which CDFs will be plotted, starting from 1.

    fig_size : tuple, default = (6,4)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
    """
    
    # check the names of variables
    try:
        idx_conditioned_parameter = parameter_names.index(conditioned_parameter_name)
    except ValueError:
        raise ValueError(f'Enter correct name of conditioned parameters: {conditioned_parameter_name}')
    
    try:
        idx_conditioning_parameter = parameter_names.index(conditioning_parameter_name)
    except ValueError:
        raise ValueError(f'Enter correct name of conditioning parameters: {conditioning_parameter_name}')
    
    # check if which cluster is within range
    n_clusters = len(clustering['medoid_indices'])
    if which_cluster < 1 or which_cluster > n_clusters:
        raise ValueError(f'which_cluster should be between 1 and {n_clusters}')
    
    # extract parameter values
    conditioned_param_values = parameter_values[:, idx_conditioned_parameter]
    conditioning_param_values = parameter_values[:, idx_conditioning_parameter]
    
    # define bin levels for the conditioning parameter
    if len(np.unique(conditioning_param_values)) == n_bins:
        bin_levels = np.sort(np.unique(conditioning_param_values))
    else:
        bin_levels = np.quantile(conditioning_param_values, np.arange(1, n_bins) / n_bins)
    
    # find indices of parameter in the specified cluster
    cluster_idx = np.where(clustering['cluster_assignments'] == which_cluster-1)[0]
    if len(cluster_idx) < 3:
        raise ValueError('There are less than 3 points in the selected cluster. Please choose a different cluster.')
    
    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize = fig_size)

    # define colors for bins
    # colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    colors = plt.cm.jet(np.linspace(0, 1, n_bins))

    # plot prior CDF (all data in the cluster)
    ecdf_result = stats.ecdf(conditioned_param_values[cluster_idx])
    fx_c, x_c = ecdf_result.cdf.probabilities, ecdf_result.cdf.quantiles
    ax.step(x_c, fx_c, linewidth=2, color='black', label='Prior (Cluster)', where='post')
    
    # plot conditional CDFs for each bin
    for b in range(n_bins):
        # find indices for this cluster and this bin
        if b == 0:  # First bin
            bin_mask = conditioning_param_values[cluster_idx] <= bin_levels[b]
        elif b == n_bins - 1:  # Last bin
            bin_mask = conditioning_param_values[cluster_idx] > bin_levels[b-1]
        else:  # Middle bins
            bin_mask = (conditioning_param_values[cluster_idx] > bin_levels[b-1]) & (conditioning_param_values[cluster_idx] <= bin_levels[b])
        
        cluster_bin_idx = cluster_idx[bin_mask]
        
        if len(cluster_bin_idx) < 3:
            raise ValueError('There are less than 3 points in the selected bin. Please choose a different bin.')

        # plot CDF for this bin
        ecdf_result = ecdf_result = stats.ecdf(conditioned_param_values[cluster_bin_idx ])
        f_xgy, xc_xgy = ecdf_result.cdf.probabilities, ecdf_result.cdf.quantiles
        ax.step(xc_xgy, f_xgy, linewidth=2, color=colors[b], label=f'Bin {b+1}', where='post')
                    
        
    # customize figure
    ax.tick_params(labelsize=font_size-2)
    ax.set_xlabel(f'{conditioned_parameter_name}|{conditioning_parameter_name}', fontsize=font_size)
    ax.set_ylabel('CDF', fontsize=font_size)
    ax.set_title(f'Conditional CDF from Cluster {which_cluster-1}', fontsize=font_size+2)
    ax.legend(fontsize=font_size-2)
    ax.grid(True, alpha=0.3)

    plt.show()
