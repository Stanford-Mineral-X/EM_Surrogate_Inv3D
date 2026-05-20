"""
Plot CDFs of parameters for each cluster.
Based upon the work of Celine Scheidt and Jihoon Park.
"""

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def single_cdf(
    parameter_names: list,
    parameter_values: NDArray[np.float64],
    clustering: dict[str, NDArray[np.int_]],
    plot_parameter_list: list = None,
    fig_size: tuple = (8, 10),
    font_size: int = 12,
    font: str = None
    ) -> None:
    """
    Plot cumulative distribution functions (CDFs) of parameters for each cluster
    
    Parameters:
    -----------
    parameter_values : np.ndarray of shape (n_samples, n_parameters)
        Matrix of model input parameters, where each column represents one parameter and each row contains one sample.
        
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

    plot_parameter_list : list, default = None
        List of parameter names for plotting. If none, CDFs of all parameters will be plotted.

    fig_size : tuple, default = (8,10)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
    """
    
    # if plot_parameter_list is not specified, use all parameters
    if plot_parameter_list is None:
        plot_parameter_list = parameter_names
    
    # find indices of specified parameters
    parameters_index = []
    for param in plot_parameter_list:
        try:
            param_idx = parameter_names.index(param)
            parameters_index.append(param_idx)
        except ValueError:
            raise ValueError(f'Enter correct name of parameters: {param} not found')
    
    # get number of clusters
    n_clusters = len(clustering['medoid_indices'])
    
    # calculate number of rows needed for subplots
    n_params_to_plot = len(parameters_index)
    n_rows = (n_params_to_plot + 1) // 2  # ceiling division to get enough rows
        
    # ensure axes is always a 2D array for consistent indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure with subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=fig_size)
        
    # define colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    # colors = plt.cm.jet(np.linspace(0, 1, n_clusters))

    # plot CDF for each specified parameter
    for k, param_idx in enumerate(parameters_index):
        # calculate row and column for subplot
        row = k // 2
        col = k % 2
        ax = axes[row, col]
        
        # calculate prior CDF (all data)
        prior_data = parameter_values[:, param_idx]
        prior_data = prior_data[~np.isnan(prior_data)]  # remove NaN values
        
        # plot prior CDF
        if len(prior_data) > 0:
            ecdf_result = stats.ecdf(prior_data)
            x_prior, f_prior = ecdf_result.cdf.quantiles, ecdf_result.cdf.probabilities
            ax.step(x_prior, f_prior, linewidth=2, color='black', label='Prior', where='post')
        
        # plot CDF for each cluster
        for c in range(n_clusters):
            # get data for current cluster
            cluster_mask = clustering['cluster_assignments'] == c
            cluster_data = parameter_values[cluster_mask, param_idx]
            cluster_data = cluster_data[~np.isnan(cluster_data)]  # remove NaN values
            
            if len(cluster_data) > 0:
                ecdf_result = stats.ecdf(cluster_data)
                x_cluster, f_cluster = ecdf_result.cdf.quantiles, ecdf_result.cdf.probabilities
                ax.step(x_cluster, f_cluster, linewidth=2, color=colors[c], label=f'Cluster {c+1}', where='post')
        
        # customize figure
        ax.tick_params(labelsize=font_size-2)
        ax.set_xlabel(parameter_names[param_idx], fontsize=font_size)
        ax.set_ylabel('CDF', fontsize=font_size)
        # ax.set_title(f'CDF of {parameter_names[param_idx]} by Cluster', fontsize=16, fontweight='bold')
        ax.legend(fontsize=font_size-2)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots if there are an odd number of parameters
    if n_params_to_plot % 2 == 1:
        axes[-1, -1].set_visible(False)
    
    plt.tight_layout()
    plt.show()