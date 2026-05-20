"""
Make a heatmap showing the standardized sensitivity values for conditional parameter sensitivity.
Based upon the work of Celine Scheidt and Jihoon Park.
"""

from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import TwoSlopeNorm 

def conditional_heatmap_standardized(
    single_sensitivity_results: dict,
    conditional_sensitivity_results: dict,
    parameter_names: list[str],
    title: str = None,
    fig_size: tuple = (10, 8),
    font: str = None,
    font_size: int = 12,
    show_values: bool = True
    ) -> None:
    """
    Make a heatmap showing the standardized sensitivity values for conditional parameter sensitivity.
    
    Parameters
    ----------
    single_sensitivity_results : dict
        Results from single-parameter sensitivity analysis containing:
        'single_l1norm' or 'single_ASL'
            - 'by_cluster' : np.ndarray of shape (n_parameters, n_clusters)
                Sensitivity of each parameter across clusters.
            - 'standardized' : np.ndarray of shape (n_parameters,)
                Standardized sensitivity values for each parameter.
            - 'hypothesis_test' : np.ndarray of shape (n_parameters,), dtype=bool
                Boolean array indicating statistically significant sensitivities.
            _ 'alpha' : float
                A user-specified standard to determine if a parameter is sensitive or not
                For the l1norm method, it is the quantile of the bootstrapped distances.
                For the ASL method, it is used to perform a hypothesis test
            - 'sensitivity_method' : str
                Method to compute sensitivity: l1norm or ASL.

    conditional_sensitivity_results : dict
        Results from conditional parameter sensitivity analysis containing:
        'conditional_l1norm' or 'conditional_ASL'
            - 'by_cluster_and_bin' : np.ndarray of shape (n_parameters,n_parameters,n_clusters,n_bins)
                Conditonal sensitivity (1st index for conditioned parameter, 2nd for conditioning parameter, i.e., 1st|2nd) for each cluster and each bin
            - 'standardized' : np.ndarray of shape (n_parameters,n_parameters)
                Standardized condtional sensitivity over clusters and bins.
            - 'hypothesis_test' : np.ndarray of shape (n_parameters,n_parameters), dtype=bool
                Boolean array indicating statistically significant sensitivities.
            _ 'alpha' : float
                A user-specified standard to determine if a parameter is sensitive or not
                For the l1norm method, it is the quantile of the bootstrapped distances.
                For the ASL method, it is used to perform a hypothesis test
            - 'sensitivity_method' : str
                Method to compute sensitivity: l1norm or ASL.

    parameter_names : list[str]
        List of parameter names

    title : str, default = None
        Title for the plot. If None, no title is displayed.

    fig_size : tuple, default = (10, 8)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
        
    show_values : bool, default = True
        Whether to show the numerical values in the heatmap cells.

    Returns
    -------
    A heatmap showing the standardized sensitivity values for conditional parameter sensitivity.
    """

    # get dimensions
    n_params = conditional_sensitivity_results['standardized'].shape[0]

    # convert inputs to numpy arrays
    parameter_names = np.asarray(parameter_names)

    # check if missing data
    required_keys = ['by_cluster', 'standardized', 'alpha', 'sensitivity_method']
    results_keys = set(single_sensitivity_results.keys())
    missing_keys = [key for key in required_keys if key not in results_keys]
    if missing_keys:
        # Use join to create a comma-separated string of the missing keys
        missing_str = ', '.join(missing_keys) 
        raise ValueError(f"The sensitivity results dictionary does not contain the required key(s): {missing_str}.")

    # check if parameter names are complete
    if parameter_names.size != single_sensitivity_results['standardized'].size:
        raise ValueError("parameter_names must match the sensitivity array length.")

    # sort from least sensitive to most sensitive
    conditional_standardized = conditional_sensitivity_results['standardized'].copy()
    single_standardized = single_sensitivity_results['standardized'].copy()
    np.fill_diagonal(conditional_standardized,single_standardized)
    sort_idx = np.argsort(single_standardized)[::-1]
    sorted_sensitivity = conditional_standardized[sort_idx][:, sort_idx]    
    sorted_names = parameter_names[sort_idx]
    
    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)

    if single_sensitivity_results['sensitivity_method'] == 'l1norm':
        vcenter, vmin, vmax = 1, 0.1, 1.9
        ticks=[0.3, 1, 1.7] # colorbar ticks
    elif single_sensitivity_results['sensitivity_method'] == 'ASL':
        # vcenter, vmin, vmax = 95, 10, 99.999
        vcenter, vmin, vmax = conditional_sensitivity_results['alpha']*100, 10, 99.999
        ticks=[30, conditional_sensitivity_results['alpha']*100, 99] # colorbar ticks
    
    # normalize colormap
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    
    # plot heatmap
    im = ax.imshow(sorted_sensitivity, cmap=cm.vik, norm=norm, interpolation='none', aspect='equal')
    
    # add colorbar with proper positioning
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_yticklabels(['Insensitive', 'Important', 'Critical'], fontsize=font_size)

    # set ticks and labels
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(sorted_names, rotation=45)
    ax.set_yticklabels(sorted_names)
    ax.set_title(title, fontsize=font_size+2)
    
    # add grid lines
    for pos in range(n_params+1):
        ax.axhline(y=pos-0.5, color='k', linewidth=1.5, alpha=0.7)
        ax.axvline(x=pos-0.5, color='k', linewidth=1.5, alpha=0.7)
    
    # add border lines
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # add text values
    if show_values:
        for (r, c), value in np.ndenumerate(sorted_sensitivity):
            if value>99.9:
                ax.text(c, r, "99.9+", ha='center', va='center', color='k', fontsize=font_size-2)
            else:
                ax.text(c, r, f"{value:.1f}", ha='center', va='center', color='k', fontsize=font_size-2)

    plt.show()