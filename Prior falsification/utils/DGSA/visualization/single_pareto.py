"""
Make Pareto plots showing single-parameter sensitivity.
Based upon the work of Celine Scheidt and Jihoon Park.
"""

from numpy.typing import NDArray
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import TwoSlopeNorm 

def single_pareto_standardized(
        single_sensitivity_results: dict,
        parameter_names: list[str],
        title: str = None,
        fig_size: tuple = (10, 6),
        font_size: int = 12,
        font: str = None
    ) -> None:
    """
    Make a Pareto plot showing the standardized measure of sensitivity values for each parameter.

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

    parameter_names : list[str]
        List of parameter names

    title : str, default = None
        Title for the plot. If None, no title is displayed.

    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.

    Returns
    -------
    A Pareto plot showing the standardized measure of sensitivity values for each parameter.
    """
     # get dimensions
    n_params = len(single_sensitivity_results['standardized'])

    # convert inputs to numpy arrays
    parameter_names = np.asarray(parameter_names)

    # check if missing data
    required_keys = ['by_cluster', 'standardized', 'hypothesis_test', 'alpha', 'sensitivity_method']
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
    sort_idx = np.argsort(single_sensitivity_results['standardized'])
    sorted_sensitivity = single_sensitivity_results['standardized'][sort_idx]
    sorted_names = parameter_names[sort_idx]

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_axisbelow(True)  # Put grid behind bars
    y_pos = np.arange(1, n_params + 1)
    
    # make the plot
    if single_sensitivity_results['sensitivity_method'] == 'l1norm':
        data_to_plot = sorted_sensitivity
        # to normalize colormap
        vcenter, vmin, vmax = 1, 0.1, 1.9
        # colorbar tick
        ticks=[0.3, 1, 1.7] 

    elif single_sensitivity_results['sensitivity_method'] == 'ASL':
        # Transform sensitivities to z-scores for visualization
        sorted_sensitivity_zscore = norm.ppf(sorted_sensitivity/100, loc=2, scale=1)
        # Replace NaN/inf with max + eps
        finite_mask = np.isfinite(sorted_sensitivity_zscore)
        if not np.all(finite_mask):
            max_val = np.max(sorted_sensitivity_zscore[finite_mask])
            sorted_sensitivity_zscore[~finite_mask] = max_val + np.finfo(float).eps

        data_to_plot = sorted_sensitivity_zscore
        # to normalize colormap
        vcenter, vmin, vmax = norm.ppf([single_sensitivity_results['alpha'], 0.1, 0.99999], loc=2, scale=1)
        # colorbar tick
        tick_low,  tick_high  = norm.ppf([0.3, 0.9997], loc=2, scale=1)
        ticks=[tick_low, vcenter, tick_high]

    # normalize colormap
    color_norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    # make the plot
    ax.barh(y_pos, data_to_plot, color=cm.vik(color_norm(data_to_plot)), height=0.8)
    # add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cm.vik), ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_yticklabels(['Insensitive', 'Important', 'Critical'], fontsize=font_size)

    # customize figure
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.set_ylim(0, n_params + 1)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel('Standardized Sensitivity Measure', fontsize=font_size)
    ax.set_title(title, fontsize=font_size+2)

    if single_sensitivity_results['sensitivity_method']  == 'l1norm':
        # add reference line at x=1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)

    elif single_sensitivity_results['sensitivity_method']  == 'ASL':
        # x tick labels
        # TickToDisplay = np.array([0.05, 0.2, 0.5, 0.8, 0.95, 0.993, 0.9995, 0.99999])
        TickToDisplay = np.array([5, 20, 50, 80, 95, 99.3, 99.95, 99.999])
        ax.set_xticks(norm.ppf(TickToDisplay/100, loc=2, scale=1))
        ax.set_xticklabels(TickToDisplay)
        # add reference line at alpha
        ax.axvline(x=norm.ppf(single_sensitivity_results['alpha'], loc=2, scale=1), color='black', linestyle='--', linewidth=2, alpha=0.5)

    plt.show()


def single_pareto_by_cluster(
        single_sensitivity_results: dict,
        parameter_names: list[str],
        title: str = None,
        fig_size: tuple = (10, 6),
        font_size: int = 12,
        font: str = None
    ) -> None:
    """
    Make a Pareto plot showing the standardized measure of sensitivity values for each parameter and each cluster.

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
                Method to compute sensitivity: l1norm or ASL .

    parameter_names : list[str]
        List of parameter names

    title : str, default = None
        Title for the plot. If None, no title is displayed.

    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.

    Returns
    -------
    A Pareto plot showing the standardized measure of sensitivity values for each parameter and each cluster..
    """
    
    # get dimensions
    n_params, n_clusters = single_sensitivity_results['by_cluster'].shape
    
    # convert inputs to numpy arrays
    parameter_names = np.asarray(parameter_names)

    # check if missing data
    required_keys = ['by_cluster', 'standardized', 'hypothesis_test', 'alpha', 'sensitivity_method']
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
    sort_idx = np.argsort(single_sensitivity_results['standardized'])
    sorted_sensitivity = single_sensitivity_results['by_cluster'][sort_idx]
    sorted_names = parameter_names[sort_idx]

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_axisbelow(True)  # Put grid behind bars
    y_pos = np.arange(n_params)
    
    # set up bar height and colors
    total_bar_height = 0.8
    bar_height = total_bar_height / n_clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    # colors = plt.cm.jet(np.linspace(0, 1, n_clusters))

    # make the plot
    if single_sensitivity_results['sensitivity_method'] == 'l1norm':
        sensitivity_to_plot = sorted_sensitivity

    elif single_sensitivity_results['sensitivity_method'] == 'ASL':
        # Transform sensitivities to z-scores for visualization
        # sorted_sensitivity_zscore = norm.ppf(sorted_sensitivity, loc=2, scale=1)
        sorted_sensitivity_zscore = norm.ppf(sorted_sensitivity/100, loc=3, scale=1)
        # Replace NaN/inf with max + eps
        finite_mask = np.isfinite(sorted_sensitivity_zscore)
        if not np.all(finite_mask):
            max_val = np.max(sorted_sensitivity_zscore[finite_mask])
            sorted_sensitivity_zscore[~finite_mask] = max_val + np.finfo(float).eps
        sensitivity_to_plot = sorted_sensitivity_zscore

    for c in range(n_clusters):
        offset = (c - (n_clusters - 1) / 2) * bar_height
        ax.barh(y_pos + offset, sensitivity_to_plot[:, c], height=bar_height, color=colors[c], label=f'Cluster {c+1}')

    # customize figure
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel('Sensitivity Measure', fontsize=font_size)
    ax.set_title(title, fontsize=font_size+2)
    ax.legend(fontsize=font_size)

    if single_sensitivity_results['sensitivity_method']  == 'l1norm':
        # add reference line at x=1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    elif single_sensitivity_results['sensitivity_method']  == 'ASL':
        # x tick labels
        # TickToDisplay = np.array([0.05, 0.2, 0.5, 0.8, 0.95, 0.993, 0.9995, 0.99999])
        TickToDisplay = np.array([5, 20, 50, 80, 95, 99.3, 99.95, 99.999])
        ax.set_xticks(norm.ppf(TickToDisplay/100, 2, 1))
        ax.set_xticklabels(TickToDisplay)

        # add reference line at alpha
        ax.axvline(x=norm.ppf(single_sensitivity_results['alpha'], loc=2, scale=1), color='black', linestyle='--', linewidth=2, alpha=0.5)

    plt.show()


def single_pareto_ci(
    standardized_low_alpha: NDArray[np.float64],
    standardized_medium_alpha: NDArray[np.float64],
    standardized_high_alpha: NDArray[np.float64],
    parameter_names: list[str],
    title: str = None,
    fig_size: tuple = (10, 6),
    font_size: int = 12,
    font: str = None
    ) -> None:
    """
    Make a Pareto plot showing the standardized single-parameter sensitivity with confidence intervals.
    
    Parameters:
    -----------
    standardized_low_alpha : np.ndarray of shape (n_parameters,)
        Standardized single-parameter sensitivity calculated using a low alpha value (e.g., 0.91).

    standardized_medium_alpha : np.ndarray of shape (n_parameters,)
        Standardized single-parameter sensitivity calculated using a medium alpha value (e.g., 0.95).

    standardized_high_alpha : np.ndarray of shape (n_parameters,)
        Standardized single-parameter sensitivity calculated using a high alpha value (e.g., 0.99).

    parameter_names : list[str]
        List of parameter names

    title : str, default = None
        Title for the plot. If None, no title is displayed.

    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
    
    Returns:
    --------
    A Pareto plot showing the standardized single-parameter sensitivity with confidence intervals.
    """
    # get dimensions
    n_params = len(standardized_medium_alpha)

    # convert inputs to numpy arrays
    parameter_names = np.asarray(parameter_names)

    # check if parameter names are complete
    if parameter_names.size != n_params:
        raise ValueError("parameter_names must match the sensitivity array length.")
    
    # sort from least sensitive to most sensitive
    sort_idx = np.argsort(standardized_medium_alpha)
    sorted_low_alpha = standardized_low_alpha[sort_idx]
    sorted_medium_alpha = standardized_medium_alpha[sort_idx]
    sorted_high_alpha = standardized_high_alpha[sort_idx]
    sorted_names = parameter_names[sort_idx]
    
    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_axisbelow(True)  # Put grid behind bars
    y_pos = np.arange(1, n_params + 1)

    # normalize colormap
    vcenter, vmin, vmax = 1, 0.1, 1.9
    color_norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    # calculate error
    errors = (sorted_low_alpha - sorted_high_alpha)/2
    # make the plot
    ax.barh(y_pos, sorted_medium_alpha, color=cm.vik(color_norm(sorted_medium_alpha)), height=0.8, xerr=errors, error_kw={'capsize': 3, 'capthick': 1})
    # add colorbar
    ticks=[0.3, 1, 1.7] 
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cm.vik), ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_yticklabels(['Insensitive', 'Important', 'Critical'], fontsize=font_size)

    # customize figure
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.set_ylim(0, n_params + 1)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel('Standardized Sensitivity Measure', fontsize=font_size)
    ax.set_title(title, fontsize=font_size+2)
    # add reference line at x=1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.show()
