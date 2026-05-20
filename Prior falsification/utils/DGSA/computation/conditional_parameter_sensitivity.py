"""
Distance-based generalized sensitivity analysis (DGSA)
Compute conditional parameter sensitivity
Based upon the work of Celine Scheidt and Jihoon Park
"""

from numpy.typing import NDArray
import numpy as np
import warnings
from computation.single_parameter_sensitivity import compute_ASL

def conditional_parameter_sensitivity(
        parameter_values: NDArray[np.float64], 
        parameter_names: list[str], 
        clustering: dict,
        alpha: float = 0.95, 
        n_bins: int = 3, 
        n_draws: int = 2000,
        method: str = 'l1norm_and_ASL'
    ) -> dict[str, NDArray[np.float64]]:
    """
    Compute conditional parameter sensitivity using the l1norm and/or the ASL method.

    Parameters
    ----------
    parameter_values : np.ndarray of shape (n_samples, n_parameters)
        Matrix of model input parameters, where each column represents one parameter and each row contains one sample.

    parameter_names : list[str]
        List of parameter names

    clustering : dict
        Results from clustering analysis containing:
          - cluster_assignments : np.ndarray of shape (n_samples,)
              Cluster index for each sample.
          - medoid_indices : np.ndarray of shape (n_clusters,)
              Indices of cluster medoids.
          - n_points : np.ndarray of shape (n_clusters,)
              Number of points belonging to each cluster.

    alpha : float, default = 0.95
        A user-specified standard to determine if a parameter is sensitive or not
        For the l1norm method, it is the quantile of the bootstrapped distances.
        For the ASL method, it is used to perform a hypothesis test

    n_bins :  int, default = 3
        Number of bins for the conditioning parameter

    n_draws : int, default = 2000
        Number of bootstrap draws

    Returns
    -------
    Conditional-parameter sensitivity results using the l1norm and/or the ASL method: dict
    'conditional_l1norm' or 'conditional_ASL'
        - 'by_cluster_and_bin' : np.ndarray of shape (n_parameters,n_parameters,n_clusters,n_bins)
            Conditonal sensitivity (1st index for conditioned parameter, 2nd for conditioning parameter, i.e., 1st|2nd) for each cluster and each bin
        - 'standardized' : np.ndarray of shape (n_parameters,n_parameters)
            Standardized condtional sensitivity by finding max values over clusters and bins for both l1norm and ASL.
        - 'hypothesis_test' : np.ndarray of shape (n_parameters,n_parameters), dtype=bool
            Boolean array indicating statistically significant sensitivities.
        _ 'alpha' : float
            Save for reference.
        - 'sensitivity_method' : str
            Save for reference.
    """

    # check method keyword
    if method not in ['l1norm', 'ASL', 'l1norm_and_ASL']:
        raise ValueError("Invalid method. Choose from: 'l1norm', 'ASL', 'l1norm_and_ASL'")
        
    # get dimensions
    n_params = parameter_values.shape[1]
    n_clusters = len(clustering['medoid_indices'])

    # initialize arrays
    l1norm_distance_observed = np.full((n_params, n_params, n_clusters, n_bins), np.nan)
    boot_distance_samples = np.full((n_params, n_params, n_clusters, n_bins, n_draws), np.nan)
    boot_distance_alpha = np.full((n_params, n_params, n_clusters, n_bins), np.nan)

    # compute observed l1norm distances and perform bootstrap sampling
    for cond_param_idx in range(n_params):
        print(f"Computing sensitivity conditioned on {parameter_names[cond_param_idx]}...")

        (l1norm_distance_observed[:, cond_param_idx, :, :], # array shape (n_parameters,n_parameters,n_clusters,n_bins)
        boot_distance_samples[:, cond_param_idx, :, :, :], # array shape (n_parameters,n_parameters,n_clusters,n_bins,n_draws) 
        boot_distance_alpha[:, cond_param_idx, :, :] # array shape (n_parameters,n_parameters,n_clusters,n_bins)
        ) = compute_conditional_l1norm_distance_observed_and_bootstrapped(
            parameter_values, 
            cond_param_idx, 
            clustering, 
            n_bins, 
            n_draws, 
            alpha
        ) 
        
    # save sensitivity results to a dictionary
    sensitivity_results = {}

    # compute l1norm sensitivity
    if method in ['l1norm', 'l1norm_and_ASL']:
        # Compute l1norm conditional parameter sensitivity
        l1norm_sensitivity_by_cluster_and_bin = l1norm_distance_observed / boot_distance_alpha # array shape (n_parameter,n_parameter,n_cluster,n_bins)
        # Compute standardized conditional parameter sensitivity
        with warnings.catch_warnings(): # Suppress the warning about all-NaN slices in the diagonal
            warnings.simplefilter("ignore", category=RuntimeWarning)
            l1norm_sensitivity_by_cluster = np.nanmax(l1norm_sensitivity_by_cluster_and_bin, axis=3) # array shape (n_parameter,n_parameter,n_cluster)
            l1norm_sensitivity_standardized = np.nanmax(l1norm_sensitivity_by_cluster, axis=2) # array shape (n_parameter,n_parameter)
            # l1norm_sensitivity_by_cluster = np.nanmean(l1norm_sensitivity_by_cluster_and_bin, axis=3) # array shape (n_parameter,n_parameter,n_cluster)
            # l1norm_sensitivity_standardized = np.nanmean(l1norm_sensitivity_by_cluster, axis=2) # array shape (n_parameter,n_parameter)
        # perform hypothesis test to determine if a parameter is sensitive or not
        l1norm_hypothesis_test = l1norm_sensitivity_standardized >= 1 # array shape (n_parameter,n_parameter)

        # save l1norm sensitivity
        sensitivity_results['conditional_l1norm'] = {
            'by_cluster_and_bin': l1norm_sensitivity_by_cluster_and_bin,
            'standardized': l1norm_sensitivity_standardized,
            'hypothesis_test': l1norm_hypothesis_test,
            'alpha': alpha,
            'sensitivity_method': 'l1norm'
        }

    # compute ASL sensitivity
    if method in ['ASL', 'l1norm_and_ASL']:
        # compute the ASL sensitivity by looping through parameters, clusters and bins
        ASL_sensitivity_by_cluster_and_bin = np.full((n_params, n_params, n_clusters, n_bins), np.nan)
        for p in range(n_params):
            for cond_param_idx in range(n_params):
                for c in range(n_clusters):
                    for b in range(n_bins):
                        ASL_sensitivity_by_cluster_and_bin[p, cond_param_idx, c, b] = compute_ASL(
                            l1norm_distance_observed[p, cond_param_idx, c, b], boot_distance_samples[p, cond_param_idx, c, b, :])

        # Compute standardized sensitivity by finding max ASL values (more informative than mean ASL)
        ASL_sensitivity_by_cluster = np.nanmax(ASL_sensitivity_by_cluster_and_bin, axis=3) # array shape (n_parameter,n_parameter,n_cluster)
        ASL_sensitivity_standardized = np.nanmax(ASL_sensitivity_by_cluster, axis=2) # array shape (n_parameter,n_parameter)
        # ASL_sensitivity_by_cluster = np.nanmean(ASL_sensitivity_by_cluster_and_bin, axis=3) # array shape (n_parameter,n_parameter,n_cluster)
        # ASL_sensitivity_standardized = np.nanmean(ASL_sensitivity_by_cluster, axis=2) # array shape (n_parameter,n_parameter)
        # perform hypothesis test to determine if a parameter is sensitive or not
        ASL_hypothesis_test = ASL_sensitivity_standardized >= alpha # array shape (n_parameter,n_parameter)

        # save ASL sensitivity
        sensitivity_results['conditional_ASL'] = {
            'by_cluster_and_bin': ASL_sensitivity_by_cluster_and_bin*100, # convert to percentages
            'standardized': ASL_sensitivity_standardized*100, # convert to percentages
            'hypothesis_test': ASL_hypothesis_test,
            'alpha': alpha,
            'sensitivity_method': 'ASL'
        }

    print(f"Completed conditional parameter sensitivity analysis using the {method} method")
    
    return sensitivity_results


def compute_conditional_l1norm_distance_observed_and_bootstrapped(
        parameter_values: NDArray[np.float64], 
        cond_idx: int, 
        clustering: dict, 
        n_bins: int = 3, 
        n_draws: int = 2000,
        alpha: float = 0.95
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    For conditional parameter sensitivity: 
        (1) compute the observed l1norm distances
        (2) perform bootstrap sampling, output all sampled l1norm distances and those at the specified alpha-quantile

    Parameters
    ----------
    parameter_values : np.ndarray of shape (n_samples, n_parameters)
        Matrix of model input parameters, where each column represents one parameter and each row contains one sample.

    cond_idx : int
        Index of the conditioning parameter (e.g y in x|y)

    clustering : dict
        Results from clustering analysis containing:
          - cluster_assignments : np.ndarray of shape (n_samples,)
              Cluster index for each sample.
          - medoid_indices : np.ndarray of shape (n_clusters,)
              Indices of cluster medoids.
          - n_points : np.ndarray of shape (n_clusters,)
              Number of points belonging to each cluster.

    n_bins :  int, default = 3
        Number of bins for the conditioning parameter

    n_draws : int, default = 2000
        Number of bootstrap draws.

    alpha : float, default = 0.95
        Quantile of the bootstrapped distribution

    Returns
    -------
    conditional_observed_l1norm_distances : np.ndarray of shape (n_parameters, n_clusters, n_bins)
        The observed l1norm distances for each parameter, each cluster, and each bin.

    conditional_boot_distance_samples : np.ndarray of shape (n_parameters, n_clusters, n_bins, n_draws)
        The bootstrapped l1norm distances for each parameter, each cluster, and each bin.

    conditional_boot_distance_quantiles : np.ndarray of shape (n_parameters, n_clusters, n_bins)
        The bootstrapped l1norm distances at the specified alpha-quantile for each parameter, each cluster, and each bin.
    """

    # get parameters
    n_params = parameter_values.shape[1]
    n_clusters = len(clustering['medoid_indices'])
    
    # set up quantiles
    quantile_grid = np.linspace(0.01, 0.99, 99)

    # initialize output array 
    # initialized with nan instead of 0, so that we can skip bins with no points.
    conditional_observed_l1norm_distances = np.full((n_params, n_clusters, n_bins), np.nan)
    # initialize output array to store bootstrapped l1norm distance samples
    conditional_boot_distance_samples = np.full((n_params, n_clusters, n_bins, n_draws), np.nan)
    # initialize output array to store bootstrapped l1norm distances at the specified alpha-quantiles
    conditional_boot_distance_quantiles = np.zeros((n_params, n_clusters, n_bins))

    # define bin levels for the conditioning parameter
    unique_param_values = np.unique(parameter_values[:, cond_idx])
    if unique_param_values.shape[0] == n_bins:
        bin_levels = np.sort(unique_param_values)
    else:
        bin_levels = np.quantile(parameter_values[:, cond_idx], np.arange(1, n_bins) / n_bins)


    for c in range(n_clusters):
        # find points in the cluster
        cluster_idx = np.where(clustering['cluster_assignments'] == c)[0]
        n_points_cluster = len(cluster_idx)
        
        # compute prior cdfs for all parameters in this cluster: F(p|c)
        q_prior = np.quantile(parameter_values[cluster_idx], quantile_grid, axis=0)

        # bin the conditioning parameter and compute bootstrap for each bin
        for b in range(n_bins):
            # find indices for this cluster and this bin
            if b == 0:
                bin_mask = parameter_values[cluster_idx, cond_idx] <= bin_levels[b]
            elif b == n_bins - 1:
                bin_mask = parameter_values[cluster_idx, cond_idx] > bin_levels[b-1]
            else:
                bin_mask = ((parameter_values[cluster_idx, cond_idx] <= bin_levels[b]) & (parameter_values[cluster_idx, cond_idx] > bin_levels[b-1]))
            cluster_bin_idx = cluster_idx[bin_mask]
            n_points_cluster_bin = len(cluster_bin_idx)

            # skip if too few points in the bin, defaut to nan
            if n_points_cluster_bin <= 3:
                print(f"Less than 3 points in bin {b} for cluster {c}, skipping this bin")
            else:
                # compute observed l1norm distances
                # compute cdfs for all parameters in this cluster and this bin
                q_bin = np.quantile(parameter_values[cluster_bin_idx], quantile_grid, axis=0)
                # compute l1norm distances for all parameters 
                # conditional_observed_l1norm_distances[:, c, b] = np.linalg.norm(q_prior - q_bin, ord=1, axis=0)
                conditional_observed_l1norm_distances[:, c, b] = np.sum(np.abs(q_prior - q_bin), axis=0)
                # set nan to the conditioning parameter
                conditional_observed_l1norm_distances[cond_idx, c, b] = np.nan 

                # perform bootstrap sampling
                for p in range(n_params):
                    # draw bootstrap samples from the cluster without replacement
                    draw_idx = np.column_stack([np.random.choice(n_points_cluster, size=n_points_cluster_bin, replace=False) for _ in range(n_draws)])
                    boot_samples = parameter_values[cluster_idx[draw_idx], p] # shape (n_points_j, n_draws)

                    # compute bootstrapped cdfs
                    q_boot = np.quantile(boot_samples, quantile_grid, axis=0) # shape (n_quantiles, n_draws)

                    # compute bootstrapped l1norm distances
                    diff = q_boot - q_prior[:, p][:, None] # broadcast to (n_quantiles, n_draws)
                    boot_distances = np.sum(np.abs(diff), axis=0) # shape (n_draws,)

                    # save bootstrapped l1norm distances
                    conditional_boot_distance_samples[p, c, b, :] = boot_distances

                    # extract bootstrapped l1norm distances at the specified alpha-quantiles
                    conditional_boot_distance_quantiles[p, c, b] = np.quantile(boot_distances, alpha)

    return  conditional_observed_l1norm_distances, conditional_boot_distance_samples, conditional_boot_distance_quantiles
