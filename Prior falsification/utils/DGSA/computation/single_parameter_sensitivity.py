"""
Distance-based generalized sensitivity analysis (DGSA)
Compute single-parameter sensitivity
Based upon the work of Celine Scheidt and Jihoon Park
"""

from numpy.typing import NDArray
import numpy as np
from scipy.stats import lognorm, gamma, weibull_min, beta

def single_parameter_sensitivity(
    parameter_values: NDArray[np.float64],
    clustering: dict[str, NDArray[np.int_]],
    alpha: float = 0.95,
    n_draws: int = 2000,
    method: str = 'l1norm_and_ASL'
    ) -> dict[str, NDArray[np.float64]]:
    
    """
    Compute single-parameter sensitivity using the l1norm and/or the ASL method.

    Parameters
    ----------
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

    alpha : float, default = 0.95
        A user-specified standard to determine if a parameter is sensitive or not
        For the l1norm method, it is the quantile of the bootstrapped distances.
        For the ASL method, it is used to perform a hypothesis test

    n_draws : int, default = 2000
        Number of bootstrap draws.

    Returns
    -------
    Single-parameter sensitivity results: dict
    'single_l1norm' or 'single_ASL'
        - 'by_cluster' : np.ndarray of shape (n_parameters, n_clusters)
            Sensitivity of each parameter across clusters.
        - 'standardized' : np.ndarray of shape (n_parameters,)
            Standardized sensitivity values by averaging over clusters for l1norm and finding max values for ASL.
        - 'hypothesis_test' : np.ndarray of shape (n_parameters,), dtype=bool
            Boolean array indicating statistically significant sensitivities.
        _ 'alpha' : float
            Save for reference.
        - 'sensitivity_method' : str
            Save for reference.
    """

    # check method keyword
    if method not in ['l1norm', 'ASL', 'l1norm_and_ASL']:
        raise ValueError("Invalid method. Choose from: 'l1norm', 'ASL', 'l1norm_and_ASL'")

    # compute observed l1norm distances and perform bootstrap sampling
    l1norm_distance_observed, boot_distance_samples, boot_distance_alpha = compute_single_l1norm_distance_observed_and_bootstrapped(
        parameter_values, clustering, n_draws, alpha) # array shape (n_parameters,n_clusters), (n_parameters,n_clusters,n_draws), (n_parameters,n_clusters)

    # save sensitivity results to a dictionary
    sensitivity_results = {}

    # compute l1norm sensitivity
    if method in ['l1norm', 'l1norm_and_ASL']:
        # compute single parameter sensitivity
        l1norm_sensitivity_by_cluster = l1norm_distance_observed / boot_distance_alpha # array shape (n_parameters,n_clusters)
        # compute standardized sensitivity measure by averaging over clusters
        l1norm_sensitivity_standardized = np.mean(l1norm_sensitivity_by_cluster, axis=1) # array shape (n_parameters,)
        # perform hypothesis test to determine if a parameter is sensitive or not
        l1norm_hypothesis_test = l1norm_sensitivity_standardized >= 1 # array shape (n_parameters,)
       
        # save l1norm sensitivity
        sensitivity_results['single_l1norm'] = {
        'by_cluster': l1norm_sensitivity_by_cluster,
        'standardized': l1norm_sensitivity_standardized,
        'hypothesis_test': l1norm_hypothesis_test,
        'alpha': alpha,
        'sensitivity_method': 'l1norm'
        }

    # compute ASL sensitivity
    if method in ['ASL', 'l1norm_and_ASL']:
        # compute the ASL sensitivity by looping through parameters and clusters
        n_params = parameter_values.shape[1]
        n_clusters = len(clustering['medoid_indices'])
        ASL_sensitivity_by_cluster = np.full((n_params, n_clusters), np.nan)
        for p in range(n_params):
            for c in range(n_clusters):
                ASL_sensitivity_by_cluster[p, c] = compute_ASL(l1norm_distance_observed[p, c], boot_distance_samples[p, c, :]) # array shape (n_parameter,n_cluster)

        # compute standardized sesnitivity by finding max ASL values (more informative than mean ASL)
        ASL_sensitivity_standardized = np.max(ASL_sensitivity_by_cluster, axis=1) # array shape (n_parameter,)
        # ASL_sensitivity_standardized = np.mean(ASL_sensitivity_by_cluster, axis=1) # array shape (n_parameter,)
        # perform hypothesis test to determine if a parameter is sensitive or not
        ASL_hypothesis_test = ASL_sensitivity_standardized >= alpha # array shape (n_parameters,)
       
        # save ASL sensitivity
        sensitivity_results['single_ASL'] = {
        'by_cluster': ASL_sensitivity_by_cluster*100, # convert to percentages
        'standardized': ASL_sensitivity_standardized*100, # convert to percentages
        'hypothesis_test': ASL_hypothesis_test,
        'alpha': alpha,
        'sensitivity_method': 'ASL'
        }

    print(f"Completed single parameter sensitivity analysis using the {method} method")
    return sensitivity_results


def compute_single_l1norm_distance_observed_and_bootstrapped(
        parameter_values: NDArray[np.float64],
        clustering: dict[str, NDArray[np.int_]],
        n_draws: int = 2000, 
        alpha: float = 0.95
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    For single parameter sensitivity: 
        (1) compute the observed l1norm distances
        (2) perform bootstrap sampling, output all sampled l1norm distances and those at the specified alpha-quantile

    Parameters
    ----------
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

    n_draws : int, default = 2000
        Number of bootstrap draws.

    alpha : float, default = 0.95
        Quantile of the bootstrapped distribution.

    Returns
    -------
    single_observed_l1norm_distances : np.ndarray of shape (n_parameters, n_clusters)
        The observed l1norm distances for each parameter and each cluster

    single_boot_distance_samples : np.ndarray of shape (n_parameters, n_clusters, n_draws)
        The bootstrapped l1norm distances for each parameter and each cluster

    single_boot_distance_alpha : np.ndarray of shape (n_parameters, n_clusters)
        The bootstrapped l1norm distances at the specified alpha-quantile for each parameter and each cluster
    """

    n_samples, n_params = parameter_values.shape
    n_clusters = len(clustering['medoid_indices'])
    
    # set up quantiles
    quantile_grid = np.linspace(0.01, 0.99, 99)

    # initialize output array to store bootstrapped l1norm distance samples
    single_observed_l1norm_distances = np.full((n_params, n_clusters), np.nan)
    # initialize output array to store bootstrapped l1norm distance samples
    single_boot_distance_samples = np.full((n_params, n_clusters, n_draws), np.nan)
    # initialize output array to store bootstrapped l1norm distances at the specified alpha-quantile
    single_boot_distance_alpha = np.full((n_params, n_clusters), np.nan)
    
    for p in range(n_params):
        # calculate prior distribution
        q_prior = np.quantile(parameter_values[:, p], quantile_grid)
        # perform bootstrap sampling for each cluster
        
        for c in range(n_clusters):
            # compute observed l1norm distances
            # get parameter values for current cluster
            cluster_mask = clustering['cluster_assignments'] == c
            cluster_values = parameter_values[cluster_mask, p]
            # calculate cluster distribution
            q_cluster = np.quantile(cluster_values, quantile_grid)
            # compute l1norm distances
            single_observed_l1norm_distances[p, c] = np.sum(np.abs(q_prior - q_cluster))

            # perform bootstrap sampling
            # find how many points in this cluster
            n_points_cluster = clustering["n_points"][c]
            # draw bootstrap samples from all samples without replacement
            draw_idx = np.column_stack([np.random.choice(n_samples, size=n_points_cluster, replace=False) for _ in range(n_draws)])
            boot_samples = parameter_values[draw_idx, p] # shape (n_points_cluster, n_draws)

            # compute bootstrapped cdfs
            q_boot = np.quantile(boot_samples, quantile_grid, axis=0) # shape (n_quantiles, n_draws)

            # compute bootstrapped l1norm distances
            diff = q_boot - q_prior[:, None] # shape (n_quantiles, n_draws), broadcast q_prior to (n_quantiles, n_draws)
            boot_distances = np.sum(np.abs(diff), axis=0) # shape (n_draws,)

            # save bootstrapped l1norm distances
            single_boot_distance_samples[p, c, :] = boot_distances

            # extract bootstrapped l1norm distances at the specified alpha-quantile
            single_boot_distance_alpha[p, c] = np.quantile(boot_distances, alpha)

    return single_observed_l1norm_distances, single_boot_distance_samples, single_boot_distance_alpha


def compute_ASL(
    l1norm_distance_observed: float, 
    l1norm_distance_bootstrapped: NDArray[np.float64]
    )-> NDArray[np.float64]:
    """
    For single parameter sensitivity, calculate the 1 - Achieved Significance Level (ASL), 
    which is the probability P(bootstrapped distances <= observed distance )

    Parameters
    ----------
    l1norm_distance_observed : float
        Observed l1norm distance

    l1norm_distance_bootstrapped : np.ndarray of shape (n_draws,)
        Bootstrapped l1norm distances

    Returns
    -------
    single_ASL_values : float
        1-ASL value
    """

    # distribution candidates for fitting
    fit_candidates = {"lognormal": lognorm, "gamma": gamma, "weibull": weibull_min, "beta": beta}
    # fit_candidates = {"lognormal": lognorm, "gamma": gamma, "weibull": weibull_min}

    # check for nans
    if np.isnan(l1norm_distance_observed) or np.all(np.isnan(l1norm_distance_bootstrapped)):
        ASL_value = np.nan

    # calculate the proportion of bootstrapped <= observed (empirical CDF)
    ASL_value = np.mean(l1norm_distance_bootstrapped <= l1norm_distance_observed)

    # extrapolate if observed > all bootstrapped
    if ASL_value == 1.0:
        # replace zeros with a very small number
        boot_nozero = np.where(l1norm_distance_bootstrapped == 0, np.finfo(float).eps, l1norm_distance_bootstrapped)

        # fit candidate distributions and pick the one with lowest AIC. Returns (best_name, best_params, best_dist).
        # initialized
        best_aic, best_fit  = np.inf, None

        for name, dist in fit_candidates.items():
            try:
                if name == 'beta':
                    boot_scaled = (boot_nozero - np.min(boot_nozero)) / (np.max(boot_nozero) - np.min(boot_nozero))
                    # fit distribution and compute log-likelihood
                    params = dist.fit(boot_scaled, floc=0, fscale=1)
                    loglik = np.sum(dist.logpdf(boot_scaled, *params))
                else: 
                    # fit distribution and compute log-likelihood
                    params = dist.fit(boot_nozero)
                    loglik = np.sum(dist.logpdf(boot_nozero, *params))

                k = len(params)  # number of parameters
                aic = 2 * k - 2 * loglik

                if aic < best_aic:
                    best_aic = aic
                    best_fit = (name, params, dist, np.min(boot_nozero), np.max(boot_nozero))
                    
            except Exception:
                continue

        if best_fit:
            name, params, dist, boot_min, boot_max = best_fit
            if name == 'beta':
                observed_scaled = (l1norm_distance_observed - boot_min) / (boot_max - boot_min)
                observed_scaled = np.clip(observed_scaled, 0, 1)
                ASL_value = dist.cdf(observed_scaled, *params)
            else:
                ASL_value = dist.cdf(l1norm_distance_observed, *params)

    return ASL_value
