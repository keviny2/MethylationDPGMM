import numpy as np
from math import lgamma
from numba import jit


@jit(nopython=True)
def calculate_posterior_parameters_given_priors_numba(data_points_idx, shifted_data, mu, kappa, alpha, beta):

    N = len(data_points_idx)

    data = shifted_data[data_points_idx]

    sum_data = np.sum(data, axis=0)  # sufficient stat 1

    sum_squared_data = np.sum(data ** 2, axis=0)  # sufficient stat 2

    mu_n = (kappa * mu + sum_data) / (kappa + N)

    kappa_n = kappa + N

    alpha_n = alpha + N / 2

    beta_n = beta + (kappa * N) * (sum_data / N - mu) ** 2 / (2 * (kappa + N)) + (1 / 2) * (sum_squared_data - N * (sum_data / N) ** 2)

    return mu_n, kappa_n, alpha_n, beta_n


@jit(nopython=True)
def calculate_normal_gamma_normalization_constant_numba(kappa, alpha, beta, log_gamma_alpha):
    """
    pass log_gamma_alpha through instead of alpha b/c math.lgamma only takes floats and scipy.loggamma is not recognized by numba
    """

    term0 = log_gamma_alpha

    term1 = -alpha * np.log(beta)

    term2 = (1 / 2) * np.log(2 * np.pi)

    term3 = -(1 / 2) * np.log(kappa)

    return term0 + term1 + term2 + term3


@jit(nopython=True)
def calculate_posterior_parameters_given_priors_using_sufficient_statistics_numba(N, sum_b, sum_squared_b, mu, kappa, alpha, beta):

    sum_data = sum_b

    sum_squared_data = sum_squared_b

    mu_n = (kappa * mu + sum_data) / (kappa + N)

    kappa_n = kappa + N

    alpha_n = alpha + N / 2

    beta_n = beta + (kappa * N) * (sum_data / N - mu) ** 2 / (2 * (kappa + N)) + (1 / 2) * (sum_squared_data - N * (sum_data / N) ** 2)

    return mu_n, kappa_n, alpha_n, beta_n


@jit(nopython=True)
def calculate_log_integral_term_given_sufficient_statistics_numba(N, sum_data, sum_squared_data, base_mean, base_precision, base_alpha, base_beta):
    """
    calculates integral of product_{i} N(y_ij | mu, lambda) x NG(mu, lambda | mu_0j, kappa_0j, alpha_0j, beta_0j) where i in data_points_idx
    calculates same value as calculate_log_integral_term except with sufficient stats

    note we are not passing through array therefore we can use lgamma from the math package
    """

    # posterior parameters

    mu_n, kappa_n, alpha_n, beta_n = calculate_posterior_parameters_given_priors_using_sufficient_statistics_numba(N,
                                                                                                                   sum_data,
                                                                                                                   sum_squared_data,
                                                                                                                   base_mean,
                                                                                                                   base_precision,
                                                                                                                   base_alpha,
                                                                                                                   base_beta)

    term0 = -(N / 2) * np.log(2 * np.pi)

    term1 = -calculate_normal_gamma_normalization_constant_numba(base_precision, base_alpha, base_beta, lgamma(base_alpha))

    term2 = calculate_normal_gamma_normalization_constant_numba(kappa_n, alpha_n, beta_n, lgamma(alpha_n))

    return term0 + term1 + term2


@jit(nopython=True)
def calculate_log_integral_matrix_given_sufficient_statistics_numba(K, M, cluster_counts, sum_b, sum_squared_b, base_mean, base_precision, base_alpha, base_beta):
    """
    calculates the log integral resulting from integrating out cluster means and cluster variances given v and chi
    This is where we leverage conjugacy and independence
    - independence breaks down the multidimensional integral into product of one dimensional integrals
    - conjugacy allows us to have an analytical solution to these one dimensional integrals
    """

    log_integral_matrix = np.zeros(shape=(K, M))

    for c in range(K):

        for j in range(M):

            log_integral_matrix[c, j] = calculate_log_integral_term_given_sufficient_statistics_numba(cluster_counts[c],
                                                                                                      sum_b[c, j],
                                                                                                      sum_squared_b[c, j],
                                                                                                      base_mean[j],
                                                                                                      base_precision[j],
                                                                                                      base_alpha[j],
                                                                                                      base_beta[j])

    return log_integral_matrix


@jit(nopython=True)
def calculate_log_integral_term_c_j_numba(N, sum_b, sum_b_squared, base_mean, base_precision, base_alpha, base_beta):

    mu_cj, kappa_cj, alpha_cj, beta_cj = calculate_posterior_parameters_given_priors_using_sufficient_statistics_numba(N,
                                                                                                                       sum_b,
                                                                                                                       sum_b_squared,
                                                                                                                       base_mean,
                                                                                                                       base_precision,
                                                                                                                       base_alpha,
                                                                                                                       base_beta)

    term0 = -(N/2) * np.log(2*np.pi)

    term1 = -calculate_normal_gamma_normalization_constant_numba(base_precision, base_alpha, base_beta, lgamma(base_alpha))

    term2 = calculate_normal_gamma_normalization_constant_numba(kappa_cj, alpha_cj, beta_cj, lgamma(alpha_cj))

    return term0 + term1 + term2


@jit(nopython=True)
def calculate_log_integral_matrix_c_j_numba(K, M, cluster_counts, sum_b, sum_b_squared, base_mean, base_precision, base_alpha, base_beta):

    log_integral_matrix = np.zeros(shape=(K, M))

    for c in range(K):

        for j in range(M):

            log_integral_matrix[c, j] = calculate_log_integral_term_c_j_numba(cluster_counts[c],
                                                                              sum_b[c, j],
                                                                              sum_b_squared[c, j],
                                                                              base_mean[j],
                                                                              base_precision[j],
                                                                              base_alpha[j],
                                                                              base_beta[j])

    return log_integral_matrix
