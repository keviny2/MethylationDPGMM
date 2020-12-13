import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
from scipy.stats import norm, gamma, uniform, foldnorm
from scipy.special import loggamma
from copy import copy, deepcopy
from utils import get_normalized_p, one_hot_encoder, get_log_normalization_constant, get_normalized_p_and_log_norm, get_cluster_matrix
from distibutions.DirichletProcessDistribution import DirichletProcessDistribution
from distibutions.NormalGammaDistribution import NormalGammaDistribution

from optimization.numba_functions import\
    calculate_normal_gamma_normalization_constant_numba,\
    calculate_posterior_parameters_given_priors_numba, \
    calculate_posterior_parameters_given_priors_using_sufficient_statistics_numba, \
    calculate_log_integral_term_given_sufficient_statistics_numba, \
    calculate_log_integral_matrix_given_sufficient_statistics_numba,\
    calculate_log_integral_term_c_j_numba, \
    calculate_log_integral_matrix_c_j_numba


class DPGMM(object):

    def __init__(self,
                 data,
                 tissue_assignments,
                 v_init_type,
                 chi_init_type,
                 Z_init_type,
                 alphaDP_init_type,
                 sample_v_bool=True,
                 sample_chi_bool=True,
                 sample_Z_bool=True,
                 sample_alphaDP_bool=True,
                 sample_split_merge_bool=True,
                 plot_iter=0,
                 adapt_iter=1,
                 plot=False):

        # sampler settings

        self.mcmc_chain = []

        self.total_iter = 0

        self.sample_v_bool = copy(sample_v_bool)

        self.sample_chi_bool = copy(sample_chi_bool)

        self.sample_Z_bool = copy(sample_Z_bool)

        self.sample_split_merge_bool = copy(sample_split_merge_bool)

        self.sample_alphaDP_bool = copy(sample_alphaDP_bool)

        self.plot_iter = copy(plot_iter)

        # mcmc chains

        self.log_posterior_chain = []

        self.log_integral_chain = []

        self.v_chain = []

        self.chi_chain = []

        self.Z_chain = []

        self.K_chain = []

        self.alphaDP_chain = []

        # data

        self.data = deepcopy(data)

        self.T = deepcopy(tissue_assignments)

        self.N = data.shape[0]

        self.M = data.shape[1]

        self.L = tissue_assignments.shape[1]

        # model info

        self.cluster_to_index = None

        self.tissue_to_index = None

        self.index_to_tissue = None

        self.cluster_counts = None

        self.shifted_data = None

        self.sum_b = None

        self.sum_squared_b = None

        # model parameters

        self.K = 0

        self.v = None

        self.chi = None

        self.Z = None

        self.alphaDP = None

        # log likelihoods

        self.log_prior_v = 0

        self.log_prior_chi = 0

        self.log_prior_alphaDP = 0

        self.log_prior_Z = 0

        self.log_integral = 0  # integral that results from integrating out phi and tau

        self.log_integral_matrix = np.zeros(shape=(self.K, self.M))

        self.log_posterior = 0

        # init type

        self.v_init_type = v_init_type

        self.chi_init_type = chi_init_type

        self.Z_init_type = Z_init_type

        self.alphaDP_init_type = alphaDP_init_type

        # prior parameters

        self.v_prior = {'mean': np.zeros(shape=(self.M,)), 'variance': (np.ptp(self.data) / 4) * np.ones(shape=(self.M, ))}

        self.chi_prior = {'mean': np.zeros(shape=(self.L, self.M)), 'variance': (np.ptp(self.data) / 4) * np.ones(shape=(self.L, self.M))}

        self.base_distribution = {'mean': np.zeros(shape=(self.M, )), 'precision': 1. * np.ones(shape=(self.M, )), 'alpha': 2.0 * np.ones(shape=(self.M, )), 'beta': (np.ptp(self.data) / 10) * np.ones(shape=(self.M, ))}

        self.alphaDP_prior = {'alpha': 0.01, 'beta': 100.}

        # adaptive proposals

        self.adapt_parameters = adapt_iter > 0

        self.adapt_iter = copy(adapt_iter)

        self.num_accepted_v = np.zeros(shape=(self.M, ), dtype=np.int64)

        self.acceptance_rate_v = np.zeros(shape=(self.M, ), dtype=np.int64)

        self.num_accepted_chi = np.zeros(shape=(self.L, self.M), dtype=np.int64)

        self.accepted_rate_chi = np.zeros(shape=(self.L, self.M), dtype=np.int64)

        self.num_accepted_alphaDP = 0.

        self.acceptance_rate_alphaDP = 0.

        self.proposal_variance_v = 2 * np.ones(shape=(self.M,), dtype=np.float64)

        self.proposal_variance_chi = 2 * np.ones(shape=(self.L, self.M), dtype=np.float64)

        self.proposal_variance_alphaDP = np.array([0.5])

        self.burnin_epsilon = 0.01

        self.inference_epsilon = 0.01

        # plot data passed through the model and priors

        if plot:

            self.plot_data()

            self.plot_priors()

    # ======================================== initialize functions =============================================

    def initialize_model(self):

        self.initialize_parameters()

        self.initialize_cluster_tissue_to_index()

        self.initialize_shifted_data()

        self.initialize_sufficient_statistics()

        self.initialize_log_probs()

    def initialize_parameters(self):

        self.initialize_v()

        self.initialize_chi()

        self.initialize_alphaDP()

        self.initialize_Z()

    def initialize_v(self):

        if self.v_init_type == 'zeros':

            self.v = np.zeros(shape=(self.M, ))

        elif self.v_init_type == 'random':

            self.v = norm.rvs(loc=self.v_prior['mean'], scale=np.sqrt(self.v_prior['variance']))

    def initialize_chi(self):

        if self.chi_init_type == 'zeros':

            self.chi = np.zeros(shape=(self.L, self.M))

        if self.chi_init_type == 'random':

            self.chi = norm.rvs(loc=self.chi_prior['mean'], scale=np.sqrt(self.chi_prior['variance']))

        if self.chi_init_type == 'truth':

            self.chi = np.vstack((1. * np.ones(shape=(self.M, )), -1. * np.ones(shape=(self.M, ))))

    def initialize_alphaDP(self):

        if self.alphaDP_init_type == 'random':

            self.alphaDP = gamma.rvs(a=self.alphaDP_prior['alpha'], scale=self.alphaDP_prior['beta'])

        elif self.alphaDP_init_type == 'one':

            self.alphaDP = 1.

    def initialize_Z(self):

        if self.Z_init_type == 'singletons':

            Z = np.diag(np.ones(shape=(self.N,), dtype=np.int64))

        elif self.Z_init_type == 'random':

            Z = DirichletProcessDistribution.rvs(self.alphaDP, self.N)

        elif self.Z_init_type == 'one_cluster':

            Z = np.ones(shape=(self.N, 1))

        elif self.Z_init_type == 'two_clusters':

            K = 2

            N = self.N

            n1 = np.arange(start=0, stop=self.N, step=2)

            n2 = np.arange(start=1, stop=self.N, step=2)

            Z = np.zeros((N, K), dtype="int64")

            Z[n1, 0] += 1

            Z[n2, 1] += 1

        elif self.Z_init_type == 'three_clusters':

            K = 3

            N = self.N

            Z = np.zeros((N, K), dtype="int64")

            n1 = int(np.ceil(self.N / 2))

            n2 = int((self.N - n1) / 2)

            n3 = int(self.N - n1 - n2)

            Z[0:n2, 0] += 1

            Z[n2:(n2 + n3), 1] += 1

            Z[(n2 + n3):(n1 + n2 + n3), 2] += 1

        else:

            raise Exception("Z initialization not recognized")

        self.Z = Z

        self.K = self.Z.shape[1]

        self.cluster_counts = np.sum(self.Z, axis=0).tolist()

    def initialize_shifted_data(self):

        self.shifted_data = self.data - self.v - self.T @ self.chi

    def initialize_sufficient_statistics(self):

        self.sum_b = np.zeros(shape=(self.K, self.M), dtype=np.float64)

        self.sum_squared_b = np.zeros(shape=(self.K, self.M), dtype=np.float64)

        for c in range(self.K):

            self.sum_b[c] = np.sum(self.shifted_data[self.cluster_to_index[c]], axis=0)

            self.sum_squared_b[c] = np.sum(self.shifted_data[self.cluster_to_index[c]] ** 2, axis=0)

        assert(np.all(self.sum_squared_b >= 0))

    def initialize_cluster_tissue_to_index(self):

        cluster_to_index = []

        tissue_to_index = []

        index_to_tissue = []

        for t in range(self.L):

            tissue_to_index.append([])

        for c in range(self.K):

            cluster_to_index.append([])

        for i in range(self.N):

            c_i = np.where(self.Z[i] == 1)[0][0]

            t_i = np.where(self.T[i] == 1)[0][0]

            cluster_to_index[c_i].append(i)

            tissue_to_index[t_i].append(i)

            index_to_tissue.append(t_i)

        self.cluster_to_index = cluster_to_index

        self.tissue_to_index = tissue_to_index

        self.index_to_tissue = index_to_tissue

    def initialize_log_probs(self):

        self.log_integral_matrix = self.calculate_log_integral_matrix_c_j()

        self.log_integral_matrix[0][0] = np.mean(self.log_integral_matrix[0][1:])

        self.log_prior_v = np.sum(norm.logpdf(self.v, loc=self.v_prior['mean'], scale=np.sqrt(self.v_prior['variance'])))

        self.log_prior_chi = np.sum(norm.logpdf(self.chi, loc=self.chi_prior['mean'], scale=np.sqrt(self.chi_prior['variance'])))

        self.log_prior_alphaDP = gamma.logpdf(self.alphaDP, a=self.alphaDP_prior['alpha'], scale=self.alphaDP_prior['beta'])

        self.log_prior_Z = DirichletProcessDistribution.log_p(self.alphaDP, self.Z)

        self.log_integral = np.sum(self.log_integral_matrix)

        self.log_posterior = self.log_prior_v + self.log_prior_chi + self.log_prior_alphaDP + self.log_prior_Z + self.log_integral

    # ======================================= integral functions for calculating log posterior ======================================================

    def calculate_log_integral_matrix_c_j(self):

        return calculate_log_integral_matrix_c_j_numba(self.K, self.M, np.array(self.cluster_counts), self.sum_b, self.sum_squared_b, self.base_distribution['mean'], self.base_distribution['precision'], self.base_distribution['alpha'], self.base_distribution['beta'])

    def calculate_log_integral_term_c_j(self, c, j):

        return calculate_log_integral_term_c_j_numba(self.cluster_counts[c], self.sum_b[c, j], self.sum_squared_b[c, j], self.base_distribution['mean'][j], self.base_distribution['precision'][j], self.base_distribution['alpha'][j], self.base_distribution['beta'][j])

    def calculate_log_integral_matrix_given_sufficient_statistics(self, K, cluster_counts, sum_b, sum_squared_b):

        return calculate_log_integral_matrix_given_sufficient_statistics_numba(K, self.M, np.array(cluster_counts), sum_b, sum_squared_b, self.base_distribution['mean'], self.base_distribution['precision'], self.base_distribution['alpha'], self.base_distribution['beta'])

    def calculate_log_integral_term_given_sufficient_statistics(self, j, N, sum_data, sum_squared_data):

        return calculate_log_integral_term_given_sufficient_statistics_numba(N, sum_data, sum_squared_data, self.base_distribution['mean'][j], self.base_distribution['precision'][j], self.base_distribution['alpha'][j], self.base_distribution['beta'][j])

    @staticmethod
    def calculate_posterior_parameters_given_priors(data_points_idx, shifted_data, mu, kappa, alpha, beta):
        """
        numba requires to wrap a data_points_idx with a np.array()
        """

        return calculate_posterior_parameters_given_priors_numba(np.array(data_points_idx), shifted_data, mu, kappa, alpha, beta)

    @staticmethod
    def calculate_posterior_parameters_given_priors_using_sufficient_statistics(N, sum_b, sum_squared_b, mu, kappa, alpha, beta):

        return calculate_posterior_parameters_given_priors_using_sufficient_statistics_numba(N, sum_b, sum_squared_b, mu, kappa, alpha, beta)

    @staticmethod
    def calculate_normal_gamma_normalization_constant(kappa, alpha, beta, log_gamma_alpha):

        return calculate_normal_gamma_normalization_constant_numba(kappa, alpha, beta, log_gamma_alpha)

    # ======== integral function to calculate columns of log integral matrix that get changed when proposing new parameter values ===================

    def calculate_log_integral_column_v_with_sufficient_statistics(self, j, v_proposed):

        new_sum_b_column, new_sum_squared_b_column = self.calculate_new_sufficient_statistics_column_change_in_v(j, v_proposed, self.v[j])

        log_integral_column = np.zeros(shape=(self.K, ), dtype=np.float64)

        for c in range(self.K):

            log_integral_column[c] = self.calculate_log_integral_term_given_sufficient_statistics(j, self.cluster_counts[c], new_sum_b_column[c], new_sum_squared_b_column[c])

        return log_integral_column, new_sum_b_column, new_sum_squared_b_column

    def calculate_log_integral_column_v_with_sufficient_statistics_vectorized(self, v_proposed):

        new_sum_b, new_sum_squared_b = self.calculate_new_sufficient_statistics_column_change_in_v_vectorized(v_proposed, self.v)

        log_integral = self.calculate_log_integral_matrix_given_sufficient_statistics(self.K, self.cluster_counts, new_sum_b, new_sum_squared_b)

        return log_integral, new_sum_b, new_sum_squared_b

    def calculate_log_integral_column_chi_with_sufficient_statistics(self, t, j, chi_proposed):

        new_sum_b_column, new_sum_squared_b_column = self.calculate_new_sufficient_statistics_column_change_in_chi(t, j, chi_proposed, self.chi[t, j])

        log_integral_column = np.zeros(shape=(self.K, ), dtype=np.float64)

        for c in range(self.K):

            log_integral_column[c] = self.calculate_log_integral_term_given_sufficient_statistics(j, self.cluster_counts[c], new_sum_b_column[c], new_sum_squared_b_column[c])

        return log_integral_column, new_sum_b_column, new_sum_squared_b_column

    def calculate_new_sufficient_statistics_column_change_in_v(self, j, v_new, v_old):

        new_sum_b = copy(self.sum_b[:, j])

        new_sum_squared_b = copy(self.sum_squared_b[:, j])

        # values needed

        sum_y = np.array([np.sum(self.data[self.cluster_to_index[c], j]) for c in range(0, self.K)])

        sum_chi = np.array([np.sum(self.chi[np.array(self.index_to_tissue)[self.cluster_to_index[c]], j]) for c in range(0, self.K)])

        # update sum_b

        sum_b_change = np.array(self.cluster_counts) * (v_old - v_new)

        new_sum_b += sum_b_change

        # update sum_squared_b

        term0 = np.array(self.cluster_counts) * (v_new ** 2 - v_old ** 2)

        term1 = 2 * sum_y * (v_old - v_new)

        term2 = 2 * sum_chi * (v_new - v_old)

        sum_b_squared_change = term0 + term1 + term2

        new_sum_squared_b += sum_b_squared_change

        return new_sum_b, new_sum_squared_b

    def calculate_new_sufficient_statistics_column_change_in_v_vectorized(self, v_new, v_old):

        new_sum_b = copy(self.sum_b)

        new_sum_squared_b = copy(self.sum_squared_b)

        # values needed

        sum_y = np.array([np.sum(self.data[self.cluster_to_index[c]], axis=0) for c in range(self.K)])

        sum_chi = np.array([np.sum(self.chi[np.array(self.index_to_tissue)[self.cluster_to_index[c]]], axis=0) for c in range(self.K)])

        # update sum_b

        sum_b_change = np.outer(np.array(self.cluster_counts), (v_old - v_new))

        new_sum_b += sum_b_change

        # update sum_squared_b

        term0 = np.outer(np.array(self.cluster_counts), (v_new ** 2 - v_old ** 2))

        term1 = 2 * sum_y * (v_old - v_new)

        term2 = 2 * sum_chi * (v_new - v_old)

        sum_b_squared_change = term0 + term1 + term2

        new_sum_squared_b += sum_b_squared_change

        return new_sum_b, new_sum_squared_b

    def calculate_log_integral_column_chi_with_sufficient_statistics_vectorized(self, t, chi_proposed):

        new_sum_b, new_sum_squared_b = self.calculate_new_sufficient_statistics_column_change_in_chi_vectorized(t, chi_proposed, self.chi[t])

        log_integral_matrix = self.calculate_log_integral_matrix_given_sufficient_statistics(self.K, self.cluster_counts, new_sum_b, new_sum_squared_b)

        return log_integral_matrix, new_sum_b, new_sum_squared_b

    def calculate_new_sufficient_statistics_column_change_in_chi(self, t, j, chi_new, chi_old):

        new_sum_b_column = copy(self.sum_b[:, j])

        new_sum_b_squared_column = copy(self.sum_squared_b[:, j])

        # values needed

        intersections = [np.intersect1d(self.tissue_to_index[t], indices) for indices in self.cluster_to_index]

        N_ct = np.array([len(intersection) for intersection in intersections])

        sum_y_minus_v = np.array([np.sum(self.data[intersection, j] - self.v[j]) for intersection in intersections])

        # update sum_b

        sum_b_change = N_ct * (chi_old - chi_new)

        new_sum_b_column += sum_b_change

        # update sum_squared_b

        term0 = N_ct * (chi_new ** 2 - chi_old ** 2)

        term1 = 2 * sum_y_minus_v * (chi_old - chi_new)

        sum_b_squared_change = term0 + term1

        new_sum_b_squared_column += sum_b_squared_change

        # assert(np.all(new_sum_b_squared_column) > 0)

        return new_sum_b_column, new_sum_b_squared_column

    def calculate_new_sufficient_statistics_column_change_in_chi_vectorized(self, t, chi_new, chi_old):

        new_sum_b = copy(self.sum_b)

        new_sum_b_squared = copy(self.sum_squared_b)

        # values needed

        intersections = [np.intersect1d(self.tissue_to_index[t], indices) for indices in self.cluster_to_index]

        N_ct = np.array([len(intersection) for intersection in intersections])

        sum_y_minus_v = np.array([np.sum(self.data[intersection] - self.v, axis=0) for intersection in intersections])

        # update sum_b

        sum_b_change = np.outer(N_ct, (chi_old - chi_new))

        new_sum_b += sum_b_change

        # update sum_squared_b

        term0 = np.outer(N_ct, (chi_new ** 2 - chi_old ** 2))

        term1 = 2 * sum_y_minus_v * (chi_old - chi_new)

        sum_b_squared_change = term0 + term1

        new_sum_b_squared += sum_b_squared_change

        return new_sum_b, new_sum_b_squared

    # ================================================== integral functions for sample Z =================================================

    def calculate_log_integral_new(self, idx):

        mu_nj, kappa_nj, alpha_nj, beta_nj = self.calculate_posterior_parameters_given_priors([idx],
                                                                                              self.shifted_data,
                                                                                              self.base_distribution['mean'],
                                                                                              self.base_distribution['precision'],
                                                                                              self.base_distribution['alpha'],
                                                                                              self.base_distribution['beta'])

        term0 = -(1 / 2) * np.log(2 * np.pi)

        term1 = -self.calculate_normal_gamma_normalization_constant(self.base_distribution['precision'],
                                                                    self.base_distribution['alpha'],
                                                                    self.base_distribution['beta'],
                                                                    loggamma(self.base_distribution['alpha']))

        term2 = self.calculate_normal_gamma_normalization_constant(kappa_nj, alpha_nj, beta_nj, loggamma(alpha_nj))

        log_integral = np.sum(term0 + term1 + term2)

        return log_integral

    def calculate_log_integral_existing(self, idx, data_points):

        mu_0, kappa_0, alpha_0, beta_0 = self.calculate_posterior_parameters_given_priors(data_points,
                                                                                          self.shifted_data,
                                                                                          self.base_distribution['mean'],
                                                                                          self.base_distribution['precision'],
                                                                                          self.base_distribution['alpha'],
                                                                                          self.base_distribution['beta'])

        mu_n, kappa_n, alpha_n, beta_n = self.calculate_posterior_parameters_given_priors([idx],
                                                                                          self.shifted_data,
                                                                                          mu_0,
                                                                                          kappa_0,
                                                                                          alpha_0,
                                                                                          beta_0)

        term0 = -(1 / 2) * np.log(2 * np.pi)

        term1 = -self.calculate_normal_gamma_normalization_constant(kappa_0, alpha_0, beta_0, loggamma(alpha_0))

        term2 = self.calculate_normal_gamma_normalization_constant(kappa_n, alpha_n, beta_n, loggamma(alpha_n))

        log_integral_existing = np.sum(term0 + term1 + term2)

        return log_integral_existing

    # ================================ integrals functions for split merge =============================================

    def calculate_log_integral_split(self, S_j, c_i, Z_new):

        K_new, cluster_counts_new, sum_b_new, sum_squared_b_new = self.calculate_new_params_split(S_j, c_i, Z_new)

        log_integral_matrix = self.calculate_log_integral_matrix_given_sufficient_statistics(K_new, cluster_counts_new, sum_b_new, sum_squared_b_new)

        return log_integral_matrix, sum_b_new, sum_squared_b_new

    def calculate_log_integral_merge(self, c_i, c_j, Z_new):

        # we are merging two clusters c_i and c_j therefore
        # 1) update number of clusters k
        # 2) add all sufficient statistics to cluster row c_i in sufficient stat matrices
        # 3) delete cluster row for cluster c_j from sufficient stat matrices

        K_new = self.K - 1

        cluster_counts_new = np.sum(Z_new, axis=0, dtype=np.int64)

        sum_b_new = deepcopy(self.sum_b)

        sum_b_squared_new = deepcopy(self.sum_squared_b)

        sum_b_new[c_i] += sum_b_new[c_j]

        sum_b_squared_new[c_i] += sum_b_squared_new[c_j]

        sum_b_new = np.delete(sum_b_new, c_j, axis=0)

        sum_b_squared_new = np.delete(sum_b_squared_new, c_j, axis=0)

        log_integral_matrix = self.calculate_log_integral_matrix_given_sufficient_statistics(K_new, cluster_counts_new, sum_b_new, sum_b_squared_new)

        return log_integral_matrix, sum_b_new, sum_b_squared_new

    def calculate_new_params_split(self, S_j, c_i, Z_new):

        # we are splitting one cluster into two clusters c_i and c_j therefore
        # 1) increase number of clusters by 1
        # 2) create new sufficient stat matrices with a new cluster row for cluster c_j (we are using the existing row for data points in c_i)
        # 3) add sufficient stats from j and S_j to new cluster row
        # 4) delete sufficient stats attributed to data points j in S_j from c_i cluster row

        K_new = self.K + 1

        cluster_counts_new = np.sum(Z_new, axis=0)

        sum_b_new = np.vstack((self.sum_b, np.zeros(shape=(self.M,))))

        sum_squared_b_new = np.vstack((self.sum_squared_b, np.zeros(shape=(self.M,))))

        for k in S_j:

            sum_b_new[self.K] += self.shifted_data[k]

            sum_squared_b_new[self.K] += self.shifted_data[k] ** 2

            sum_b_new[c_i] -= self.shifted_data[k]

            sum_squared_b_new[c_i] -= self.shifted_data[k] ** 2

        return K_new, cluster_counts_new, sum_b_new, sum_squared_b_new

    # ======================================== sample Z functions =================================================

    def sample_Z(self):

        for i in range(self.N):

            self.sample_Z_gibbs(i)

        # get new fields

        log_integral_matrix_new = self.calculate_log_integral_matrix_c_j()

        log_Z_prior_diff = DirichletProcessDistribution.log_p(self.alphaDP, self.Z) - self.log_prior_Z

        log_integral_diff = np.sum(log_integral_matrix_new) - self.log_integral

        # update fields

        self.log_integral_matrix = log_integral_matrix_new

        self.log_posterior += log_Z_prior_diff + log_integral_diff

        self.log_prior_Z += log_Z_prior_diff

        self.log_integral += log_integral_diff

    def sample_Z_gibbs(self, idx):

        current_cluster = np.where(self.Z[idx] == 1)[0][0]

        singleton = self.get_singleton(current_cluster)

        other_clusters = self.get_other_clusters(current_cluster, singleton)

        new_log_probs = self.calculate_new_log_probs(idx)

        existing_log_probs = self.calculate_existing_log_probs(idx, other_clusters)

        p = get_normalized_p(np.append(existing_log_probs, new_log_probs))

        new_cluster = choice(len(other_clusters) + 1, p=p)

        self.update_fields_Z(idx, new_cluster, current_cluster, other_clusters, singleton)

    def calculate_existing_log_probs(self, idx, other_clusters):

        existing_log_probs = np.zeros(shape=(len(other_clusters), ))

        for i, c in enumerate(other_clusters):

            cond_cluster_to_index = self.conditional_cluster_to_index(idx, c)

            cond_count = len(cond_cluster_to_index)

            coefficient = np.log(cond_count) - np.log(self.alphaDP + self.N - 1)

            log_integral = self.calculate_log_integral_existing(idx, cond_cluster_to_index)

            existing_log_probs[i] = coefficient + log_integral

        return existing_log_probs

    def calculate_new_log_probs(self, idx):

        coefficient = np.log(self.alphaDP) - np.log(self.N - 1 + self.alphaDP)

        log_integral_new = coefficient + self.calculate_log_integral_new(idx)

        return log_integral_new

    def get_other_clusters(self, current_cluster, singleton):

        if singleton:

            other_clusters = np.delete(np.arange(self.K), current_cluster)

        else:

            other_clusters = np.arange(self.K)

        return other_clusters

    def get_singleton(self, cluster):

        return self.cluster_counts[cluster] - 1 == 0

    def conditional_cluster_to_index(self, idx, c):
        """
        calculates the data points in specified cluster c excluding a data point row_idx
        """
        if self.Z[idx, c] == 1:

            cond_cluster_to_index = copy(self.cluster_to_index[c])

            cond_cluster_to_index.remove(idx)

        else:

            cond_cluster_to_index = copy(self.cluster_to_index[c])

        return cond_cluster_to_index

    def update_fields_Z(self, idx, new_cluster, current_cluster, other_clusters, singleton):

        chose_existing = new_cluster < len(other_clusters)

        if chose_existing and singleton:

            new_cluster = other_clusters[new_cluster]

            self.case1(idx, new_cluster, current_cluster)

        elif chose_existing and not singleton:

            new_cluster = other_clusters[new_cluster]

            self.case2(idx, new_cluster, current_cluster)

        elif not chose_existing and singleton:

            self.case3(idx, current_cluster)

        elif not chose_existing and not singleton:

            self.case4(idx, current_cluster)

    def case1(self, idx, new_cluster, current_cluster):
        """
        updates fields when data point chose an existing cluster and was a singleton
        """
        # update self.cluster_to_index

        self.cluster_to_index[new_cluster].append(idx)

        self.cluster_to_index.pop(current_cluster)

        # update self.cluster_counts

        self.cluster_counts[new_cluster] += 1

        self.cluster_counts.pop(current_cluster)

        # update sufficient statistics

        self.sum_b[new_cluster] += self.shifted_data[idx]

        self.sum_squared_b[new_cluster] += self.shifted_data[idx] ** 2

        self.sum_b = np.delete(self.sum_b, current_cluster, axis=0)

        self.sum_squared_b = np.delete(self.sum_squared_b, current_cluster, axis=0)

        # update self.Z

        self.Z[idx] = one_hot_encoder(new_cluster, self.K)

        self.Z = np.delete(self.Z, current_cluster, axis=1)

        # update self.K

        self.K -= 1

        # check domain of sufficient stats

        # self.check_sufficient_stats(self.sum_squared_b)

    def case2(self, idx, new_cluster, current_cluster):
        """
        updates fields when data point chose an existing cluster and not a singleton
        """
        # update self.cluster_to_index

        self.cluster_to_index[new_cluster].append(idx)

        self.cluster_to_index[current_cluster].remove(idx)

        # update self.cluster_counts

        self.cluster_counts[new_cluster] += 1

        self.cluster_counts[current_cluster] -= 1

        # update sufficient statistics

        self.sum_b[new_cluster] += self.shifted_data[idx]

        self.sum_b[current_cluster] -= self.shifted_data[idx]

        self.sum_squared_b[new_cluster] += self.shifted_data[idx] ** 2

        self.sum_squared_b[current_cluster] -= self.shifted_data[idx] ** 2

        # update self.Z

        self.Z[idx] = one_hot_encoder(new_cluster, self.K)

        # check domain of sufficient stats

        # self.check_sufficient_stats(self.sum_squared_b)

    def case3(self, idx, current_cluster):
        """
        update fields when data point chose a new cluster and was a singleton
        """

        # update self.cluster_to_index

        self.cluster_to_index.append([idx])

        self.cluster_to_index.pop(current_cluster)

        # update self.cluster_counts

        self.cluster_counts.append(1)

        self.cluster_counts.pop(current_cluster)

        # update sufficient statistics

        self.sum_b = np.vstack((self.sum_b, np.zeros(shape=(self.M, ))))

        self.sum_squared_b = np.vstack((self.sum_squared_b, np.zeros(shape=(self.M,))))

        self.sum_b = np.delete(self.sum_b, current_cluster, axis=0)

        self.sum_squared_b = np.delete(self.sum_squared_b, current_cluster, axis=0)

        self.sum_b[self.K - 1] += self.shifted_data[idx]

        self.sum_squared_b[self.K - 1] += self.shifted_data[idx] ** 2

        # update self.Z

        self.Z = np.hstack((self.Z, np.zeros(shape=(self.N, 1))))

        self.Z[idx] = one_hot_encoder(self.K, self.K+1)

        self.Z = np.delete(self.Z, current_cluster, axis=1)

        # check domain of sufficient stats

        # self.check_sufficient_stats(self.sum_squared_b)

    def case4(self, idx, current_cluster):
        """
        updates fields when data point selected new cluster and was not a singleton
        """

        # update self.cluster_to_index

        self.cluster_to_index.append([idx])

        self.cluster_to_index[current_cluster].remove(idx)

        # update self.cluster_counts

        self.cluster_counts.append(1)

        self.cluster_counts[current_cluster] -= 1

        # update sufficient statistics

        self.sum_b = np.vstack((self.sum_b, np.zeros(shape=(self.M, ))))

        self.sum_squared_b = np.vstack((self.sum_squared_b, np.zeros(shape=(self.M,))))

        self.sum_b[self.K] += self.shifted_data[idx]

        self.sum_squared_b[self.K] += self.shifted_data[idx] ** 2

        self.sum_b[current_cluster] -= self.shifted_data[idx]

        self.sum_squared_b[current_cluster] -= self.shifted_data[idx] ** 2

        # update self.Z

        self.Z = np.hstack((self.Z, np.zeros(shape=(self.N, 1))))

        self.Z[idx] = one_hot_encoder(self.K, self.K+1)

        # update self.K

        self.K += 1

        # check domain of sufficient stats

        # self.check_sufficient_stats(self.sum_squared_b)

    # ======================================== split merge functions ==================================================

    def split_merge(self):
        """
        one iteration of split merge move as defined in paper ref
        Note: this method requires conjugacy of the likelihood with base distribution
        paper ref: "https://www.semanticscholar.org/paper/Sequentially-Allocated-Merge-Split-Sampler-for-and-Dahl/"
        """
        i, j = self.select_two_indices_at_random()

        c_i = np.where(self.Z[i] == 1)[0][0]

        c_j = np.where(self.Z[j] == 1)[0][0]

        S = self.get_other_data_points(i, j, c_i, c_j)

        if c_i == c_j:

            S_i, S_j, mh_log_probs = self.propose_split(i, j, S)

            self.mh_step_split(S_j, c_i, c_j, mh_log_probs)

        else:

            S_merge, mh_log_probs = self.propose_merge(i, j, S)

            self.mh_step_merge(c_i, c_j, mh_log_probs)

    def mh_step_split(self, S_j, c_i, c_j, mh_log_probs):

        Z_new, cluster_to_index_new = self.get_new_configuration_split(c_j, S_j)

        log_integral_matrix_proposed, sum_b_new, sum_b_squared_new = self.calculate_log_integral_split(S_j, c_i, Z_new)

        log_integral_diff = np.sum(log_integral_matrix_proposed) - self.log_integral

        log_prior_diff = DirichletProcessDistribution.log_p(self.alphaDP, Z_new) - self.log_prior_Z

        log_q_diff = np.log(1) - np.sum(mh_log_probs)

        accepted = self.mh_step(log_prior_diff, log_integral_diff, log_q_diff)

        if accepted:

            self.Z = Z_new

            self.cluster_to_index = cluster_to_index_new

            self.cluster_counts = np.sum(Z_new, axis=0, dtype=np.int64).tolist()

            self.K += 1

            self.log_posterior += log_integral_diff + log_prior_diff

            self.log_integral += log_integral_diff

            self.log_integral_matrix = log_integral_matrix_proposed

            self.log_prior_Z += log_prior_diff

            self.sum_b = sum_b_new

            self.sum_squared_b = sum_b_squared_new

            print("accepted split !!!")

    def mh_step_merge(self, c_i, c_j, mh_log_probs):

        Z_new, cluster_to_index_new = self.get_new_configuration_merge(c_i, c_j)

        log_integral_matrix_proposed, sum_b_new, sum_squared_b_new = self.calculate_log_integral_merge(c_i, c_j, Z_new)

        log_integral_diff = np.sum(log_integral_matrix_proposed) - self.log_integral

        log_prior_diff = DirichletProcessDistribution.log_p(self.alphaDP, Z_new) - self.log_prior_Z

        log_q_diff = np.sum(mh_log_probs) - np.log(1)

        accepted = self.mh_step(log_prior_diff, log_integral_diff, log_q_diff)

        if accepted:

            self.Z = Z_new

            self.cluster_to_index = cluster_to_index_new

            self.cluster_counts = np.sum(Z_new, axis=0, dtype=np.int64).tolist()

            self.K -= 1

            self.log_posterior += log_integral_diff + log_prior_diff

            self.log_integral += log_integral_diff

            self.log_prior_Z += log_prior_diff

            self.sum_b = sum_b_new

            self.sum_squared_b = sum_squared_b_new

            self.log_integral_matrix = log_integral_matrix_proposed

            print("accepted merge !!!")

    def propose_split(self, i, j, S):

        mh_log_probs = np.zeros(shape=(len(S),))

        S_i = [i]

        S_j = [j]

        for i, k in enumerate(S):

            new_cluster, log_prob = self.split_sample_new_cluster(k, S_i, S_j)

            mh_log_probs[i] = log_prob

            if new_cluster == 0:

                S_i.append(k)

            else:

                S_j.append(k)

        return S_i, S_j, mh_log_probs

    def propose_merge(self, i, j, S):

        mh_log_probs = self.mh_log_probs_merge(i, j, S)

        S_merge = copy(S)

        S_merge.append(i)

        S_merge.append(j)

        return S_merge, mh_log_probs

    def get_new_configuration_split(self, c_j, S_j):
        """
        changes Z matrix to split c_i and c_j.
        Notice that we only need j and S_j b/c we are keeping i and S_i in their current cluster
        and moving j and S_j to a new cluster.
        """

        Z_new = deepcopy(self.Z)

        cluster_to_index_new = deepcopy(self.cluster_to_index)

        Z_new = np.hstack((Z_new, np.zeros(shape=(self.N, 1))))

        cluster_to_index_new.append([])

        for k in S_j:

            Z_new[k] = one_hot_encoder(self.K, self.K + 1)

            cluster_to_index_new[c_j].remove(k)

            cluster_to_index_new[self.K].append(k)

        return Z_new, cluster_to_index_new

    def get_new_configuration_merge(self, c_i, c_j):
        """
        changes Z matrix to merge c_i and c_j
        notice that we only need j and S_j b/c we are moving them to cluter c_i
        """

        Z_new = deepcopy(self.Z)

        cluster_to_index_new = deepcopy(self.cluster_to_index)

        S_j = deepcopy(self.cluster_to_index[c_j])

        for k in S_j:

            Z_new[k] = one_hot_encoder(c_i, self.K)

            cluster_to_index_new[c_i].append(k)

        Z_new = np.delete(Z_new, c_j, axis=1)

        cluster_to_index_new.pop(c_j)

        return Z_new, cluster_to_index_new

    def split_sample_new_cluster(self, k, S_i, S_j):

        size_i = len(S_i)

        size_j = len(S_j)

        log_prob_i = self.calculate_log_integral_existing(k, S_i)

        log_prob_j = self.calculate_log_integral_existing(k, S_j)

        log_p = np.array([np.log(size_i) + log_prob_i, np.log(size_j) + log_prob_j])

        p, log_norm = get_normalized_p_and_log_norm(log_p)

        new_cluster = np.random.choice([0, 1], p=p)

        return new_cluster, log_p[new_cluster] - log_norm

    def mh_log_probs_merge(self, i, j, S_merge):

        S_i = [i]

        S_j = [j]

        q_merge = 0

        for k in np.random.permutation(S_merge):

            size_i = len(S_i)

            size_j = len(S_j)

            log_prob_i = self.calculate_log_integral_existing(k, S_i)

            log_prob_j = self.calculate_log_integral_existing(k, S_j)

            log_p = np.array([log_prob_i + np.log(size_i), log_prob_j + np.log(size_j)])

            log_norm = get_log_normalization_constant(log_p)

            if np.array_equal(self.Z[k], self.Z[i]):

                q_merge += log_prob_i - log_norm

                S_i.append(k)

            else:

                q_merge += log_prob_j - log_norm

                S_j.append(k)

        return q_merge

    def get_other_data_points(self, i, j, c_i, c_j):

        S = []

        for k in np.random.permutation(self.N):

            c_k = np.where(self.Z[k] == 1)[0][0]

            cond0 = k != i and k != j  # current data point is not equal to the ones randomly selected

            cond1 = c_k == c_i or c_k == c_j  # current data point is in a clusters with data points randomly selected

            if cond0 and cond1:

                S.append(k)

        return S

    def select_two_indices_at_random(self):

        i = np.random.randint(0, self.N)

        j = np.random.randint(0, self.N)

        while i == j:

            j = np.random.randint(0, self.N)

        return i, j

    # =================================== functions to sample parameters ===============================================

    def sample_v(self):

        for j in range(self.M):

            proposal_variance = self.proposal_variance_v[j]

            v_proposed = norm.rvs(loc=self.v[j], scale=np.sqrt(proposal_variance))

            v_diff = v_proposed - self.v[j]

            log_prior_diff = self.normal_log_pdf(v_proposed, self.v_prior['mean'][j], self.v_prior['variance'][j]) - self.normal_log_pdf(self.v[j], self.v_prior['mean'][j], self.v_prior['variance'][j])

            new_log_integral_column_j, new_sum_b_column, new_sum_squared_b_column = self.calculate_log_integral_column_v_with_sufficient_statistics(j, v_proposed)

            log_integral_diff = np.sum(new_log_integral_column_j) - np.sum(self.log_integral_matrix[:, j])  # changing v_j only changes the jth column of the log_integral_matrix

            accept = self.mh_step(log_prior_diff, log_integral_diff, 0)  # proposed is symmetric

            if accept:

                self.v[j] = v_proposed

                self.log_prior_v += log_prior_diff

                self.log_integral += log_integral_diff

                self.log_posterior += log_prior_diff + log_integral_diff

                self.shifted_data[:, j] += -v_diff

                self.log_integral_matrix[:, j] = new_log_integral_column_j

                self.sum_b[:, j] = new_sum_b_column

                self.sum_squared_b[:, j] = new_sum_squared_b_column

                self.num_accepted_v[j] += 1

                self.acceptance_rate_v[j] += 1

    def sample_v_vectorized(self):

        proposal_variance = self.proposal_variance_v

        v_proposed = norm.rvs(loc=self.v, scale=np.sqrt(proposal_variance))

        v_diff = v_proposed - self.v

        log_prior_diff = self.normal_log_pdf(v_proposed, self.v_prior['mean'], self.v_prior['variance']) - self.normal_log_pdf(self.v, self.v_prior['mean'], self.v_prior['variance'])

        new_log_integral_matrix, new_sum_b, new_sum_squared_b = self.calculate_log_integral_column_v_with_sufficient_statistics_vectorized(v_proposed)

        log_integral_diff = np.sum(new_log_integral_matrix, axis=0) - np.sum(self.log_integral_matrix, axis=0)  # changing v_j only changes the jth column of the log_integral_matrix

        accept = self.mh_step_vectorize(log_prior_diff, log_integral_diff, 0)  # proposed is symmetric

        self.v[accept] = v_proposed[accept]

        self.log_posterior += np.sum(log_prior_diff[accept] + log_integral_diff[accept])

        self.log_prior_v += np.sum(log_prior_diff[accept])

        self.log_integral += np.sum(log_integral_diff[accept])

        self.log_integral_matrix[:, accept] = new_log_integral_matrix[:, accept]

        self.sum_b[:, accept] = new_sum_b[:, accept]

        self.sum_squared_b[:, accept] = new_sum_squared_b[:, accept]

        self.shifted_data[:, accept] += -v_diff[accept]

        self.num_accepted_chi += accept

        self.accepted_rate_chi += accept

    def sample_chi(self):

        for t in range(self.L):

            for j in range(self.M):

                proposal_variance = self.proposal_variance_chi[t, j]

                chi_proposed = norm.rvs(loc=self.chi[t, j], scale=np.sqrt(proposal_variance))

                chi_diff = chi_proposed - self.chi[t, j]

                log_prior_diff = self.normal_log_pdf(chi_proposed, self.chi_prior['mean'][t, j], self.chi_prior['variance'][t, j]) - self.normal_log_pdf(self.chi[t, j], self.chi_prior['mean'][t, j], self.chi_prior['variance'][t, j])

                new_log_integral_column_j, new_sum_b_column, new_sum_squared_b_column = self.calculate_log_integral_column_chi_with_sufficient_statistics(t, j, chi_proposed)

                log_integral_diff = np.sum(new_log_integral_column_j) - np.sum(self.log_integral_matrix[:, j])

                accept = self.mh_step(log_prior_diff, log_integral_diff, 0)  # proposal is symmetric

                if accept:

                    self.chi[t, j] = chi_proposed

                    self.log_posterior += log_prior_diff + log_integral_diff

                    self.log_prior_chi += log_prior_diff

                    self.log_integral += log_integral_diff

                    self.shifted_data[self.tissue_to_index[t], j] += -chi_diff

                    self.log_integral_matrix[:, j] = new_log_integral_column_j

                    self.sum_b[:, j] = new_sum_b_column

                    self.sum_squared_b[:, j] = new_sum_squared_b_column

                    self.num_accepted_chi[t, j] += 1

                    self.accepted_rate_chi[t, j] += 1

    def sample_chi_vectorized(self):

        for t in range(self.L):

            proposal_variance = self.proposal_variance_chi[t]

            chi_proposed = norm.rvs(loc=self.chi[t], scale=np.sqrt(proposal_variance))

            chi_diff = chi_proposed - self.chi[t]

            log_prior_diff = self.normal_log_pdf(chi_proposed, self.chi_prior['mean'][t], self.chi_prior['variance'][t]) - self.normal_log_pdf(self.chi[t], self.chi_prior['mean'][t], self.chi_prior['variance'][t])

            new_log_integral_matrix, new_sum_b, new_sum_squared_b = self.calculate_log_integral_column_chi_with_sufficient_statistics_vectorized(t, chi_proposed)

            log_integral_diff = np.sum(new_log_integral_matrix, axis=0) - np.sum(self.log_integral_matrix, axis=0)

            accept = self.mh_step_vectorize(log_prior_diff, log_integral_diff, 0)

            self.chi[t, accept] = chi_proposed[accept]

            self.log_posterior += np.sum(log_prior_diff[accept] + log_integral_diff[accept])

            self.log_prior_chi = np.sum(log_prior_diff[accept])

            self.log_integral += np.sum(log_integral_diff[accept])

            self.log_integral_matrix[:, accept] = new_log_integral_matrix[:, accept]

            self.sum_b[:, accept] = new_sum_b[:, accept]

            self.sum_squared_b[:, accept] = new_sum_squared_b[:, accept]

            row_idx = np.array(self.tissue_to_index[t])

            col_idx = accept

            self.shifted_data[row_idx[:, np.newaxis], col_idx] += -chi_diff[accept]

            self.num_accepted_chi[t] += accept

            self.accepted_rate_chi[t] += accept

    def sample_alphaDP(self):

        proposal_variance = 10

        alpha_proposed = foldnorm.rvs(c=self.alphaDP/proposal_variance, scale=proposal_variance)

        log_prior_alphaDP_diff = self.gamma_log_pdf(alpha_proposed, self.alphaDP_prior['alpha'], self.alphaDP_prior['beta']) - self.gamma_log_pdf(self.alphaDP, self.alphaDP_prior['alpha'], self.alphaDP_prior['beta'])

        log_prior_Z_diff = DirichletProcessDistribution.log_p(alpha_proposed, self.Z) - DirichletProcessDistribution.log_p(self.alphaDP, self.Z)

        log_q_diff = foldnorm.logpdf(self.alphaDP, c=alpha_proposed/proposal_variance, scale=proposal_variance) - foldnorm.logpdf(alpha_proposed, c=self.alphaDP/proposal_variance, scale=proposal_variance)

        accept = self.mh_step(log_prior_alphaDP_diff, log_prior_Z_diff, log_q_diff)

        if accept:

            self.alphaDP = alpha_proposed

            self.log_posterior += log_prior_alphaDP_diff + log_prior_Z_diff

            self.log_prior_alphaDP += log_prior_alphaDP_diff

            self.log_prior_Z += log_prior_Z_diff

            self.num_accepted_alphaDP += 1

            self.acceptance_rate_alphaDP += 1

    # ========================================= log pdf functions ==============================================================

    def normal_log_pdf(self, val, mean, variance):

        if np.any(variance) <= 0:

            raise ValueError("Tried Pass through a variance that is less than or equal to 0 for gene {} at iteration {} ")

        return -(1/2) * np.log(2*np.pi) + -np.log(np.sqrt(variance)) + -(1/(2 * variance)) * (val - mean) ** 2

    def gamma_log_pdf(self, val, shape, scale):

        if shape < 0:

            raise ValueError("Tried to pass through a shape parameter that is <= 0")

        if scale < 0:

            raise ValueError("Tried to pass through a scale parameter that is <= to 0")

        # if np.any(val <= 1e-30):
        #
        #     return -np.inf
        #
        # else:

        return -loggamma(shape) + -shape*np.log(scale) + (shape-1)*np.log(val) + -val/scale

    # ======================================== inference functions =====================================================

    def mcmc(self, num_burnin, num_inference):

        print(10 * "=", "Performing Burnin", 10 * "=")

        for _ in range(num_burnin):

            self.print_model_info()

            self.inference()

            if self.adapt_parameters and (self.total_iter % self.adapt_iter == 0):

                self.adapt_proposals()

        print(10 * "=", "Performing Inference", 10 * "=")

        self.clear_chains()

        for _ in range(num_inference):

            self.print_model_info()

            self.inference()

            if self.adapt_parameters and (self.total_iter % self.adapt_iter == 0):

                self.adapt_proposals()

        self.mcmc_chain = {'log_posterior': self.log_posterior_chain,
                           'log_integral': self.log_integral_chain,
                           'v': self.v_chain,
                           'chi': self.chi_chain,
                           'alphaDP': self.alphaDP_chain,
                           'Z': self.Z_chain,
                           'K': self.K_chain,
                           'acceptance_rate_v': self.acceptance_rate_v / self.total_iter,
                           'acceptance_rate_chi': self.accepted_rate_chi / self.total_iter,
                           'acceptance_rate_alphaDP': self.acceptance_rate_alphaDP / self.total_iter}

    def inference(self):

        if self.sample_Z_bool:

            self.sample_Z()

        if self.sample_split_merge_bool:

            self.split_merge()

        if self.sample_v_bool:

            # self.sample_v()

            self.sample_v_vectorized()

        if self.sample_chi_bool:

            self.sample_chi_vectorized()

            # self.sample_chi()

        if self.sample_alphaDP_bool:

            self.sample_alphaDP()

        # self.check_sufficient_stats(self.sum_squared_b)

        self.store_iter()

        self.total_iter += 1

    def mh_step(self, log_prior_diff, log_integral_diff, log_q_diff):

        log_alpha = log_prior_diff + log_integral_diff + log_q_diff

        u = np.log(uniform.rvs())

        accept = False

        if u < log_alpha:

            accept = True

        return accept

    def mh_step_vectorize(self, log_prior_diff, log_integral_diff, log_q_diff):

        log_alpha = log_prior_diff + log_integral_diff + log_q_diff

        u = np.log(uniform.rvs(size=log_alpha.shape[0]))

        accept = u < log_alpha

        return accept

    # ======================================== adaptive proposal functions =============================================

    def adapt_proposals(self, burnin=False):

        self.adapt_v_proposals(burnin)

        self.adapt_chi_proposals(burnin)

        self.adapt_alphaDP_proposals(burnin)

    def adapt_v_proposals(self, burnin):

        acceptance_rate = self.num_accepted_v / self.adapt_iter

        new_proposals = self.calculate_new_proposals(acceptance_rate, self.proposal_variance_v, burnin)

        self.proposal_variance_v = new_proposals

        self.num_accepted_v = np.zeros(shape=(self.M, ), dtype=np.int64)

    def adapt_chi_proposals(self, burnin):

        acceptance_rate = self.num_accepted_chi / self.adapt_iter

        new_proposals = self.calculate_new_proposals(acceptance_rate, self.proposal_variance_chi, burnin)

        self.proposal_variance_chi = new_proposals

        self.num_accepted_chi = np.zeros(shape=(self.L, self.M))

    def adapt_alphaDP_proposals(self, burnin):

        acceptance_rate = self.num_accepted_alphaDP / self.adapt_iter

        new_proposals = self.calculate_new_proposals(acceptance_rate, self.proposal_variance_alphaDP, burnin)

        self.proposal_variance_alphaDP = new_proposals

        self.num_accepted_alphaDP = 0

    def calculate_new_proposals(self, acceptance_rate, current_proposals, burnin):

        if burnin:

            epsilon = min(self.burnin_epsilon, 1 / np.sqrt(self.total_iter))

        else:

            epsilon = min(self.inference_epsilon, 1 / np.sqrt(self.total_iter))

        low_indices = acceptance_rate < 0.44

        log_new_proposals = np.log(current_proposals) + (-2 * low_indices + 1) * epsilon

        new_proposals = np.exp(log_new_proposals)

        # new_proposals = current_proposals + (-2 * low_indices + 1) * epsilon

        ceiling = new_proposals > 5

        floor = new_proposals < 0.01

        new_proposals[ceiling] = 5.

        new_proposals[floor] = 0.01

        return new_proposals

    # ========================================= storage and util functions ==============================================

    def store_iter(self):

        self.log_posterior_chain.append(self.log_posterior)

        self.log_integral_chain.append(self.log_integral)

        self.v_chain.append(self.v)

        self.chi_chain.append(self.chi)

        self.alphaDP_chain.append(self.alphaDP)

        self.Z_chain.append(self.Z)

        self.K_chain.append(self.K)

    def print_model_info(self):

        print("Iteration: {}".format(self.total_iter))

        print("Log Posterior: {}".format(self.log_posterior))

        print("Cluster Counts: {}".format(self.cluster_counts))

        if (self.plot_iter != 0) and (self.total_iter % self.plot_iter == 0):

            plt.matshow(self.shifted_data)

            plt.title("Shifted Data at Iteration {}".format(self.total_iter))

            # plt.clim(vmin=-5, vmax=5)

            plt.colorbar()

            plt.show()

    def clear_chains(self):

        self.log_posterior_chain = []

        self.log_integral_chain = []

        self.v_chain = []

        self.chi_chain = []

        self.alphaDP_chain = []

        self.Z_chain = []

        self.K_chain = []

    # ========================================= assertion statement functions ===============================================

    def check_sufficient_stats(self, sum_squared_b):
        """
        assertion statements to ensure that sufficient statistics are in proper domain
        """

        assert np.all(sum_squared_b >= 0.), "Assertion failed at iteration: {}".format(self.total_iter)

    def check_sample_chi(self, t, j, new_sum_b_column, new_sum_squared_b_column, new_log_integral_column_j, new_sum_b, new_sum_squared_b, new_log_integral_column_j_test):

        assert np.all(np.round(new_sum_b_column, 3) == np.round(new_sum_b[:, j], 3)), "iteration {} tissue {} gene {}".format(self.total_iter, t, j)

        assert np.all(np.round(new_sum_squared_b_column, 3) == np.round(new_sum_squared_b[:, j], 3)), "iteration {} tissue {} gene {}".format(self.total_iter, t, j)

        assert np.all(np.round(new_log_integral_column_j, 3) == np.round(new_log_integral_column_j_test, 3)), "iteration {} tissue {} gene {}".format(self.total_iter, t, j)

    def check_sample_v(self, j, new_sum_b_column, new_sum_squared_b_column,  new_log_integral_column_j, new_sum_b, new_sum_squared_b, new_log_integral_column_j_test_test):

        assert (np.all(np.round(new_sum_b_column, 3) == np.round(new_sum_b[:, j], 3)))

        assert (np.all(np.round(new_sum_squared_b_column, 3) == np.round(new_sum_squared_b[:, j], 3)))

        assert (np.all(np.round(new_log_integral_column_j, 3) == np.round(new_log_integral_column_j_test_test, 3)))

    # ========================================= posterior inference on phi and tau parameters ==================================

    def posterior_cluster_parameters(self, cluster_assignments, cluster_to_index, cluster_counts, v, chi, absolute_path):
        """
        calculates mean of posterior cluster parameters conditioned on cluster assignments, v, and chi
        """

        Z = get_cluster_matrix(cluster_assignments)

        sum_b_posterior, sum_squared_b_posterior = self.calculate_posterior_sufficient_statistics(cluster_assignments, v, chi, cluster_to_index)

        K = np.max(cluster_assignments) + 1

        phi = np.zeros(shape=(K, self.M), dtype=np.float64)

        tau = np.zeros(shape=(K, self.M), dtype=np.float64)

        for c in range(K):

            n_c = cluster_counts[c]

            for j in range(self.M):

                mu_cj, lambda_c, alpha_c, beta_cj = self.calculate_posterior_parameters_given_priors_using_sufficient_statistics(n_c,
                                                                                                                                 sum_b_posterior[c, j],
                                                                                                                                 sum_squared_b_posterior[c, j],
                                                                                                                                 self.base_distribution['mean'][j],
                                                                                                                                 self.base_distribution['precision'][j],
                                                                                                                                 self.base_distribution['alpha'][j],
                                                                                                                                 self.base_distribution['beta'][j])

                phi[c, j] = mu_cj

                tau[c, j] = alpha_c / beta_cj

        if absolute_path is not None:

            plt.matshow(phi)

            plt.title("Inferred Phi")

            plt.colorbar()

            plt.savefig(absolute_path + str('/plots') + str('/inferred_phi.png'))

            plt.show()

            plt.matshow(tau)

            plt.title("Inferred Tau")

            plt.colorbar()

            plt.savefig(absolute_path + str('/plots') + str('/inferred_tau.png'))

            plt.show()

            plt.matshow(Z @ phi + self.T @ chi + v)

            plt.title("Inferred Mean")

            plt.colorbar()

            plt.savefig(absolute_path + str('/plots') + str('/inferred_mean.png'))

            plt.show()

        return phi, tau

    def calculate_posterior_sufficient_statistics(self, cluster_assignments, v, chi, cluster_to_index):

        K = np.max(cluster_assignments) + 1

        sum_b = np.zeros(shape=(K, self.M), dtype=np.float64)

        sum_squared_b = np.zeros(shape=(K, self.M), dtype=np.float64)

        for c in range(K):

            cluster_of_data_points = cluster_to_index[c]

            tissue_of_data_points = np.array(self.index_to_tissue)[cluster_of_data_points]

            for j in range(self.M):

                # TODO: double check these calculation are correct

                sum_b[c, j] = np.sum(self.data[cluster_of_data_points, j] - v[0][j] - chi[tissue_of_data_points, j])

                sum_squared_b[c, j] = np.sum((self.data[cluster_of_data_points, j] - v[0][j] - chi[tissue_of_data_points, j]) ** 2)

        return sum_b, sum_squared_b

    # =============================== Geweke Test for Sampler Correctness ======================================================

    def geweke_test(self):

        # forward samples

        raise NotImplementedError

    def forward_simulate(self):

        raise NotImplementedError

    def plot_results(self):

        raise NotImplementedError

    def calculate_statistics(self):

        raise NotImplementedError

    # =============================== Plot Data and Prior Distributions =======================================================

    def plot_data(self):

        plt.matshow(self.data)

        plt.title("Data passed through model")

        plt.colorbar()

        plt.show()

    def plot_priors(self):

        # plot normal gamma

        normal_gamma_dist = NormalGammaDistribution(self.base_distribution['mean'][0], self.base_distribution['precision'][0], self.base_distribution['alpha'][0], self.base_distribution['beta'][0])

        normal_gamma_dist.plot(-5, 5, 0, 10)

        chi = np.linspace(-5, 5, 100)

        v = np.linspace(-5, 5, 100)

        M, T = np.meshgrid(chi, v, indexing="ij")

        Z = np.zeros_like(M)

        for i in range(Z.shape[0]):

            for j in range(Z.shape[1]):

                Z[i][j] = norm.pdf(chi[i], self.chi_prior['mean'][0][0], self.chi_prior['variance'][0][0]) * norm.pdf(v[j], self.v_prior['mean'][0], self.v_prior['variance'][0])

        plt.contourf(M, T, Z)

        plt.title("chi and v prior distribution")

        plt.xlabel("chi")

        plt.ylabel("v")

        plt.colorbar()

        plt.show()


