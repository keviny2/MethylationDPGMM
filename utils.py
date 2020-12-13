import numpy as np
import pandas as pd
import numba
import time
import matplotlib.pyplot as plt
import os

from scipy.special import binom
from scipy.cluster.hierarchy import cut_tree
from sklearn.decomposition import PCA


# ======================================== UncollapsedNormalModel utils ================================================= #
from sklearn import preprocessing


def reorder(arr, index):
    n = len(arr)
    temp = [0] * n;

    # arr[i] should be
    # present at index[i] index
    for i in range(0, n):
        temp[index[i]] = arr[i]

    # Copy temp[] to arr[]
    for i in range(0, n):
        arr[i] = temp[i]
        # index[i] = i


def one_hot_encoder(cluster, num_clusters):

    vec = np.zeros(shape=(num_clusters,))

    vec[cluster] = 1

    return vec


def get_cluster_matrix(cluster_labels):
    """
    :param cluster_labels: np.array([1,1,2])
    :return: np.array([[1,0,0], [1,0,0], [0,1,0]]
    """

    while not np.any(cluster_labels == 0):

        cluster_labels -= 1

    N = cluster_labels.size

    K = np.unique(cluster_labels).size

    Z = np.zeros(shape=(N, K))

    for row_idx in range(N):

        cluster = int(cluster_labels[row_idx])

        Z[row_idx][cluster] = 1

    return Z.astype(np.int64)


# @numba.njit(cache=True)
def get_normalized_p(log_p):
    """
    :param log_p: np.array([-1.61, -0.36, -3.00, -3.22, -4.61])
    :return: np.array([0.2, 0.7, 0.05, 0.04, 0.01])
    """

    log_C = np.logaddexp(log_p[0], log_p[1])

    for i in range(len(log_p) - 2):

        log_C = np.logaddexp(log_C, log_p[i+2])

    log_p = log_p - log_C

    p = np.exp(log_p)

    # rounded_p = np.round(p, decimals=3)

    return p


def get_normalized_p_and_log_norm(log_p):

    log_norm = get_log_normalization_constant(log_p)

    p = np.exp(log_p - log_norm)

    p = p / p.sum()

    return p, log_norm


def get_log_normalization_constant(log_X):
    '''
    Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    Numerically safer than naive method.
    Andy's code pgsm
    '''
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.njit(cache=True)
def get_singleton(Z, idx):

    m = get_conditional_counts(Z, idx)

    singleton = np.any(m == 0)

    return singleton


@numba.njit(cache=True)
def get_conditional_counts(Z, row_idx):

    m = np.sum(Z, axis=0)

    m -= Z[row_idx]

    return m

# ============================================= post processing utils ============================================== #


# @numba.njit(cache=True)
def array_to_int(curr_cluster_vec):
    """
    :param curr_cluster_vec: np.array([0, 0, 1])
    :return: 2
    """

    return np.squeeze(np.where(curr_cluster_vec == 1))


def get_cluster_array(cluster_matrix):
    """
    :param cluster_matrix: np.array([[1,0,0], [1,0,0], [0,1,0]]
    :return: np.array([1,1,2])
    """

    N = cluster_matrix.shape[0]

    cluster_assignments = np.zeros(shape=(N, ), dtype=np.int64)

    for i in range(N):

        cluster_assignments[i] = array_to_int(cluster_matrix[i])

    return cluster_assignments


def compute_mpear(cluster_labels, sim_mat):

    N = sim_mat.shape[0]

    ind_mat = get_indicator_matrix(cluster_labels)

    i_s = np.tril(ind_mat * sim_mat, k=-1).sum()

    i = np.tril(ind_mat, k=-1).sum()

    s = np.tril(sim_mat, k=-1).sum()

    c = binom(N, 2)

    z = (i * s) / c

    num = i_s - z

    den = 0.5 * (i + s) - z

    return num / den


@numba.jit(cache=True, nopython=True)
def get_indicator_matrix(cluster_labels):

    N = len(cluster_labels)

    I = np.zeros(shape=(N, N))

    for i in range(N):

        for j in range(i):

            if cluster_labels[i] == cluster_labels[j]:

                I[i, j] = 1

            else:

                I[i, j] = 0

    return I + I.T


def get_flat_clustering(Z, number_of_clusters):

    N = len(Z) + 1

    if number_of_clusters == N:

        return np.arange(1, N + 1)

    return np.squeeze(cut_tree(Z, n_clusters=number_of_clusters))

# ============================================== get data utils  =====================================================


def get_tumour_data(num_patients_per_tissue, num_genes, tumour_locations, gene_selection_type='variance', log_normalize=False, scale=1, plot=False, seed=123):

    num_tumours = tumour_locations.shape[0]

    subset_data, T, tissue_location_subset, sample_ids, gene_ids = get_subset_of_data(tumour_locations, num_tumours, num_patients_per_tissue, num_genes, gene_selection_type, seed)

    if log_normalize:

        subset_data = log_z_score_data(subset_data, scale=scale)  # scale data to avoid underflow for variance parameter

    if plot:

        plt.matshow(subset_data)

        plt.title("Data")

        plt.colorbar()

        plt.show()

    return {'data': subset_data, 'T': T}


def get_subset_of_data_methylation(tissue_locations, num_tissues, num_patients, num_genes, gene_selection_type, specified_genes, random_seed, scale=1, log_normalize_type='log_normalize', plot=False, joint=False):

    # import small tumour data set, get patient, get gene ids, get tissue locations

    expression_matrix, sample_ids, gene_ids, tissue_location_data = load_data_methylation(joint)

    total_num_genes = expression_matrix.shape[1]

    tissue_to_patient_index = get_tissue_to_patient_index(num_tissues, tissue_locations, tissue_location_data)

    subset_patients = get_random_subset_of_patients(num_tissues, num_patients, tissue_to_patient_index, random_seed)

    subset_data = expression_matrix[subset_patients.flatten()]

    subset_genes = get_subset_of_genes(gene_selection_type, subset_data, num_genes, total_num_genes, specified_genes, gene_ids)

    if gene_selection_type == 'pca':

        subset_data = subset_genes

        subset_gene_ids = np.array(['PCA' + str(i+1) for i in range(num_genes)])

    else:

        subset_data = subset_data[:, subset_genes]

        subset_gene_ids = gene_ids[0][subset_genes]

    subset_tissue = tissue_location_data[0][subset_patients.flatten()]

    tissue_matrix = get_tissue_matrix(num_patients, num_tissues, subset_tissue)

    subset_sample_ids = sample_ids[0][subset_patients.flatten()]

    if log_normalize_type is not None:

        subset_data = log_z_score_data(subset_data, scale=scale, log_normalize_type=log_normalize_type)  # scale data to avoid underflow for variance parameter

    if plot:

        plt.matshow(subset_data)

        plt.title("Data")

        plt.colorbar()

        plt.show()

    return subset_data, tissue_matrix, subset_tissue, subset_sample_ids, subset_gene_ids


def get_subset_of_data(tissue_locations, num_tissues, num_patients, num_genes, gene_selection_type, specified_genes, random_seed, scale=1, log_normalize_type='log_normalize', plot=False):

    # import small tumour data set, get patient, get gene ids, get tissue locations

    expression_matrix, sample_ids, gene_ids, tissue_location_data = load_data_rna('seq')

    # expression_matrix, sample_ids, gene_ids, tissue_location_data, subset_survival = load_data_from_numpy()

    total_num_genes = expression_matrix.shape[1]

    tissue_to_patient_index = get_tissue_to_patient_index(num_tissues, tissue_locations, tissue_location_data)

    subset_patients = get_random_subset_of_patients(num_tissues, num_patients, tissue_to_patient_index, random_seed)

    subset_data = expression_matrix[subset_patients.flatten()]

    subset_genes = get_subset_of_genes(gene_selection_type, subset_data, num_genes, total_num_genes, specified_genes, gene_ids)

    subset_data = subset_data[:, subset_genes]

    subset_tissue = tissue_location_data[0][subset_patients.flatten()]

    tissue_matrix = get_tissue_matrix(num_patients, num_tissues, subset_tissue)

    subset_sample_ids = sample_ids[0][subset_patients.flatten()]

    subset_gene_ids = gene_ids[0][subset_genes]

    # subset_survival = get_survival_subset(sample_to_survival, subset_sample_ids)

    if log_normalize_type is not None:

        subset_data = log_z_score_data(subset_data, scale=scale, log_normalize_type=log_normalize_type)  # scale data to avoid underflow for variance parameter

    if plot:

        plt.matshow(subset_data)

        plt.title("Data")

        plt.colorbar()

        plt.show()

    return subset_data, tissue_matrix, subset_tissue, subset_sample_ids, subset_gene_ids#, subset_survival


def load_data_methylation(joint):

    dirpath = os.path.dirname(__file__)

    if joint:
        expression_matrix = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/meth/meth_array.csv')).to_numpy()

        sample_ids = np.transpose(pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/meth/icgc_donor_id.csv')).to_numpy())

        gene_ids = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/meth/gene_id.csv')).dropna()

        gene_ids = np.transpose(gene_ids.to_numpy())

        tissue_location_data = np.transpose(pd.read_csv(os.path.join(dirpath,
                                                                     'data/icgc/methylation/meth_rna_joint/meth/tissue_location_data.csv')).to_numpy())

    else:

        expression_matrix = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/ovarian_breast_colorectal/ov_br_col.csv')).to_numpy()

        sample_ids = np.transpose(pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/ovarian_breast_colorectal/icgc_donor_id.csv')).to_numpy())

        gene_ids = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/ovarian_breast_colorectal/gene_id.csv')).dropna()

        gene_ids = np.transpose(gene_ids.to_numpy())

        tissue_location_data = np.transpose(pd.read_csv(os.path.join(dirpath,
                                                                     'data/icgc/methylation/ovarian_breast_colorectal/tissue_location_data.csv')).to_numpy())

    return expression_matrix, sample_ids, gene_ids, tissue_location_data


def load_data_rna(rna_type):

    dirpath = os.path.dirname(__file__)

    if rna_type == 'seq':

        expression_matrix = pd.read_csv(os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_seq/exp_seq.csv'), dtype=np.float64).to_numpy()

        sample_ids = np.transpose(pd.read_csv(os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_seq/icgc_donor_id.csv')).to_numpy())

        gene_ids = pd.read_csv(os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_seq/gene_id.csv')).dropna()

        gene_ids = np.transpose(gene_ids.to_numpy())

        tissue_location_data = np.transpose(pd.read_csv(os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_seq/tissue_location_data.csv')).to_numpy())

    elif rna_type == 'array':

        expression_matrix = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_array/rna_plot.csv'), dtype=np.float64).fillna(0).to_numpy()

        sample_ids = np.transpose(
            pd.read_csv(os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_array/icgc_donor_id.csv')).to_numpy())

        gene_ids = pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_array/gene_id.csv')).dropna()

        gene_ids = np.transpose(gene_ids.to_numpy())

        tissue_location_data = np.transpose(pd.read_csv(
            os.path.join(dirpath, 'data/icgc/methylation/meth_rna_joint/rna/rna_array/tissue_location_data.csv')).to_numpy())

    return expression_matrix, sample_ids, gene_ids, tissue_location_data


def load_data_from_numpy():

    dirpath = os.path.dirname(__file__)

    expression_matrix = np.load(os.path.join(dirpath, 'data/icgc/savednumpy/ExpressionMatrix.npy'), allow_pickle=True)

    sample_ids = np.load(os.path.join(dirpath, 'data/icgc/savednumpy/SampleIds.npy'), allow_pickle=True)

    gene_ids = np.load(os.path.join(dirpath, 'data/icgc/savednumpy/GeneLabs.npy'), allow_pickle=True)

    tissue_location_data = np.load(os.path.join(dirpath, 'data/icgc/savednumpy/SampleLocations.npy'), allow_pickle=True)

    sample_to_survival = np.load(os.path.join(dirpath, 'data/icgc/savednumpy/SampleToSurvival.npy'), allow_pickle=True)

    return expression_matrix, sample_ids, gene_ids, tissue_location_data, sample_to_survival


def get_tissue_to_patient_index(num_tissues, tissues_locations, tissues_location_data):

    tumour_to_patients_index = []

    for t in range(num_tissues):

        tumour_to_patients_index.append(np.where(tissues_location_data == tissues_locations[t])[1])

    return tumour_to_patients_index


def get_random_subset_of_patients(num_tissues, num_patients, tissue_to_patient_index, random_seed):

    np.random.seed(random_seed)

    sample_patients_tumour_location_idx = np.array([], dtype=int)

    for t in range(num_tissues):

        curr_num_patients = num_patients

        if len(tissue_to_patient_index[t]) < num_patients:

            curr_num_patients = len(tissue_to_patient_index[t])

        sample_patients_tumour_location_idx = np.append(sample_patients_tumour_location_idx,
                                                        np.random.choice(tissue_to_patient_index[t], curr_num_patients, replace=False))

    return sample_patients_tumour_location_idx


def get_subset_of_genes(gene_selection_type, subset_data, num_genes, total_num_genes, specified_genes, gene_ids):

    if gene_selection_type == 'variance':

        sampled_genes = np.argsort(-subset_data.std(axis=0))[0:num_genes]

    elif gene_selection_type == 'mean_absolute_deviation':

        mad = mean_absolute_deviation(subset_data, axis=0)

        sampled_genes = np.argsort(-mad)[0:num_genes]

    elif gene_selection_type == 'random':

        sampled_genes = get_genes_that_satisfy_cond(subset_data, num_genes, total_num_genes)

    elif gene_selection_type == 'nano_string':

        sampled_genes = get_nano_string_genes(gene_ids)

    elif gene_selection_type == 'nano_string_n_more':

        mad = mean_absolute_deviation(subset_data, axis=0)

        mad_genes = np.argsort(-mad)[0:num_genes]

        nano_string_genes = get_nano_string_genes(gene_ids)

        sampled_genes = np.unique(np.append(nano_string_genes, mad_genes))

    elif gene_selection_type == 'pca':

        pca = PCA(n_components=num_genes)

        sampled_genes = pca.fit_transform(subset_data)

    else:

        raise Exception(' not recognized')

    if specified_genes is not None:

        specified_gene_indices = get_specified_genes(specified_genes, gene_ids)

        sampled_genes = np.unique(np.append(specified_gene_indices, sampled_genes))

    return sampled_genes


def get_specified_genes(specified_genes, gene_ids):

    specified_genes_indices = np.array([], dtype=np.int64)

    for gene in specified_genes:

        index = np.where(gene_ids == gene)[1][0]

        specified_genes_indices = np.append(specified_genes_indices, int(index))

    return specified_genes_indices


def get_tissue_matrix(num_patients, num_tissues, subset_tissue=None):

    if subset_tissue is not None:

        factors = pd.factorize(subset_tissue)[0] + 1

        tissue_matrix = get_cluster_matrix(factors)

        return tissue_matrix

    N = num_patients * num_tissues

    tissue_matrix = np.zeros(shape=(N, num_tissues), dtype=np.int64)

    for t in range(num_tissues):

        idx = np.arange(start=num_patients*t, stop=num_patients*(t+1))

        tissue_matrix[idx, t] += 1

    return tissue_matrix


def get_survival_subset(sample_to_survival, sample_ids):

    survival_subset = np.ndarray(shape=(len(sample_ids), 2), dtype=object)

    for i, id in enumerate(sample_ids):

        index = np.where(sample_to_survival[:, 0] == id)[0][0]

        survival_subset[i] = sample_to_survival[index, 2:4]

    return survival_subset

# TODO: make shift = 8.4738
def log_z_score_data(data, scale=8.4738, shift=1, log_normalize_type='normalize_across_patient'):

    if np.any(data == 0):

        data = data + shift

        print('Note: A 0 value was found in data matrix - shifted data by {}'.format(shift))

    N = data.shape[0]

    if log_normalize_type == 'log_normalize_across_patient':

        transform_data = np.log(data)

        mean_transform_data = np.mean(transform_data, axis=1).reshape(N, 1)

        sd_transform_data = np.sqrt(np.var(transform_data, axis=1)).reshape(N, 1)

        log_z_score = (transform_data - mean_transform_data) / sd_transform_data

    elif log_normalize_type == 'log_normalize':

        transform_data = np.log(data)

        mean_transform_data = np.mean(transform_data)

        sd_transform_data = np.sqrt(np.var(transform_data))

        log_z_score = (transform_data - mean_transform_data) / sd_transform_data

    else:

        raise Exception("Did not recognize the log normalization type")

    if scale != 1:

        log_z_score = scale * log_z_score

        print('Note: user scaled data matrix by {}'.format(scale))

    return log_z_score


def get_genes_that_satisfy_cond(subset_data, M_subset, total_number_genes):

    sampled_genes = -1 * np.ones(shape=(M_subset,), dtype=np.int64)

    for j in range(M_subset):

        gene_j = np.random.choice(range(0, total_number_genes))

        while (mean_absolute_deviation(subset_data[:, gene_j]) <= 1e-3) or (gene_j in sampled_genes):

            gene_j = np.random.choice(range(0, total_number_genes))

        sampled_genes[j] = gene_j

    return sampled_genes


def get_nano_string_genes(gene_ids):

    dirpath = os.path.dirname(__file__)

    nano_string_genes = np.transpose(pd.read_csv(os.path.join(dirpath, 'data/intersection_nano_string_genes.csv'), header=None).to_numpy(np.str))

    test0 = np.array([gene.strip().lower() for gene in nano_string_genes[0]])

    test1 = np.array([gene.strip().lower() for gene in gene_ids[0]])

    intersection_test = np.intersect1d(ar1=test0, ar2=test1)

    intersection = np.intersect1d(ar1=gene_ids[0], ar2=nano_string_genes)

    gene_indices = []

    for g in intersection:

        gene_indices.append(np.where(gene_ids[0] == g)[0][0])

    return gene_indices

# ============================================== math utils ============================================================= #

@numba.jit(nopython=True)
def log_factorial(x):

    return log_gamma(x + 1)


@numba.vectorize([numba.float64(numba.float64)])
def log_gamma(x):

    return np.math.lgamma(x)


def mean_absolute_deviation(data, axis=None):

    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

# ============================================== algorithms 3 neal utils ===============================================


def get_indices_c(c, i, Z):

    N = Z.shape[0]

    indices = []

    num_points = 0

    for k in range(N):

        if Z[k, c] == 1 and k != i:
            indices.append(k)

            num_points += 1

    return indices, num_points


# ================================================== timer ============================================================== #

class Timer:
    """ Taken from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html
    """

    def __init__(self, func=time.time):
        self.elapsed = 0.0

        self._func = func

        self._start = None

    @property
    def running(self):
        return self._start is not None

    def reset(self):
        self.elapsed = 0.0

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')

        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')

        end = self._func()

        self.elapsed += end - self._start

        self._start = None

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()



