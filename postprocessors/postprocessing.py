import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import os
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from utils import get_cluster_array, get_flat_clustering, compute_mpear, array_to_int, get_cluster_matrix
from lifelines import KaplanMeierFitter
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PostProcessor(object):

    def __init__(self, generated_dictionary, data, T, colours, chain, absolute_path, tissue_locations=None, survival_data=None):
        """
        :param generated_dictionary: if data is forward simulated, then this is a dictionary of all parameters from forward simulation
        :param data:data we are performing inference on
        :param T: tissue assignments
        :param colours: colours of each tissue used for plotting
        :param chain: mcmc chain from model inference (different from previous post processors)
        :param absolute_path: path to store chains and plots
        """
        # ======================================= assertion statements =================================================

        assert(len(colours) == T.shape[1])

        # ======================================= post processor =======================================================

        self.generated = generated_dictionary

        self.data = data

        self.T = T

        self.N = data.shape[0]

        self.M = data.shape[1]

        self.L = self.T.shape[1]

        self.tissue_assignments = get_cluster_array(self.T)

        self.tissue_locations = tissue_locations

        self.survival_data = survival_data

        # ========================================== graph settings =====================================

        self.colours = colours

        self.absolute_path = absolute_path + str(datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S"))

        # ========================================== create chains ===============================================

        self.log_posterior_chain = chain['log_posterior']

        self.log_integral_factor_chain = chain['log_integral']

        self.K_chain = chain['K']

        self.Z_chain = chain['Z']

        self.v_chain = chain['v']

        self.chi_chain = chain['chi']

        self.alphaDP_chain = chain['alphaDP']

        self.num_mcmc_iter = len(self.v_chain)

        self.cluster_labels_chain = None

        # ======================================== values used for posterior inference ===========================

        self.avg_v = None

        self.avg_chi = None

        self.mpear_clusters = None

        self.cluster_to_index = None

        self.cluster_counts = None

        # ===================================== acceptance rate =================================================

        self.acceptance_rate_v = chain['acceptance_rate_v']

        self.acceptance_rate_chi = chain['acceptance_rate_chi']

        self.acceptance_rate_alphaDP = chain['acceptance_rate_alphaDP']

        # ============================================ utils =====================================================

        self.make_directories()

        self.get_cluster_labels()

        self.get_averages()

    def make_directories(self):

        os.mkdir(self.absolute_path)

        os.mkdir(self.absolute_path + str('/plots'))

        os.mkdir(self.absolute_path + str('/chains'))

    def get_cluster_labels(self):
        """
        same function as the self.Z chain part in get_parameters_chain in other post processors
        """
        cluster_label_chain = np.zeros(shape=(self.N, self.num_mcmc_iter), dtype=np.int64)

        for i in range(self.num_mcmc_iter):

            for j in range(self.N):

                curr_Z = self.Z_chain[i]

                curr_cluster = curr_Z[j]

                curr_label = array_to_int(curr_cluster)

                cluster_label_chain[j, i] = curr_label

        self.cluster_labels_chain = cluster_label_chain

    def get_averages(self):

        self.avg_v = np.average(self.v_chain, axis=0).reshape(1, self.M)

        self.avg_chi = np.average(self.chi_chain, axis=0).reshape(self.L, self.M)

    def cluster_with_mpear(self, max_clusters=None, plot=False):

        dist_mat = pdist(self.cluster_labels_chain, metric='hamming')

        diff_mat = squareform(dist_mat)

        sim_mat = 1 - diff_mat

        Z = linkage(dist_mat, method='average')

        if max_clusters is None:

            max_clusters = len(self.cluster_labels_chain) + 1

        else:

            max_clusters = min(max_clusters, len(self.cluster_labels_chain))

        max_clusters = max(max_clusters, 1)

        best_cluster_labels = get_flat_clustering(Z, 1)

        max_pear = 0

        for i in range(2, max_clusters + 1):

            cluster_labels = get_flat_clustering(Z, number_of_clusters=i)

            pear = compute_mpear(cluster_labels, sim_mat)

            if pear > max_pear:

                max_pear = pear

                best_cluster_labels = cluster_labels

        if plot:

            cmap = sns.cubehelix_palette(light=1, as_cmap=True)

            # get colours for each tissue

            tissue_to_colour_dict = dict(zip(np.unique(self.tissue_assignments), self.colours))

            tissue_to_colour_mapping = np.vectorize(lambda x: tissue_to_colour_dict[x])

            row_colours = tissue_to_colour_mapping(self.tissue_assignments)

            sns.clustermap(sim_mat, cmap=cmap, row_colors=row_colours)

            plt.savefig(self.absolute_path + str('/plots') + str('/clustermap.png'))

            plt.show()

        self.get_cluster_fields(best_cluster_labels)

    def get_cluster_fields(self, cluster_labels):

        # get values for fields

        cluster_to_index = []

        cluster_counts = []

        K = 0

        for i in range(self.N):

            if cluster_labels[i] >= K:

                cluster_to_index.append([])

                cluster_counts.append(0)

                K += 1

            cluster_to_index[cluster_labels[i]].append(i)

            cluster_counts[cluster_labels[i]] += 1

        # update fields

        self.mpear_clusters = cluster_labels

        self.cluster_to_index = cluster_to_index

        self.cluster_counts = cluster_counts

    def post_process_plots(self,
                           data=False,
                           log_posterior=False,
                           log_integral_factor=False,
                           cluster_matrix=False,
                           v_average=False,
                           chi_average=False,
                           alphaDP=False,
                           shifted_data=False,
                           acceptance_rate=False):

        # ==================================== Plot 1: Data ==================================================
        if data:

            plt.matshow(self.data)

            plt.title('data')

            plt.colorbar()

            plt.savefig(self.absolute_path + str('/plots') + str('/data.png'))

            plt.show()

        # ==================================== Plot 2: Log joint ==============================================

        if log_posterior:

            plt.plot(self.log_posterior_chain)

            plt.xlabel('iteration')

            plt.title('log posterior')

            plt.savefig(self.absolute_path + str('/plots') + str('/log_posterior.png'))

            plt.show()

        if log_integral_factor:

            plt.plot(self.log_integral_factor_chain)

            plt.xlabel('iteration')

            plt.title('log integral')

            plt.savefig(self.absolute_path + str('/plots') + str('/log_integral_factor.png'))

            plt.show()

        # ==================================== Plot 3: Cluster Matrix based on MCMC trace  ========================

        if cluster_matrix:

            self.cluster_with_mpear(plot=True)

        # ==================================== Plot 4: average v from mcmc trace ========================

        if v_average:

            plt.matshow(self.avg_v)

            plt.title('average v')

            plt.colorbar()

            plt.savefig(self.absolute_path + str('/plots') + str('/average_v.png'))

            plt.show()

        # ==================================== Plot 5: average chi from mcmc trace  ========================

        if chi_average:

            plt.matshow(self.avg_chi)

            plt.title('average_chi')

            plt.colorbar()

            plt.savefig(self.absolute_path + str('/plots') + str('/average_chi.png'))

            plt.show()

            print(self.avg_chi)

        # ==================================== Plot 6: average chi from mcmc trace  ===========================

        if alphaDP:

            plt.hist(self.alphaDP_chain)

            plt.title("alphaDP")

            plt.savefig(self.absolute_path + str('/plots') + str('/alphaDP.png'))

            plt.show()

        # ==================================== Plot 6: average chi from mcmc trace  ===========================

        if shifted_data:

            v_average = np.average(self.v_chain, axis=0).reshape(1, self.M)

            chi_average = np.average(self.chi_chain, axis=0).reshape(self.L, self.M)

            shifted_data = self.data - v_average - self.T @ chi_average

            plt.matshow(shifted_data)

            plt.colorbar()

            plt.title('Shifted Data')

            plt.savefig(self.absolute_path + str('/plots') + str('/shifted_data.png'))

            plt.show()

        # ==================================== Plot 7: acceptance rate v, chi  ===========================

        if acceptance_rate:

            plt.matshow(self.acceptance_rate_v.reshape(1, self.M))

            plt.colorbar()

            plt.title('v acceptance rate')

            plt.savefig(self.absolute_path + str('/plots') + str('/v_acceptance_rate.png'))

            plt.show()

            plt.matshow(self.acceptance_rate_chi.reshape(self.L, self.M))

            plt.colorbar()

            plt.title('chi acceptance rate')

            plt.savefig(self.absolute_path + str('/plots') + str('/chi_acceptance_rate.png'))

            plt.show()

    def survival_analysis(self):

        survival_time = self.survival_data.reshape(self.N, 2)

        Z_mpear = get_cluster_matrix(self.mpear_clusters)

        df_survival = self.get_survival_data_frame(survival_time, Z_mpear)

        df_survival.to_csv(self.absolute_path + str("/survival_data.csv"), index=False)

        path_to_script = dirpath + "/survival/SurvivalAnalysis.R"

        os.system("Rscript {} {}".format(path_to_script, self.absolute_path))

    def km_curves(self):

        survival_time = pd.DataFrame(data=self.survival_data, columns=['Status', 'Time'])

        survival_time['Cluster'] = self.mpear_clusters

        survival_time['Tissue'] = self.tissue_assignments

        survival_time.loc[survival_time.Status == 'Dead', 'Status'] = 1

        survival_time.loc[survival_time.Status == 'Censored', 'Status'] = 0

        T = survival_time['Time'].values

        E = survival_time['Status'].values

        kmf = KaplanMeierFitter()

        # for tissue in np.unique(survival_time['Tissue']):
        #
        #     idx = np.where(survival_time['Tissue'] == tissue)[0]
        #
        #     kmf.fit(T[idx], E[idx], label=str(self.tissue_locations[tissue]))
        #
        #     kmf.plot(ci_show=False)

        for cluster in np.unique(survival_time['Cluster']):

            idx = np.where(survival_time['Cluster'] == cluster)[0]

            kmf.fit(T[idx], E[idx], label=str("Cluster") + str(cluster))

            kmf.plot(ci_show=False)

            kmf.plot(ci_show=False).set_ylabel('probability of survival')

        plt.savefig(self.absolute_path + '/plots/km_curves.png')

        plt.show()

    def get_survival_data_frame(self, survival_time, Z):

        # step 1) create matrix

        # Note: first column - survival time, next self.L columns - tissue assignments, rest of columns - cluster assignments

        # data_matrix = np.hstack((np.hstack((survival_time, self.T)), Z))

        data_matrix = np.hstack((survival_time, self.T))

        # step 2) clean matrix

        if np.isnan(np.sum(data_matrix[:, 1])):

            # delete NA values

            index = np.where(data_matrix[:, 1] == np.nan)[0][0]

            data_matrix = np.delete(data_matrix, index, axis=0)

        elif np.any((data_matrix[:, 1]) < 0):

            # delete negative values

            index = np.where(data_matrix[:, 1] < 0)[0][0]

            data_matrix = np.delete(data_matrix, index, axis=0)

        # step 3) create dataframe

        column_names = ['Status', 'Time'] + ["Tissue{}".format(i) for i in range(0, self.L)] + ["Cluster{}".format(i) for i in range(0, Z.shape[1])]

        column_names[2:2 + self.L] = self.tissue_locations

        return pd.DataFrame(data_matrix, columns=column_names)







