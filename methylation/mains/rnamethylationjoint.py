from models.dpgmm_numba import DPGMM
from postprocessors.postprocessing import PostProcessor
from utils import get_subset_of_data_methylation, get_subset_of_data, reorder

import os
import numpy as np

dirpath = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    tissue_locations = np.array(['Brain', 'Lung', 'Blood'])

    colours = 'rbc'

    info = 'joint, num_genes=100, pca, with tissue, [Brain, Lung, Blood])'

    data_rna, T_rna, subset_tissue_rna, subset_sample_ids_rna, subset_gene_ids_rna = get_subset_of_data(tissue_locations=tissue_locations,
                                                                                                        specified_genes=None,
                                                                                                        num_tissues=tissue_locations.shape[0],
                                                                                                        num_patients=50,
                                                                                                        num_genes=500,
                                                                                                        gene_selection_type='pca',
                                                                                                        scale=1,
                                                                                                        random_seed=123,
                                                                                                        log_normalize_type='log_normalize_across_patient',
                                                                                                        plot=False)



    data_meth, T_meth, subset_tissue_meth, subset_sample_ids_meth, subset_gene_ids_meth = get_subset_of_data_methylation(tissue_locations=tissue_locations,
                                                                                                                         specified_genes=None,
                                                                                                                         num_tissues=tissue_locations.shape[0],
                                                                                                                         num_patients=50,
                                                                                                                         num_genes=110,
                                                                                                                         gene_selection_type='pca',
                                                                                                                         scale=1,
                                                                                                                         random_seed=123,
                                                                                                                         log_normalize_type=None,
                                                                                                                         plot=False,
                                                                                                                         joint=True)




    # subset meth data

    meth_indices = np.where(np.isin(subset_sample_ids_meth, subset_sample_ids_rna))[0]

    subset_sample_ids_meth = subset_sample_ids_meth[meth_indices]

    data_meth = data_meth[meth_indices, ]

    T_meth = T_meth[meth_indices, ]

    subset_tissue_meth = subset_tissue_meth[meth_indices, ]



    # subset rna data

    rna_indices = np.where(np.isin(subset_sample_ids_rna, subset_sample_ids_meth))[0]

    subset_sample_ids_rna = subset_sample_ids_rna[rna_indices]

    data_rna = data_rna[rna_indices, ]

    T_rna = T_rna[rna_indices, ]

    subset_tissue_rna = subset_tissue_rna[rna_indices, ]



    # make meth in same order as rna

    ordered_index = []

    rna_id_temp = subset_sample_ids_rna.tolist()

    meth_id_temp = subset_sample_ids_meth.tolist()

    # I think this explanation is wrong lol, code works beautifully
    # after for-loop, ordered_index will contain the indexes that methylation should change to in order to match rna
    # ex. ordered_index = [5,2,6,1,3,4], then in order for methylation to be in the same order as rna, we need to create
    # a new list that looks like: [meth[5],meth[2],meth[6],meth[1],meth[3],meth[4]]

    for rna_id in rna_id_temp:

        idx = meth_id_temp.index(rna_id)

        ordered_index.append(idx)


    reorder(subset_sample_ids_rna, ordered_index)

    reorder(data_rna, ordered_index)

    reorder(T_rna, ordered_index)

    reorder(subset_tissue_rna, ordered_index)





    # create joint data

    data = np.concatenate((data_rna, data_meth), 1)

    T = T_rna


    model = DPGMM(data=data,
                  tissue_assignments=T,
                  v_init_type='random',
                  chi_init_type='random',
                  Z_init_type='random',
                  alphaDP_init_type='one',
                  sample_v_bool=True,
                  sample_chi_bool=True,
                  sample_Z_bool=True,
                  sample_alphaDP_bool=True,
                  sample_split_merge_bool=True,
                  adapt_iter=10,
                  plot_iter=0,
                  plot=False)

    model.v_prior = {'mean': np.zeros(shape=(model.M,)), 'variance': (np.ptp(model.data) / 4) * np.ones(shape=(model.M,))}

    model.chi_prior = {'mean': np.zeros(shape=(model.L, model.M)),
                      'variance': (np.ptp(model.data) / 4) * np.ones(shape=(model.L, model.M))}

    model.base_distribution = {'mean': np.zeros(shape=(model.M,)), 'precision': 2 * np.ones(shape=(model.M,)),
                              'alpha': 0.1 * np.ones(shape=(model.M,)),
                              'beta': (np.ptp(model.data) / 100) * np.ones(shape=(model.M,))}

    model.alphaDP_prior = {'alpha': 0.1, 'beta': 80.}

    model.initialize_model()

    model.mcmc(num_burnin=5, num_inference=5)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=data,
                                   T=T,
                                   colours=colours,
                                   chain=model.mcmc_chain,
                                   tissue_locations=tissue_locations,
                                   absolute_path=os.path.join('C:/Users/kevin/OneDrive/Documents/BC_Cancer_Research/DPGMM', 'runs/methylationdata/'))

    file = open(post_processor.absolute_path + '\\info.txt', 'w')
    file.write(info)
    file.close()

    post_processor.post_process_plots(data=True,
                                      log_posterior=True,
                                      log_integral_factor=True,
                                      cluster_matrix=True,
                                      alphaDP=True,
                                      v_average=True,
                                      chi_average=True,
                                      shifted_data=True,
                                      acceptance_rate=True)

    # post_processor.survival_analysis()