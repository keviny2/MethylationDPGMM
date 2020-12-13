import sys
import os
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)
print(sys.path)
import numpy as np
from numpy.random import seed
from models.dpgmm_numba import DPGMM
from postprocessors.postprocessing import PostProcessor
from utils import get_subset_of_data

if __name__ == "__main__":

    tissue_locations = np.array(['Ovary', 'Breast', 'Colorectal'])

    colours = 'rbc'

    data, T, subset_tissue, subset_sample_ids, subset_gene_ids = get_subset_of_data(tissue_locations=tissue_locations,
                                                                                                     specified_genes=None,
                                                                                                     num_tissues=tissue_locations.shape[0],
                                                                                                     num_patients=50,
                                                                                                     num_genes=100,
                                                                                                     gene_selection_type='pca',
                                                                                                     scale=1,
                                                                                                     random_seed=123,
                                                                                                     log_normalize_type='log_normalize_across_patient',
                                                                                                     plot=False)

    model = DPGMM(data=data,
                  tissue_assignments=T,
                  v_init_type='random',
                  chi_init_type='random',
                  Z_init_type='random',
                  alphaDP_init_type='random',
                  sample_v_bool=True,
                  sample_chi_bool=True,
                  sample_Z_bool=True,
                  sample_alphaDP_bool=True,
                  sample_split_merge_bool=True,
                  adapt_iter=10,
                  plot_iter=0,
                  plot=True)

    model.v_prior = {'mean': np.zeros(shape=(model.M,)), 'variance': (np.ptp(model.data) / 4) * np.ones(shape=(model.M,))}

    model.chi_prior = {'mean': np.zeros(shape=(model.L, model.M)),
                      'variance': (np.ptp(model.data) / 4) * np.ones(shape=(model.L, model.M))}

    model.base_distribution = {'mean': np.zeros(shape=(model.M,)), 'precision': 4 * np.ones(shape=(model.M,)),
                              'alpha': 1 * np.ones(shape=(model.M,)),
                              'beta': (np.ptp(model.data) / 100) * np.ones(shape=(model.M,))}

    model.alphaDP_prior = {'alpha': 1, 'beta': 80.}

    model.initialize_model()

    model.mcmc(num_burnin=200, num_inference=1000)

    # post_processor = PostProcessor(generated_dictionary=None,
    #                                data=data,
    #                                T=T,
    #                                colours=colours,
    #                                chain=model.mcmc_chain,
    #                                absolute_path=os.path.join(dirpath, 'runs/realdata/'),
    #                                tissue_locations=tissue_locations,
    #                                survival_data=subset_survival)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=data,
                                   T=T,
                                   colours=colours,
                                   chain=model.mcmc_chain,
                                   absolute_path=os.path.join(dirpath, 'runs/realdata/'),
                                   tissue_locations=tissue_locations)

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
    #
    # post_processor.km_curves()
