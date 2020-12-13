import sys
import os
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)
import numpy as np
from models.dpgmm_numba import DPGMM
from postprocessors.postprocessing import PostProcessor
from data.pancan32.pancan32_utils import get_data

if __name__ == "__main__":

    np.random.seed(123)

    # tissue_locations = np.array(['ACC', 'BLCA', 'BRCA', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP',
    #                              'LGG', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'READ', 'SKCM',
    #                              'STAD', 'THCA', 'UCEC'])

    tissue_locations = np.array(['ACC', 'BLCA', 'BRCA', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP'])

    colours = 'bgrcmykw'

    data, T, subset_tissue, subset_sample_ids, subset_gene_ids = get_data(num_patients=30,
                                                                          tissue_locations=tissue_locations,
                                                                          normalize_type='normalize_patients',
                                                                          plot=True)

    model = DPGMM(data=data,
                  tissue_assignments=T,
                  v_init_type='random',
                  chi_init_type='random',
                  Z_init_type='singletons',
                  alphaDP_init_type='random',
                  sample_v_bool=True,
                  sample_chi_bool=True,
                  sample_Z_bool=True,
                  sample_alphaDP_bool=True,
                  sample_split_merge_bool=True,
                  adapt_iter=0,
                  plot_iter=0,
                  plot=True)

    model.initialize_model()

    model.mcmc(num_burnin=100, num_inference=1000)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=data,
                                   T=T,
                                   colours=colours,
                                   chain=model.mcmc_chain,
                                   absolute_path=os.path.join(dirpath, 'runs/realdata/'))

    post_processor.post_process_plots(data=True,
                                      log_posterior=True,
                                      log_integral_factor=True,
                                      cluster_matrix=True,
                                      alphaDP=True,
                                      v_average=True,
                                      chi_average=True,
                                      shifted_data=True,
                                      acceptance_rate=True)
