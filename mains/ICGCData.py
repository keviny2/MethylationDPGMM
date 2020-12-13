import sys
import os
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)
print(sys.path)
import numpy as np
from numpy.random import seed
from models.dpgmm import DPGMM
from postprocessors.postprocessing import PostProcessor
from utils import get_subset_of_data

if __name__ == "__main__":

    tissue_locations = np.array(['Brain', 'Breast', 'Ovary', 'Lung', 'Colorectal'])

    colours = 'rbcgk'

    data, T, subset_tissue, subset_sample_ids, subset_gene_ids, subset_survival = get_subset_of_data(tissue_locations=tissue_locations,
                                                                                                     specified_genes=None,
                                                                                                     num_tissues=tissue_locations.shape[0],
                                                                                                     num_patients=30,
                                                                                                     num_genes=200,
                                                                                                     gene_selection_type='nano_string',
                                                                                                     scale=1,
                                                                                                     random_seed=123,
                                                                                                     log_normalize=True,
                                                                                                     plot=True)

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
                  adapt_iter=0,
                  plot_iter=0,
                  plot=True)

    model.initialize_model()

    model.mcmc(num_burnin=1000, num_inference=24000)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=data,
                                   T=T,
                                   colours=colours,
                                   chain=model.mcmc_chain,
                                   absolute_path=os.path.join(dirpath, 'runs/realdata/'),
                                   survival_data=subset_survival)

    post_processor.post_process_plots(data=True,
                                      log_posterior=True,
                                      log_integral_factor=True,
                                      cluster_matrix=True,
                                      alphaDP=True,
                                      v_average=True,
                                      chi_average=True,
                                      shifted_data=True,
                                      acceptance_rate=True)

    post_processor.survival_analysis()
