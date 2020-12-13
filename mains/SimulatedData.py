import sys
import os
import numpy as np
from models.dpgmm_numba import DPGMM
from postprocessors.postprocessing import PostProcessor
from data.simulated.normal_model import simulated_data_normal
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)


if __name__ == "__main__":

    colours = 'rb'

    np.random.seed(123)

    data = simulated_data_normal(N=100, M=100, simulation_type="ratio", ratio=1, normalize_type=False, plot=True)

    model = DPGMM(data=data['data'],
                  tissue_assignments=data['T'],
                  v_init_type='random',
                  chi_init_type='zeros',
                  Z_init_type='random',
                  alphaDP_init_type='random',
                  sample_v_bool=True,
                  sample_chi_bool=False,
                  sample_Z_bool=True,
                  sample_alphaDP_bool=True,
                  sample_split_merge_bool=True,
                  adapt_iter=0,
                  plot_iter=0,
                  plot=False)

    model.initialize_model()

    model.mcmc(num_burnin=100, num_inference=1000)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=data['data'],
                                   T=data['T'],
                                   colours=colours,
                                   chain=model.mcmc_chain,
                                   absolute_path=os.path.join(dirpath, 'runs/simulateddata/'))

    post_processor.post_process_plots(data=True,
                                      log_posterior=True,
                                      log_integral_factor=True,
                                      cluster_matrix=True,
                                      alphaDP=True,
                                      v_average=True,
                                      chi_average=True,
                                      shifted_data=True,
                                      acceptance_rate=True)
