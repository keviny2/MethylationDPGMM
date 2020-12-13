import sys
import os
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)
print(sys.path)
import numpy as np
from models.dpgmm_numba import DPGMM
from postprocessors.postprocessing import PostProcessor
from data.gtex.gtex_simulation import get_simulated_data
from data.gtex.gtex_data import get_GTex_data

if __name__ == "__main__":

    colours = 'rb'

    np.random.seed(123)

    primary_tissue = np.array(["Brain - Cortex", "Ovary"])

    latent_tissue = np.array(["Lung", "Thyroid"])

    data, patients_to_tissue, tissue_to_patients = get_GTex_data(num_patients=100,
                                                                 num_genes=1000,
                                                                 tissues=np.append(primary_tissue, latent_tissue),
                                                                 use_nano_string=True,
                                                                 plot=False)

    simulated_data, simulated_log_data, T = get_simulated_data(N=50,
                                                               data=data,
                                                               primary_tissue=primary_tissue,
                                                               latent_tissue=latent_tissue,
                                                               tissue_to_patients=tissue_to_patients,
                                                               prop=0.5,
                                                               log_normalize=True,
                                                               plot=False)

    model = DPGMM(data=simulated_log_data,
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
                  adapt_iter=50,
                  plot_iter=0,
                  plot=False)

    model.initialize_model()

    model.mcmc(num_burnin=100, num_inference=1000)

    post_processor = PostProcessor(generated_dictionary=None,
                                   data=simulated_log_data,
                                   T=T,
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
