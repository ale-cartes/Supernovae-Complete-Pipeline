"""
Module for MCMC sampling of a ΛCDM model using supernova distance modulus data.

This script handles:
- Loading fitted light curve parameters.
- Estimating the distance modulus and filtering data.
- Defining the likelihood and prior functions.
- Performing MCMC sampling with emcee.
- Plotting and summarizing the posterior distributions.

Author: Alejandro Cartes
"""

from utils import *
from e_MCMC_flat_LCDM_nuisance import *

def load_and_preprocess_data(file_fitted_data="lc_params.pkl",
                             file_nuisance='nuisance/flat_lcdm_nuisance.h5',
                             sigma_int=0,
                             filter_mu=False, mu_err_max=0.1):
    """
    Load fitted light curve data, estimate distance modulus, and filter
    """

    nuisance_params_dict = load_nuisance_params(filename=file_nuisance)
    alpha, beta, gamma = [nuisance_params_dict[key]
                          for key in ['alpha', 'beta', 'gamma']]
    
    # alpha, beta, gamma = 0.161, 3.12, 0.038

    data = load_fitted_data(name_file=file_fitted_data)
    data = estimate_distance_modulus(data,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     sigma_int=sigma_int)

    mu_err_max += sigma_int
    return filter_data(data, filter_mu=filter_mu, mu_err_max=mu_err_max)

# MCMC objective function
def chi2_marg(params, data, reduced=False):
    """Compute the marginalized chi² for a ΛCDM model"""

    omega_m, omega_de = params
    mu_marg_th = distance_modulus(data.z, omega_m=omega_m, omega_de=omega_de)
    delta_mu_marg = data.mu - mu_marg_th
    
    sigma2_inv = 1 / data.mu_err ** 2
    A = (delta_mu_marg ** 2 * sigma2_inv).sum()
    B = (delta_mu_marg * sigma2_inv).sum()
    C = (sigma2_inv).sum()
    
    chi_squared = A + np.log(C / (2 * np.pi)) - B ** 2 / C
    dof = np.shape(data)[0] - len(params)

    return chi_squared / dof if reduced else chi_squared


prior_lims = [(0.0, 1.0),  # omega_m
              (0.0, 1.0)]  # omega_de
def log_prior(params):
    """Define the prior"""

    conds = [prior_lims[i][0] < param < prior_lims[i][1]
             for i, param in enumerate(params)]
    
    if all(conds):
        return 0.0
    
    return -np.inf

def log_likelihood(params, data):
    """Compute the log-likelihood function"""
    chi_squared = chi2_marg(params, data)
    log_norm = 0 # np.sum(np.log(2 * np.pi * data.mu_err ** 2))

    return -0.5 * (chi_squared + log_norm)

def log_posterior(params, data):
    """
    Compute the log-posterior (log-prior + log-likelihood)
    """
    lp = log_prior(params)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(params, data)

def best_params_max_posterior(data, x0):
    """
    Find the best-fit by maximizing the posterior
    """
    best_params = optimize.minimize(lambda params: -log_posterior(params, data),
                                    x0=x0, method="Powell")
    print(f"Best-fit: {best_params}")
    return best_params.x

def run_mcmc(data, best_params, nwalkers=50, nsteps=5000, load=False,
             filename="lcdm.h5"):
    """Perform MCMC sampling with emcee."""
    # Set up backend
    filename = f"data_folder/emcee/{filename}"
    backend = emcee.backends.HDFBackend(filename)

    if load:
        sampler = backend

    else:
        # omega_m, omega_de = best_params
        # param_initial = {'omega_m': omega_m + np.random.uniform(-1e-1, 1e-1, nwalkers),
        #                  'omega_de': omega_de + np.random.uniform(-1e-1, 1e-1, nwalkers)}
        # ndim = len(param_initial)
        # p0 = np.stack(list(param_initial.values()), axis=1)

        ndim = len(prior_lims)
        p0 = np.array([np.random.uniform(low, high, nwalkers)
                       for low, high in prior_lims]).T

        backend.reset(nwalkers, ndim)

        # Run MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(data,),
                                            backend=backend, pool=pool)
            sampler.run_mcmc(p0, nsteps, progress=True)

    return sampler

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data = load_and_preprocess_data(file_fitted_data='classified_Ia_0.9.pkl',
                                    file_nuisance='test_nuisance_low_z_3.h5',
                                    filter_mu=True, mu_err_max=1)
    
    data = data[data.z >= 0.1].sample(5_000, random_state=42)

    # Step 2: Minimize chi²
    x0 = [0.3, 0.7]
    best_params = best_params_max_posterior(data, x0=x0)

    # Step 3: Run MCMC sampling
    nwalkers, nsteps = 1_000, 1_000
    sampler = run_mcmc(data, best_params,
                       nwalkers=nwalkers, nsteps=nsteps,
                       load=True,
                       filename="test_lcdm_2.h5")

    # Step 4: Plot results
    labels = [r'$\Omega_m$', r'$\Omega_{de}$']
    plot_chains(sampler, labels=labels)

    params_dict = plot_corner(sampler, labels, truths=x0, pretty=True)
