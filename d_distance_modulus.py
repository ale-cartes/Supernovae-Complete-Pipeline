"""
Module for estimating and filtering the distance modulus from supernova data.

This script handles:
- Estimating the distance modulus using standardization parameters.
- Filtering data based on uncertainties and statistical constraints.

Author: Alejandro Cartes
"""

from c_LC_fitting import *

def estimate_distance_modulus(data_fitted,
                              alpha=0.14, beta=3.2, gamma=0, M0=0,
                              sigma_int=0):
    """
    Function that estimates distance modulus using the Tripp formula
    \mu = m + \alpha x_1 - \beta c + \gamma G - M_0
    """
    # Calculate standardized absolute magnitude
    mabs_standard = (M0 - alpha * data_fitted.x1 + beta * data_fitted.c -
                     gamma * data_fitted.G_host)
    
    # Calculate error in standardized absolute magnitude
    jacobian_M = np.array([-alpha, beta])
    cov_matrices = np.stack(data_fitted.cov_matrix.values)
    cov_matrices_M = cov_matrices[:, 1:3, 1:3]
    
    var_M = np.einsum('i, nij, j -> n', jacobian_M, cov_matrices_M, jacobian_M)
    mabs_standard_err = np.sqrt(var_M)
    
    # Calculate distance modulus
    mu = data_fitted.m_B - mabs_standard
    
    # Calculate error in distance modulus
    # Error contribution from SALT fit
    jacobian = np.array([1, alpha, -beta])
    sigma_fit_2 = np.einsum('i, nij, j -> n',
                            jacobian, cov_matrices, jacobian) # J @ cov @ J.T

    # Error contribution from redshift and peculiar velocities
    vpec_err = data_fitted.vpec_err if "vpec_err" in data_fitted.columns else 0
    z, z_err = data_fitted.z, data_fitted.z_err

    sigma_z = ((5 / np.log(10)) * (1 + z) / (z * (1 + z / 2)) *
                 np.sqrt(z_err ** 2 + (vpec_err ** 2 / c) ** 2))
    
    # Error contribution from gravitational lensing
    sigma_lens = 0.055 * z
    
    # total error in distance modulus
    mu_err = np.sqrt(sigma_fit_2 + sigma_z ** 2 + sigma_lens ** 2 +
                     sigma_int ** 2)

    # Assign results to DataFrame
    data_fitted['mu'] = mu
    data_fitted['mu_err'] = mu_err
    
    data_fitted['mabs_stand'] = mabs_standard
    data_fitted['mabs_stand_err'] = mabs_standard_err
    
    # Calculate distance modulus using absolute magnitude estimated by sncosmo
    mu_mM = data_fitted.m_B - data_fitted.mabs_B
    mu_mM_err = np.sqrt([cov[0, 0] for cov in data_fitted.cov_matrix.values])

    data_fitted['mu_mM'] = mu_mM
    data_fitted['mu_mM_err'] = mu_mM_err

    return data_fitted

def filter_data(data_fitted, x1_range=(-3, 3), c_range=(-0.3, 0.3),
                x1_err_max=1, c_err_max=0.2,
                filter_mu=False, mu_err_max=0.1):
    
    x1_err = np.sqrt([cov[1, 1] for cov in data_fitted.cov_matrix])
    c_err = np.sqrt([cov[2, 2] for cov in data_fitted.cov_matrix])
    
    mask = (data_fitted.x1.between(*x1_range) &
            data_fitted.c.between(*c_range) &
            (x1_err < x1_err_max) &
            (c_err < c_err_max))
    
    data_filtered = data_fitted[mask].copy()
    
    if filter_mu:
        # Remove unphysical values and high uncertainties
        data_filtered = data_filtered[(data_filtered['mu'] > 0) &
                                      (data_filtered['mu_err'] < mu_err_max)]

    return data_filtered

def plot_histograms(data_fitted, columns, title_prefix=""):
    """
    Plot histograms for the given columns of the dataframe before and after filtering.
    """
    for col in columns:
        plt.figure()
        data_fitted[col].hist()

        plt.title(f"{title_prefix}{col} - histogram")
        plt.yscale('log')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        plt.show()
    
def plot_distance_modulus_vs_z(data, marg=True):
    """
    Plot the distance modulus vs redshift with error bars and a 2D histogram.

    Parameters:
        data: DataFrame containing the data related to distance modulus
    """
    # Create subplots: one for the error bars and another for the 2D histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                           sharey=True, tight_layout=True)

    mu_str = "mu" if marg else "mu_mM"
    mu_err_str = mu_str + '_err'

    # Plot error bars in the first subplot
    ax[0].errorbar(data['z'], data[mu_str], yerr=data[mu_err_str], fmt='o',
                   color='black', mew=0.3, mec='white', capsize=4, alpha=0.1,
                   zorder=-1)
    
    y_label = "Marginalized Distance modulus" if marg else "Distance modulus"
    ax[0].set_ylabel(y_label)

    # Plot 2D histogram in the second subplot
    hist2d = ax[1].hist2d(data['z'], data[mu_str], bins=50,
                          cmap='viridis', vmax=250)
    cbar = plt.colorbar(hist2d[3], ax=ax[1])
    cbar.set_label('Counts')

    # Set labels and grids
    for ax_i in ax:
        ax_i.set_xlabel("z")
        ax_i.grid(ls=':', alpha=0.5)

        # theoretical distance modulus
        h = None if marg else 0.7

        z_plot = np.sort(data['z'].unique())
        ax_i.plot(z_plot,
                  distance_modulus(z_plot, omega_m=0.3, omega_de=0.7, h=h),
                  'r--', label=r"$\Omega_m = 0.3, \Omega_{\Lambda} = 0.7$")
        
        ax_i.plot(z_plot,
                  distance_modulus(z_plot, omega_m=1.0, omega_de=0, h=h),
                  'y--', label=r"No dark energy")
    
    ax[0].set_ylabel(r"Distance modulus")
    ax[0].legend()

    # Set the title for the entire figure
    title = r"$\mu_{marg}$" if marg else r"$\mu$"
    fig.suptitle(fr"{title} vs z")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load fitted light curve data
    data_fitted = load_fitted_data(name_file="classified_Ia_0.9.pkl")
    
    # Estimate distance modulus
    data_fitted = estimate_distance_modulus(data_fitted)

    # Apply data filtering
    data_filtered = filter_data(data_fitted, filter_mu=True, mu_err_max=0.1)

    # Plot histograms after filtering
    plot_histograms(data_filtered, ['x0', 'x1', 'c'],
                    title_prefix="After Filtering: ")

    # Plot distance modulus vs redshift
    plot_distance_modulus_vs_z(data_filtered, marg=True)
    plot_distance_modulus_vs_z(data_filtered, marg=False)
