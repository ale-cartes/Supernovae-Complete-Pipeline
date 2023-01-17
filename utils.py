import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits


def reader(file_name, fits_header=False):
    """
    Function that reads fits files and return a light-curves
    data frame

    Input
    ------
    file_name: str
    fits file name
    """

    if fits_header:
        with fits.open(file_name) as fits_file:
            global header
            header = fits_file[0].header
            print(repr(header))

    light_curves = Table.read(file_name, format='fits')
    light_curves = light_curves.to_pandas()
    index_obs_separator = light_curves[light_curves['MJD'] == -777].index

    n_obs = 1
    obs = []

    for mjd in light_curves['MJD']:
        if mjd == -777:
            n_obs += 1
        obs.append(n_obs)

    light_curves.insert(0, 'obs', obs)
    light_curves.drop(index_obs_separator, inplace=True)
    light_curves.reset_index(inplace=True)
    light_curves.drop(columns='index', inplace=True)

    light_curves.name = file_name.split('/')[-1]

    return light_curves


def summary(dump_file):
    data = pd.read_csv(dump_file, delimiter=' ', header=5)
    data.drop(data.columns[[0, 1]], axis=1, inplace=True)

    for i, type in enumerate(data['SNTYPE']):
        if type == 20:
            data.at[i, 'SNTYPE'] = "II+IIP"

        elif type == 21:
            data.at[i, 'SNTYPE'] = "IIn+IIN"

        elif type == 22:
            data.at[i, 'SNTYPE'] = "IIL"

        elif type == 32:
            data.at[i, 'SNTYPE'] = "Ib"

        elif type == 33:
            data.at[i, 'SNTYPE'] = "Ic+Ibc"

        elif type == 1:
            data.at[i, 'SNTYPE'] = "Ia"

    return data


def mjd_to_days(lightcurve, specific_obs=None, inplace=False, output=False):
    """
    Function that transforms MJD dates to days considering
    initial observation as day 0

    Input
    =====
    lightcurve: lightcurve data frame

    inplace: bool (optional)
    if inplace is True, the function adds a new column to the
    dataFrame labeled as "Days"

    specific_obs: None or int (optional)
    if it is None, days will be calculted for all data in Data Frame

    output: bool (optional)
    if it's True, an array of days will be returned
    """
    n = (lightcurve.obs).max()
    iterator = range(1, n+1)

    if type(specific_obs) == int:
        iterator = [specific_obs]
        lightcurve = lightcurve[lightcurve.obs == specific_obs]

    days = []
    for obs in iterator:
        mjd_obs = (lightcurve[lightcurve.obs == obs].MJD).values
        days.append(mjd_obs - mjd_obs[0])

    if inplace and ("Days" not in lightcurve.columns):
        lightcurve['Days'] = np.concatenate(days)

    if output:
        return np.concatenate(days)


def plotter(data_frame, obs, summary=None, days=False):
    """
    Function that plots supernova lightcurve

    Input
    =====
    data_frame: df with all supernovae data
    obs: observation number

    summary (optional): None or path to file with
    information related to nonIa supernovae (.DUMP)

    days (optional): bool
    if it's True, MJD will be expressed as days
    """
    data_obs = data_frame[data_frame['obs'] == obs]
    xlabel = 'MJD'

    if days:
        mjd_to_days(data_obs, inplace=True, specific_obs=obs, output=False)
        xlabel = 'Days'

    color = {'u': 'purple', 'g': 'green', 'r': 'red',
             'i': (150/255, 0, 0), 'z': (60/255, 0, 0)}

    fig, ax = plt.subplots(figsize=(14, 8))

    for band in (data_obs['BAND'].value_counts()).index:
        data_to_plot = data_obs[data_obs.BAND == band]

        band_str = band.decode("utf-8").replace(' ', '')

        xdata_plot = data_to_plot.MJD
        if days:
            xdata_plot = data_to_plot.Days

        ax.errorbar(xdata_plot, data_to_plot.FLUXCAL,
                    yerr=data_to_plot.FLUXCALERR, marker='o',
                    ls='--', capsize=2, color=color[band_str], label=band_str)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Flux (ADU)')
    ax.set_title(f'{data_frame.name}, obs: {obs}')
    if not isinstance(summary, type(None)):
        ax.set_title(rf"{data_frame.name}, " +
                     rf"obs: {obs}, " +
                     rf"peak: {summary['PEAKMJD'].values[0]}, " +
                     rf"SN type: {summary['SNTYPE'].values[0]}")
    ax.legend()
    plt.show()
