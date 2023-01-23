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

    specific_obs: None or int (optional)
    if it is None, days will be calculted for all data in Data Frame

    inplace: bool (optional)
    if inplace is True, the function adds a new column labeled as "Days"
    to the dataFrame

    output: bool (optional)
    if it's True, an array of days will be returned
    """

    if type(specific_obs) == int:
        lightcurve = lightcurve[lightcurve.obs == specific_obs]

    min_MJD = lightcurve.groupby('obs').MJD.min()

    def row_day_calc(row):
        i = row.obs
        day = row.MJD - min_MJD[i]
        return day

    if inplace and ("Days" not in lightcurve.columns):
        lightcurve['Days'] = lightcurve.apply(row_day_calc, axis=1)

    if output:
        return lightcurve.apply(row_day_calc, axis=1).to_numpy()


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

    try:
        name = data_frame.name
    except Exception:
        name = ''

    ax.set_title(f'{name}, obs: {obs}')
    if not isinstance(summary, type(None)):
        summ_obs = summary.query('CID == @obs')
        ax.set_title(rf"{name}, " +
                     rf"obs: {obs}, " +
                     rf"peak: {summ_obs['PEAKMJD'].values[0]}, " +
                     rf"SN type: {summ_obs['SNTYPE'].values[0]}")
    ax.legend()
    plt.show()
