import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import splrep, splev
import itertools


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

    light_curves = Table.read(file_name, format='fits').to_pandas()
    index_obs_separator = light_curves[light_curves['MJD'] == -777].index

    obs = np.cumsum(light_curves['MJD'] == -777) + 1

    light_curves.insert(0, 'obs', obs)
    light_curves.drop(index_obs_separator, inplace=True)
    light_curves.reset_index(inplace=True)
    light_curves.drop(columns='index', inplace=True)

    light_curves['BAND'] = light_curves.BAND.str.decode('utf-8')
    light_curves.name = file_name.split('/')[-1]

    return light_curves


def summary(dump_file):
    """
    Function that reads a CSV file containing supernova
    data and sumarizes the SNTYPE column
    """
    data = pd.read_csv(dump_file, delimiter=' ', header=5, usecols=[11])

    type_map = {1: 'Ia',
                20: 'II+IIP', 21: 'IIn+IIN', 22: 'IIL',
                32: 'Ib', 33: 'Ic+Ibc'}

    data = data.replace(type_map)

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

    min_MJD = lightcurve.groupby('obs').MJD.transform('min')

    days = lightcurve.MJD - min_MJD

    if inplace and ("Days" not in lightcurve.columns):
        lightcurve['Days'] = days

    if output:
        return days.to_numpy()


def peakmjd_to_days(lightcurve, dump_file, specific_obs=None, inplace=False,
                    output=False):
    """
    Function that transforms MJD dates to days, where day 0 corresponds to
    the moment of the peak.

    Input
    =====
    lightcurve: lightcurve data frame

    dump_file: file that contains the moment of the peak

    specific_obs: None or int (optional)
    if it is None, days will be calculted for all data in Data Frame

    inplace: bool (optional)
    if inplace is True, the function adds a new column labeled as "Days"
    to the dataFrame

    output: bool (optional)
    if it's True, an array of days will be returned
    """

    data = pd.read_csv(dump_file, delimiter=' ', header=5, usecols=[2, 4],
                       names=['obs', 'PEAKMJD'])

    if type(specific_obs) == int:
        lightcurve = lightcurve[lightcurve.obs == specific_obs]
        data = data[data.obs == specific_obs]

    data = pd.merge(lightcurve, data, on='obs')
    days = data.MJD - data.PEAKMJD

    if inplace and ("Days" not in lightcurve.columns):
        lightcurve['Days'] = days

    if output:
        return days.to_numpy()


def fitter_Bspline(curve, band, t_ev, order=3, w_power=1):
    curve_band = curve[curve.BAND == band]

    time = curve_band.Days
    flux = curve_band.FLUXCAL
    fluxerr = curve_band.FLUXCALERR

    spl = splrep(time, flux, w=1/(fluxerr ** w_power), k=order)
    flux_fit = splev(t_ev, spl)

    return flux_fit


def plotter(data_frame, obs, summary=None, days=False, dump=None):
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
        xlabel = 'Days'

        # if Days have not been calculated
        if 'Days' not in data_obs.columns:
            peakmjd_to_days(data_obs, dump, inplace=True, specific_obs=obs,
                            output=False)

    color = {'u ': 'purple', 'g ': 'green', 'r ': 'red',
             'i ': (150/255, 0, 0), 'z ': (60/255, 0, 0)}

    fig, ax = plt.subplots(figsize=(14, 8))

    for band in (data_obs['BAND'].value_counts()).index:
        data_to_plot = data_obs[data_obs.BAND == band]

        xdata_plot = data_to_plot.MJD
        if days:
            xdata_plot = data_to_plot.Days

        ax.errorbar(xdata_plot, data_to_plot.FLUXCAL,
                    yerr=data_to_plot.FLUXCALERR, marker='o',
                    ls='--', capsize=2, color=color[band], label=band)

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
    
    return fig, ax


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')