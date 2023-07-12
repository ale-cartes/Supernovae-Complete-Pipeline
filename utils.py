import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import splrep, splev
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras import utils
import itertools


def reader(file_name, fits_header=False, band='BAND'):
    """
    Function that reads fits files and return a light-curves
    data frame

    Input
    ------
    file_name: str
        fits file name
    
    fits_header: bool
        if it is True, the header will be printed
    
    band: str (optional)
        name of the column related to the filter used for observation
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

    light_curves['BAND'] = light_curves[band].str.decode('utf-8')
    light_curves.name = file_name.split('/')[-1]

    return light_curves


def summary(dump_file):
    """
    Function that reads a CSV file containing supernova
    data and sumarizes the SNTYPE column

    Input
    =====

    dump_file: str
      dump_file's name
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
    lightcurve: pd.dataFrame
      lightcurve data frame

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
    lightcurve: pd.dataFrame
      lightcurve data frame

    dump_file: str
      name of the file that contains the moment of the peak

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
    """
    Function that interpolate data using B-splines and then
    evaluates it at a specific time

    Input
    =====
    curve: pd.dataFrame
        light curve data Frame

    band: str
        band of observation to be interpolated

    t_ev: np.array
        time at which the B-spline is evaluated

    order: int (optional)
        order of the spline
    
    w_power: 1
        power of the weight applicated to the incerteinty related to the fluxes
    """
    curve_band = curve[curve.BAND == band]

    time = curve_band.Days
    flux = curve_band.FLUXCAL
    fluxerr = curve_band.FLUXCALERR

    spl = splrep(time, flux, w=1/(fluxerr ** w_power), k=order)
    flux_fit = splev(t_ev, spl)

    return flux_fit


def preprocess(curves_file, dump_file=None, min_obs=5, normalize=False):
    """
    Function that interpolates light curves, discarding curves that contain
    less than a certain amount of observation.

    Input
    =====
    curves_file: str
        name of the data file
    
    dump_file: None or str (optional)
        name of the file that add information related to the data

    min_obs: int (optional)
        quantity of minimum observation for discarding light curves
    
    normalize: bool
        if it is True, flux will be normalized
    """
    curves = reader(curves_file)

    if not np.equal(dump_file, None):
        peakmjd_to_days(curves, dump_file, inplace=True, output=False)
    
    else:
        mjd_to_days(curves, inplace=True, output=False)

    columns = ['obs', 'MJD', 'Days', 'BAND', 'FLUXCAL', 'FLUXCALERR']

    curves_nonfiltered = curves[columns]

    # discard light curves with few observations in a band
    curves_nonfilt_group = curves_nonfiltered.groupby('obs')
    curves_band_counts = curves_nonfilt_group.BAND.value_counts()

    obs_discard = []
    for obs in curves_nonfiltered.obs.unique():
        if not (curves_band_counts[obs] > min_obs).all():
            obs_discard.append(obs)

    curves = curves_nonfiltered[~curves_nonfiltered.obs.isin(obs_discard)]

    global bands
    bands = ['g ', 'r ', 'i ', 'z ']

    curves_group = curves.groupby('obs')
    dict_curves_fitted = {}

    for obs, curve in curves_group:
        t_ev = np.linspace(curve.Days.min(), curve.Days.max(), 100)
        dict_curve_fitted = {"Days": t_ev}

        for band in bands:
            flux_fitted = fitter_Bspline(curve, band, t_ev, order=min_obs)

            if normalize:
                flux_fitted = utils.normalize(flux_fitted)
                
            dict_curve_fitted[band] = flux_fitted
        
        dict_curves_fitted[obs] = dict_curve_fitted

    curves_fitted = pd.DataFrame(dict_curves_fitted).transpose()

    return curves_fitted
    

def curves_augmentation(curves_preprocessed):
    """
    Function that combines different observations under different filters,
    adding additional data

    The combinations considered are:
    g  r  i  z
    g  r  i
    g  r  z
    g  i  z
    r  i  z
    g  r
    g  i
    g  z
    r  i
    r  z
    i  z

    Input
    =====
    curves_preprocessed: DataFrame
    """

    combinations = []
    for r in range(2, len(bands) + 1):
        for subset in itertools.combinations(bands, r):
            combinations.append(subset)
    
    df_augmentation = pd.DataFrame([])

    for combination in combinations:
        cols = ['Days', *combination]
        df_augmentation = pd.concat([df_augmentation,\
                                     curves_preprocessed[cols]],
                                    ignore_index=True)
    
    # curves = pd.concat([curves_preprocessed, df_augmentation])
    return df_augmentation


def replace_nan_array(df_with_nan, array=np.zeros(100)):
    """
    Function that replace NaN values by an array
    """
    df_without_nan = df_with_nan.copy()
    for column_name, column  in df_with_nan.items():
        if not np.any(column.isna()): continue

        column_copy = column.copy()
        for i, content in enumerate(column):
            if np.any(np.isnan(content)):
                column_copy[i] = array
            
        df_without_nan[column_name] = column_copy
    return df_without_nan


def RNN_reshape(curves):
    """
    Function that reshape the data in a way that the Neural Network can
    work with those
    """
    bands = ['g ', 'r ', 'i ', 'z ']
    features = ['Days', *bands]
    curves_RNN = curves[features].to_numpy().tolist()
    types = curves.Type.to_numpy()

    n_obs = curves.index.size
    n_seq = 100
    n_features = len(features)

    curves_RNN = np.reshape(curves_RNN, (n_obs, n_seq, n_features))
    types = types.reshape((-1, 1))

    return curves_RNN, types


    
def plotter(data_frame, obs, summary=None, days=False, dump=None):
    """
    Function that plots supernova lightcurve

    Input
    =====
    data_frame: pd.dataFrame
      df with all supernovae data

    obs: int
      observation number

    summary (optional): None or str
      path to file with information related to nonIa supernovae (.DUMP)

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
        data_to_plot = data_obs[data_obs['BAND'] == band]

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


def plot_confusion_matrix(test_data, pred_data, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Function that plots the Confusion Matrix given the testing
    and predicted data.

    Input
    =====
    test_data: np.array
      True labels from test set

    pred_data: np.array
      Predicted labels from the model

    normalize: bool (optional)
      if True, the confusion matrix will be normalized. Defaulte is False
    """

    cm = confusion_matrix(test_data, pred_data, labels=[1, 0])

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


def plot_roc_curve(test_data, pred_data, auc_print=False):
    """
    Function that plots the ROC curve given the testing
    and predicted data

    Input
    =====
    test_data: np.array
      True labels from test set

    pred_data: np.array
      Predicted labels from the model

    auc_print: bool (optional)
      if it is True, AUC will be printed
    """
    fpr, tpr, threshold = roc_curve(test_data, pred_data)

    if auc_print:
        print(f'AUC = {auc(fpr, tpr)}')

    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC Curve')
