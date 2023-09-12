import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits, ascii
from scipy.interpolate import splrep, splev
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras import utils
import itertools


def reader(file_name, fits_header=False, band='BAND'):
    """
    Function that reads fits files and return a light-curves
    data frame

    Input
    =====
    file_name: str
        fits file name

    fits_header: bool
        if it is True, the header will be printed

    band: str (optional)
        name of the column related to the filter used for observation
    """

    if fits_header:
        with fits.open(file_name) as fits_file:
            header = fits_file[0].header
            print(repr(header))

    light_curves = Table.read(file_name, format='fits').to_pandas()
    index_obs_separator = light_curves[light_curves['MJD'] == -777].index

    obs = np.cumsum(light_curves['MJD'] == -777)

    light_curves.insert(0, 'obs', obs)
    light_curves.drop(index_obs_separator, inplace=True)
    light_curves.reset_index(inplace=True)
    light_curves.drop(columns='index', inplace=True)

    light_curves['BAND'] = light_curves[band].str.decode('utf-8')
    light_curves.name = file_name.split('/')[-1]

    return light_curves


def summary(head_file):
    """
    Function that reads a file containing supernova
    data and sumarizes the SNTYPE column

    Input
    =====

    head_file: str
       HEAD.FITS file
    """
    data = Table.read(head_file, format='fits').to_pandas()
    data['obs'] = data.index

    type_map = {101: 'Ia',
                20: 'II+IIP', 120: 'II+IIP',
                21: 'IIn+IIN', 121: 'IIn+IIN',
                22: 'IIL', 122: 'IIL',
                32: 'Ib', 132: 'Ib',
                33: 'Ic+Ibc', 133: 'Ic+Ibc'}

    data['SNTYPE'] = data['SNTYPE'].replace(type_map)

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


def peakmjd_to_days(lightcurve, head_file, specific_obs=None, inplace=False,
                    output=False):
    """
    Function that transforms MJD dates to days, where day 0 corresponds to
    the moment of the peak.

    Input
    =====
    lightcurve: pd.dataFrame
        lightcurve data frame

    head_file: str
        name of the file that contains the moment of the peak

    specific_obs: None or int (optional)
        if it is None, days will be calculted for all data in Data Frame

    inplace: bool (optional)
        if inplace is True, the function adds a new column labeled as "Days"
        to the dataFrame

    output: bool (optional)
        if it's True, an array of days will be returned
    """

    data = summary(head_file)

    lightcurve_index = lightcurve.index
    data_index = data.index

    if type(specific_obs) == int:
        lightcurve_index = lightcurve[lightcurve.obs == specific_obs].index
        data_index = data[data.obs == specific_obs].index

    data = pd.merge(lightcurve.iloc[lightcurve_index],
                    data.iloc[data_index],
                    on='obs')

    days = data.MJD - data.PEAKMJD

    if inplace:
        lightcurve.loc[lightcurve_index, 'Days'] = days.to_numpy()

    if output:
        return days.to_numpy()


def fitter_Bspline(curve, band, t_ev, order=5, w_power=2):
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

    order: int (optional, default=5)
        order of the spline (only values between 1 to 5)

    w_power: int (optional, default=2)
        power of the weight applicated to the incerteinty related to the fluxes
    """
    curve_band = curve[curve.BAND == band]

    time = curve_band.Days
    flux = curve_band.FLUXCAL
    fluxerr = curve_band.FLUXCALERR

    spl = splrep(time, flux, w=1/(fluxerr ** w_power), k=order)
    flux_fit = splev(t_ev, spl)

    return flux_fit


def preprocess(curves_file, band='BAND', head_file=None, min_obs=5,
               w_power=2, normalize=False):
    """
    Function that interpolates light curves, discarding curves that contain
    less than a certain amount of observation.

    Input
    =====
    curves_file: str or DataFrame
        name of the data file or dataFrame with the light curves

    head_file: None or str (optional)
        name of the file that add information related to the data

    spline_order: int (optional)
        order of the spline (only values between 1 to 5)

    min_obs: int (optional, default=5)
        quantity of minimum observation for discarding light curves

    normalize: bool
        if it is True, flux will be normalized
    """
    if type(curves_file) == str:
        curves = reader(curves_file, band=band)

        if not np.equal(head_file, None):
            peakmjd_to_days(curves, head_file, inplace=True, output=False)

        else:
            mjd_to_days(curves, inplace=True, output=False)

    else:
        curves = curves_file.copy()

        if "Days" not in curves.columns:
            print("""
                  Please, use mjd_to_days or peakmjd_to_days functions to
                  generate the Days column in the whole dataFrame
                  """)

            return None

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

    bands = ['g ', 'r ', 'i ', 'z ']

    curves_group = curves.groupby('obs')
    dict_curves_fitted = {}

    for obs, curve in curves_group:
        len_seq = 100
        t_ev = np.linspace(curve.Days.min(), curve.Days.max(), len_seq)
        dict_curve_fitted = {"Days": t_ev}

        for band in bands:
            if curve[curve.BAND == band].empty:
                flux_fitted = np.zeros(len_seq)

            else:
                flux_fitted = fitter_Bspline(curve, band, t_ev,
                                             order=min_obs,
                                             w_power=w_power)

                if normalize:
                    flux_fitted = utils.normalize(flux_fitted)[0]

            dict_curve_fitted[band] = flux_fitted

        dict_curves_fitted[obs] = dict_curve_fitted

    curves_fitted = pd.DataFrame.from_dict(dict_curves_fitted, orient='index')

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
    bands = ['g ', 'r ', 'i ', 'z ']
    columns = [col for col in curves_preprocessed.columns if col not in bands]

    for r in range(2, len(bands) + 1):
        for subset in itertools.combinations(bands, r):
            combinations.append(subset)

    df_augmentation = pd.DataFrame([])

    for combination in combinations:
        cols = [*columns, *combination]
        df_augmentation = pd.concat([df_augmentation,
                                     curves_preprocessed[cols]],
                                    ignore_index=True)

    return df_augmentation


def replace_nan_array(df_with_nan, array=np.zeros(100)):
    """
    Function that replace NaN values in a Data Frame by an array
    """
    df_without_nan = df_with_nan.copy()
    for column_name, column in df_with_nan.items():
        if not np.any(column.isna()):
            continue

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

    n_obs = curves.index.size
    n_seq = 100
    n_features = len(features)

    curves_RNN = np.reshape(curves_RNN, (n_obs, n_seq, n_features))

    if "Type" in curves.columns:
        types = curves.Type.to_numpy()
        types = types.reshape((-1, 1))

        return curves_RNN, types

    return curves_RNN, None


def plotter(data_frame, obs, days=False, head_file=None):
    """
    Function that plots supernova light curve

    Input
    =====
    data_frame: pd.dataFrame
        df with supernovae data

    obs: int
        observation number

    head_file (optional): None or str
        path to file with information related to nonIa supernovae (HEAD.FITS)

    days (optional): bool
        if it's True, MJD will be expressed as days
    """
    xlabel = 'MJD'
    df_copy = data_frame.copy()

    if days:
        xlabel = 'Days'

        # if Days have not been calculated
        if 'Days' not in df_copy.columns and not np.equal(head_file, None):
            peakmjd_to_days(df_copy, head_file, inplace=True,
                            specific_obs=obs, output=False)

    data_obs = df_copy[df_copy['obs'] == obs]

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
    if not np.equal(head_file, None):
        summ = summary(head_file)
        summ_obs = summ[summ.obs == obs]
        ax.set_title(rf"{name}, " +
                     rf"obs: {obs}, " +
                     rf"peak: {summ_obs['PEAKMJD'].values[0]}, " +
                     rf"SN type: {summ_obs['SNTYPE'].values[0]}")
    ax.legend()

    return fig, ax


def plotter_preprocess(dataframe, obs):
    """
    Function that plots supernova light curve from preprocess data frame

    Input
    =====
    data_frame: pd.dataFrame
        df with all supernovae data

    obs: int
        observation number
    """

    if "Type" not in dataframe.columns:
        title = f"obs: {obs}"
        days, g, r, i, z = dataframe.iloc[obs]

    else:
        title = f"obs: {obs} - type: {'Ia' if Type==1 else 'nonIa'}"
        days, g, r, i, z, Type = dataframe.iloc[obs]

    fig, ax = plt.subplots(figsize=(14, 8))

    color = {'u ': 'purple', 'g ': 'green', 'r ': 'red',
             'i ': (150/255, 0, 0), 'z ': (60/255, 0, 0)}

    ax.plot(days, g, color=color['g '], label='g')
    ax.plot(days, r, color=color['r '], label='r')
    ax.plot(days, i, color=color['i '], label='i')
    ax.plot(days, z, color=color['z '], label='z')

    ax.legend()
    ax.set_title(title)
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

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig, ax


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

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'r--')
    ax.plot(fpr, tpr, marker='o')
    ax.set_xlabel('False Positive rate')
    ax.set_ylabel('True Positive rate')
    ax.set_title('ROC Curve')

    return fig, ax
