import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])

from astropy.table import Table
from astropy.io import fits

from scipy.interpolate import splrep, splev
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import itertools

import keras
from keras import optimizers, initializers
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, GRU, LSTM, \
                         Dense, Bidirectional, Dropout, \
                         Flatten, BatchNormalization, \
                         Input, concatenate, Add
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy
import pickle
import pyhopper

import sncosmo as snc
import scipy.integrate as integrate
from scipy import optimize
import emcee
import corner

from numba import njit

from getdist import plots, MCSamples
from IPython.display import display, Math

seed = 42
keras.utils.set_random_seed(seed)

c = 299_792.458  # km/s

color_plot = {'u': 'purple', 'g': 'green', 'r': 'red',
              'i': (150/255, 0, 0), 'z': (60/255, 0, 0)}


def fitter_Bspline(curve, t_ev, order=5, w_power=1, normalize=True):
    """
    Function that interpolate data using B-splines and then
    evaluates it at a specific time

    Input
    =====
    curve: pd.dataFrame
        light curve data Frame

    t_ev: np.array
        time at which the B-spline is evaluated

    order: int (optional, default=5)
        order of the spline (only values between 1 to 5)

    w_power: int (optional, default=1)
        power of the weight applicated to the incerteinty related to the fluxes
    """

    time = curve.days
    flux = curve.FLUXCAL
    fluxerr = curve.FLUXCALERR

    spl = splrep(time, flux, w=1/(fluxerr ** w_power), k=order)
    flux_fit = splev(t_ev, spl)

    if np.isnan(flux_fit).any():
        flux_fit = np.zeros(t_ev.shape)

    if normalize:
        flux_fit /= np.max(np.abs(flux_fit))

    return flux_fit


class SN_data:
    """
    Class for handling supernova data from FITS files. This class provides
    methods to read, summarize, preprocess, and plot light curves data.

    Attributes
    ==========

    phot_file: str
        path to the photometry file

    head_file: str
        path to the head file

    lc_df: pd.DataFrame
        light curve data frame

    obs_info: pd.DataFrame
        observation information data frame

    bands: list
        list of bands used in the light curves

    lc_fitted: pd.DataFrame
        fitted light curve data frame

    obs_discarded: list
        list of observations discarded in the preprocessing

    len_seq: int
        number of points in the interpolation process

    Methods
    =======

    reader(fits_header=False, band_col='BAND')
        Function that reads fits files and return a light-curves
        data frame

    obs_summary()
        Function that reads the head file containing supernova
        data and sumarizes the SNTYPE column

    mjd_to_days()
        Function that transforms MJD dates to days considering
        initial observation as day 0

    peakmjd_to_days()
        Function that transforms MJD dates to days, where day 0 corresponds to
        the moment of the peak. Head file with peak information should be given
        as attribute

    preprocess(min_obs=5, w_power=1, len_seq=100, z_host=True, normalize=True)
        Function that interpolates light curves, discarding curves that contain
        less than a certain amount of observation.

    plotter(obs, days=True, fitted=False)
        Function that plots supernova light curve
    """

    def __init__(self, phot_file, head_file=None):
        self.phot_file = phot_file
        self.head_file = head_file

    def reader(self, fits_header=False, band_col='BAND'):
        """
        Function that reads fits files and return a light-curves
        data frame

        Input
        =====
        fits_header: bool
            if it is True, the header will be printed

        band_col: str (optional)
            name of the column related to the filter used for observation
        """
        if fits_header:
            header = fits.getheader(self.phot_file)
            print(repr(header))

        light_curves = Table.read(self.phot_file, format='fits').to_pandas()
        index_obs_separator = light_curves[light_curves['MJD'] == -777].index

        obs = np.cumsum(light_curves['MJD'] == -777)

        light_curves.insert(0, 'obs', obs)
        light_curves.drop(index_obs_separator, inplace=True)
        light_curves.set_index('obs', inplace=True)

        light_curves['BAND'] = (light_curves[band_col]
                                .str.decode('utf-8')
                                .str.strip()
                                )
        light_curves.name = self.phot_file.split('/')[-1]

        # non-detected points (PHOTFLAG == 0) are discarded
        self.lc_df = light_curves[light_curves['PHOTFLAG'] != 0] 
        self.bands = light_curves.BAND.unique()

    def obs_summary(self):
        """
        Function that reads the head file containing supernova
        data and sumarizes the SNTYPE column
        """
        if np.equal(self.head_file, None):
            print("This function only works if a Head file is provided")
            return None

        obs_info = Table.read(self.head_file, format='fits').to_pandas()
        obs_info['obs'] = obs_info.index
        obs_info.set_index('obs', inplace=True)

        type_map = {101: 'Ia',
                    20: 'II+IIP', 120: 'II+IIP',
                    21: 'IIn+IIN', 121: 'IIn+IIN',
                    22: 'IIL', 122: 'IIL',
                    32: 'Ib', 132: 'Ib',
                    33: 'Ic+Ibc', 133: 'Ic+Ibc'}

        obs_info['SNTYPE'] = obs_info['SNTYPE'].replace(type_map)

        self.obs_info = obs_info
        self.lc_df = pd.merge(self.lc_df, obs_info, on='obs')

    def mjd_to_days(self):
        """
        Function that transforms MJD dates to days considering
        initial observation as day 0
        """

        min_MJD = self.lc_df.groupby('obs').MJD.transform('min')
        days = self.lc_df.MJD - min_MJD
        self.lc_df['days'] = days

    def peakmjd_to_days(self):
        """
        Function that transforms MJD dates to days, where day 0 corresponds to
        the moment of the peak. Head file with peak information should be given
        as attribute
        """
        if np.equal(self.head_file, None):
            print("This function only works if a Head file is provided")
            return None

        try:
            self.obs_info
        except:
            self.obs_summary()

        days = self.lc_df.MJD - self.lc_df.PEAKMJD
        self.lc_df['days'] = days

    def preprocess(self, min_obs=5, w_power=1, len_seq=100, z_host=True,
                   normalize=True):
        """
        Function that interpolates light curves, discarding curves that contain
        less than a certain amount of observation.

        Input
        =====
        min_obs: int (default=5)
            quantity of minimum observation for discarding light curve bands

        w_power: int (default=1)
            power weights to inverse error value, i.e., w=1/yerr^w
            (lower error implies a greater weight)

        len_seq: int (default=100)
            number of points in interpolation process

        z_host: bool (default: True)
            if it is True the Host redshift data will be added as a new column
        """
        if not np.equal(self.head_file, None):
            self.obs_summary()
            self.peakmjd_to_days()

        else:
            self.mjd_to_days()

        curves_group = self.lc_df.groupby('obs')

        dict_curves_fitted = {}
        zero_array = np.zeros(len_seq)
        obs_discarded = []

        for obs, curve in curves_group:
            day_min = np.nanmax([curve[curve.BAND == band].days.min()
                                 for band in self.bands])
            day_max = np.nanmin([curve[curve.BAND == band].days.max()
                                 for band in self.bands])

            t_ev = np.linspace(day_min, day_max, len_seq)

            dict_curve_fitted = {"days": t_ev}

            if ('SIM_REDSHIFT_HOST' in curve.columns) and z_host:
                z_host_obs = curve.SIM_REDSHIFT_HOST.unique()[0]
                dict_curve_fitted['z_host'] = np.repeat(z_host_obs,
                                                        len_seq)

            if not np.equal(self.head_file, None):
                sn_type = curve.SNTYPE.unique()[0]
                hot_encoder = (1 if sn_type == 'Ia' else 0)
                dict_curve_fitted['sn_type'] = hot_encoder

            for band in self.bands:
                band_data = curve[curve.BAND == band]
                if band_data.empty or (band_data.shape[0] <= min_obs):
                    flux_fitted = zero_array

                else:
                    flux_fitted = fitter_Bspline(band_data, t_ev,
                                                 order=min_obs,
                                                 w_power=w_power,
                                                 normalize=normalize)

                    if pd.isna(flux_fitted).any():
                        flux_fitted = zero_array

                dict_curve_fitted[band] = flux_fitted

            if not np.all([np.equal(dict_curve_fitted[band], zero_array)
                          for band in self.bands]):
                dict_curves_fitted[obs] = dict_curve_fitted

            else:
                obs_discarded.append(obs)

        curves_fitted = pd.DataFrame.from_dict(dict_curves_fitted,
                                               orient='index')
        self.lc_fitted = curves_fitted
        self.obs_discarded = obs_discarded
        self.len_seq = len_seq

    def plotter(self, obs, days=True, fitted=False, ls='--'):
        """
        Function that plots supernova light curve

        Input
        =====
        obs: int
            observation number

        days (optional): bool
            if it's True, x label will be expressed as days
            if it's False, x label will be expressed as MJD
        """

        if days:
            try:
                self.lc_df.days
            except:
                if not np.equal(self.head_file, None):
                    self.peakmjd_to_days()

                else:
                    self.mjd_to_days()

        if fitted:
            try:
                self.lc_fitted
            except:
                self.preprocess()

            data_obs = self.lc_fitted[self.lc_fitted.index == obs]
            if data_obs.empty:
                if obs in self.obs_discarded:
                    print(f"Obs: {obs} was discarded because was not possible "
                          "to fit it")

                else:
                    print(f"Obs: {obs} was not found, try with another one")

                return None

            fig, ax = plt.subplots(figsize=(14, 8))
            data_obs = self.lc_fitted[self.lc_fitted.index == obs]

            for band in self.bands:
                ax.plot(*data_obs.days.values, *data_obs[band].values,
                        color=color_plot[band], label=band)

            ax.set_xlabel('Days', fontsize=18)

        else:
            data_obs = self.lc_df[self.lc_df.index == obs]

            if data_obs.empty:
                print(f"Obs: {obs} was not found, try with another one")
                return None

            fig, ax = plt.subplots(figsize=(14, 8))
            for band in np.unique(data_obs['BAND']):
                data_to_plot = data_obs[data_obs['BAND'] == band]

                xaxis_plot = data_to_plot.days if days else data_to_plot.MJD
                xlabel = 'Days' if days else 'MJD'

                ax.errorbar(xaxis_plot, data_to_plot.FLUXCAL,
                            yerr=data_to_plot.FLUXCALERR, marker='o',
                            ls=ls, capsize=2, color=color_plot[band],
                            label=band)

            ax.set_xlabel(xlabel, fontsize=18)

        ax.set_ylabel('Flux (ADU)', fontsize=18)
        ax.grid(ls=':', alpha=0.3)
        ax.legend()
        return fig, ax


class NN_classifier:
    """
    Class for handling Neural Network classifier for supernova data. This class
    provides methods to preprocess, split data, train, evaluate, and plot the
    classifier performance.

    Attributes
    ==========

    sn_classes: list
        list of SN_data classes

    name: str
        name of the classifier

    data: pd.DataFrame
        data frame with all the light curves

    X_train, X_val, X_test: pd.DataFrame
        data frame with the features for training, validation and test data

    y_train, y_val, y_test: pd.DataFrame
        data frame with the labels for training, validation and test data

    X_train_nn, X_val_nn, X_test_nn: np.array
        numpy array with the features for training, validation and test data
        with the shape (n_obs, n_seq, n_features)

    y_train_nn, y_val_nn, y_test_nn: np.array
        numpy array with the labels for training, validation and test data
        with the shape (n_obs, 1)

    model: keras.Model
        Neural Network model

    fit_hist: list
        list with the history of the training process

    train_stats, val_stats, test_stats: list
        list with the mean and standard deviation of the predictions

    train_preds, val_preds, test_preds: np.array
        numpy array with the predictions for training, validation and test data

    Methods
    =======

    data_sample(frac, seed=42)
        Function that samples the data

    train_test_split(train_size=0.7, val_size=0.15, rand_state=42)
        Function that splits the data into training, validation and test data

    NN_reshape(data_ext=None)
        Function that reshape the data in a way that the Neural Network can
        work with those

    model_nl_creator(n=None, rnns_i=[1, 1, 1], neurons=[8, 8, 8],
                     activations_i=[1, 1, 1], init_weights_i=[0, 0, 0],
                     dropout=0.2, optimizer=optimizers.Adam, lr=1e-3,
                     plot_model=False)
        Function that creates a Neural Network model with the given
        hyperparameters using the Functional API of Keras

    model_fit(epochs=200, batch_size=8, plot=True, verbose=1, patience=15)
        Function that fits the Neural Network model

    best_hyp_pyhopper(search_params, model_creator, time=None, steps=None,
                      patience=15, plot_loss=False, plot_bf=True, verbose=0,
                      epochs=250, nwrap=5, pruner=0.75, n_jobs=1, save=False,
                      load=False)
        Function that uses Pyhopper to find the best hyperparameters for the
        Neural Network model

    model_statistics(num_it=10, batch_size=8, epochs=250, verbose_fit=0,
                     patience=15, load=False)
        Function that evaluates the Neural Network model in a certain number
        of iterations  and returns the mean and standard deviation of the
        predictions

    training_loss_plot()
        Function that plots the training history of the Neural Network model

    plot_roc_curve()
        Function that plots the ROC curve for train, validation and test data

    plot_confusion_matrix(normalize=False)
        Function that plots the Confusion Matrix for train, validation and
        test data
    """

    def __init__(self, sn_classes, name=None):
        data = pd.concat([sn_class.lc_fitted for sn_class in sn_classes])
        data = pd.concat([data, pd.DataFrame(columns=['g', 'r', 'i', 'z'])])

        data['obs'] = data.index
        data.reset_index(inplace=True, drop=True)
        self.data = data
        self.name = name

    def data_sample(self, frac, seed=42):
        """
        Function that samples the data with a certain fraction and seed

        Input
        =====
        frac: float
            fraction of the data to sample
        
        seed: int (optional, default=42)
            seed for the random state
        """
        if frac < 0 or frac > 1:
            print("Fraction must be between 0 and 1")
            return None

        self.data = self.data.sample(frac=frac, random_state=seed)

    def train_test_split(self, train_size=0.7, val_size=0.15, rand_state=42):
        data_wo_types = self.data.drop(columns=['sn_type', 'obs'])

        if data_wo_types.isna().any().any():
            len_seq = self.data.days[0].shape[0]
            array = np.zeros(len_seq)
            data_wo_types = data_wo_types.map(lambda x: array
                                              if np.array(pd.isnull(x)).any()
                                              else x)

        train_test = train_test_split(data_wo_types,
                                      self.data.sn_type,
                                      train_size=train_size,
                                      random_state=rand_state)

        X_train, X_test, y_train, y_test = train_test

        test_size = 1 - train_size - val_size
        test_size = test_size / (test_size + val_size)

        val_test = train_test_split(X_test, y_test,
                                    test_size=test_size,
                                    random_state=rand_state)

        X_val, X_test, y_val, y_test = val_test

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def NN_reshape(self, data_ext=None):
        """
        Function that reshape the data in a way that the Neural Network can
        work with those

        data_ext: optional (default: None)
            external data to be reshaped
        """
        def func(data):
            shape = np.array(data.shape)

            if shape.size == 1:
                return data.values.reshape((-1, 1))

            else:
                n_obs, n_features = shape
                n_seq = data.values[0, 0].shape[0]

                data_RNN = data.to_numpy().tolist()
                return np.reshape(data_RNN, (n_obs, n_seq, n_features))

        if np.all(np.equal(data_ext, None)):
            self.X_train_nn, self.y_train_nn = (func(self.X_train),
                                                func(self.y_train))
            self.X_val_nn, self.y_val_nn = func(self.X_val), func(self.y_val)
            self.X_test_nn, self.y_test_nn = (func(self.X_test),
                                              func(self.y_test))

        else:
            return func(data_ext)

    def model_nl_creator(self, n=None, rnns_i=[1, 1, 1], neurons=[8, 8, 8],
                         activations_i=[1, 1, 1],
                         init_weights_i=[0, 0, 0],
                         dropout=0.2,
                         optimizer=optimizers.Adam, lr=1e-3,
                         plot_model=False):
        """
        Function that creates a Neural Network model with the given
        hyperparameters using the Functional API of Keras

        Input
        =====
        n: int (optional, default=None)
            number of layers of the Neural Network

        rnns_i: list (optional, default=[1, 1, 1])
            list with the index of the RNNs to be used in the model.
            0: SimpleRNN, 1: LSTM, 2: GRU

        neurons: list (optional, default=[8, 8, 8])
            list with the number of neurons for each layer

        activations_i: list (optional, default=[1, 1, 1])
            list with the index of the activation functions to be used in
            the model.
            0: linear, 1: tanh, 2: relu, 3: sigmoid, 4: softmax

        init_weights_i: list (optional, default=[0, 0, 0])
            list with the index of the initializers to be used in the model.
            0: he_uniform, 1: RandomUniform, 2: GlorotUniform

        dropout: float (optional, default=0.2)
            dropout rate

        optimizer: keras.optimizers (optional, default=optimizers.Adam)
            optimizer to be used in the model

        lr: float (optional, default=1e-3)
            learning rate

        plot_model: bool (optional, default=False)
            if it is True, the model will be plotted
        """

        n = len(rnns_i) if n is None else n

        rnn_var = [SimpleRNN, LSTM, GRU]
        activation_var = ['linear', 'tanh', 'relu', 'sigmoid', 'softmax']
        init_weight_var = [initializers.he_uniform(seed=seed),
                           initializers.RandomUniform(seed=seed),
                           initializers.GlorotUniform(seed=seed)]

        inputs = Input(shape=self.X_train_nn.shape[1:])
        layers = inputs

        for i in np.arange(n):
            rnn = rnn_var[rnns_i[i]]

            units = int(neurons[i])
            activation = activation_var[activations_i[i]]
            kernel_init = init_weight_var[init_weights_i[i]]

            ret_seq = False if i + 1 == n else True

            layers = rnn(units=units, activation=activation,
                         kernel_initializer=kernel_init,
                         return_sequences=ret_seq)(layers)

            layers = BatchNormalization()(layers)
            layers = Dropout(dropout)(layers)

        layers = Flatten()(layers)
        layers = Dense(self.y_train_nn.shape[1],
                       activation='sigmoid')(layers)

        model = Model(inputs, layers, name=f'{n}_layer')
        model.compile(optimizer=optimizer(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=[BinaryAccuracy()])

        self.model = model
        self.fit_hist = []

        if plot_model:
            folder = f"data_folder/images/model_{self.name}.pdf"
            keras.utils.plot_model(self.model, to_file=folder,
                                   show_shapes=True)

        return model

    def model_fit(self, epochs=200, batch_size=8, plot=True,
                  verbose=1, patience=15):
        """
        Function that fits the Neural Network model with the given
        hyperparameters

        Input
        =====
        epochs: int (optional, default=200)
            number of epochs

        batch_size: int (optional, default=8)
            batch size

        plot: bool (optional, default=True)
            if it is True, the training history will be plotted

        verbose: int (optional, default=1)
            verbose level

        patience: int (optional, default=15)
            patience for the early stopping
        """

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        hist = self.model.fit(self.X_train_nn, self.y_train_nn,
                              validation_data=(self.X_val_nn, self.y_val_nn),
                              epochs=epochs, batch_size=batch_size,
                              callbacks=[early_stopping], verbose=verbose)

        self.fit_hist.append(hist)

        if plot:
            self.training_loss_plot()

    def best_hyp_pyhopper(self, model_creator, search_params=None,
                          time=None, steps=None, patience=15,
                          plot_loss=False, plot_bf=True, verbose=0,
                          epochs=250, nwrap=5, pruner=0.75,
                          n_jobs=1, save=False, load=True):
        """
        Function that uses Pyhopper to find the best hyperparameters for the
        Neural Network model

        Input
        =====
        model_creator: function
            function that creates the Neural Network model

        search_params: pyhopper.Search (if load is False this parameter must be
        provided)
            search parameters for the Pyhopper

        time: int (optional, default=None)
            time in pyhopper format for the optimization

        steps: int (optional, default=None)
            number of steps for the optimization

        patience: int (optional, default=15)
            patience for the early stopping

        plot_loss: bool (optional, default=False)
            if it is True, the loss will be plotted

        plot_bf: bool (optional, default=True)
            if it is True, the best so far will be plotted

        verbose: int (optional, default=0)
            verbose level

        epochs: int (optional, default=250)
            number of epochs

        nwrap: int (optional, default=5)
            number of times the model will be wrapped

        pruner: float (optional, default=0.75)
            pruner for the Pyhopper

        n_jobs: int (optional, default=1)
            number of jobs for parallelization

        save: bool (optional, default=False)
            if it is True, the search parameters will be saved

        load: bool (optional, default=True)
            if it is True, the search parameters will be loaded

        Output
        =====
        best_params: dict
            best hyperparameters for the Neural Network model

        batch_size: int
            best batch size for the Neural Network model
        """
        if not load and search_params is None:
            print("If load is False, search_params must be provided")
            return None

        def model_to_pyhopper(param_grid):
            model_creator(**{key: value for key, value in param_grid.items()
                             if key != 'batch_size'})
            self.model_fit(plot=plot_loss, verbose=verbose,
                           epochs=epochs, batch_size=param_grid['batch_size'],
                           patience=patience)

            return self.model.evaluate(self.X_val_nn, self.y_val_nn,
                                       verbose=verbose)[1]

        obj_func = pyhopper.wrap_n_times(model_to_pyhopper, n=nwrap)
        pruner = pyhopper.pruners.QuantilePruner(pruner)

        folder = f"./data_folder/checkpoints/classifier_{self.name}.ckpt"

        if load:
            search_params = pyhopper.Search()
            search_params.load(folder)
            best_params = search_params.best

        else:
            cktp_file = folder if save else None
            best_params = search_params.run(obj_func, 'max',
                                            steps=steps, runtime=time,
                                            pruner=pruner, n_jobs=n_jobs,
                                            checkpoint_path=cktp_file)

        print(f"Best params: {best_params}")

        if 'batch_size' in best_params:
            batch_size = best_params['batch_size']
            del best_params['batch_size']

        if plot_bf:
            fig, ax = plt.subplots(figsize=(8, 5))

            steps = np.array(search_params.history.steps) + 1
            fs = search_params.history.fs
            print(f"Steps: {max(steps)} - Best fs: {max(fs):0.3}")

            ax.scatter(x=steps, y=fs, label="Sampled")

            ax.plot(steps, search_params.history.best_fs,
                    ls='--', color="red",
                    label="Best so far", zorder=0)

            ax.grid(ls=':', alpha=0.4, zorder=0)
            ax.set(xlim=[0.5, len(search_params.history) + 0.5],
                   xlabel='Step',
                   ylabel='Validation Accuracy')

            ax.legend()

            folder = ("data_folder/images/"
                      f"pyhopper_opt_classifier_{self.name}.svg")
            fig.savefig(folder, transparent=True, bbox_inches='tight')

        return best_params, batch_size

    def model_statistics(self, num_it=10, batch_size=8, epochs=250,
                         verbose_fit=0, patience=15, load=False):
        """
        Function that evaluates the Neural Network model in a certain number
        of iterations and returns the mean and standard deviation of the
        predictions

        Input
        =====
        num_it: int (optional, default=10)
            number of iterations

        batch_size: int (optional, default=8)
            batch size

        epochs: int (optional, default=250)
            number of epochs

        verbose_fit: int (optional, default=0)
            verbose level

        patience: int (optional, default=15)
            patience for the early stopping

        load: bool (optional, default=False)
            if it is True, the weights will be loaded
        """
        train_preds = []
        val_preds = []
        test_preds = []

        file = f"./data_folder/weights/classifier_weights_{self.name}.pkl"
        if load:
            file_weights = open(file, "rb")
            weights, fit_hist = pickle.load(file_weights)
            self.fit_hist = fit_hist
            num_it = len(weights)
            file_weights.close()

        else:
            initial_weights = self.model.get_weights()
            weights = []

        for i in range(num_it):
            if i == 0:
                print(f"{i}/{num_it}", end="\r")

            if load:
                self.model.set_weights(weights[i])

            else:
                self.model.set_weights(initial_weights)

                self.model_fit(epochs=epochs, batch_size=batch_size,
                               plot=False, verbose=verbose_fit,
                               patience=patience)
                weights.append(self.model.get_weights())

            verbose_pred = 1 if i == num_it - 1 else 0
            pred_train = self.model.predict(self.X_train_nn,
                                            verbose=verbose_pred)
            pred_val = self.model.predict(self.X_val_nn,
                                          verbose=verbose_pred)
            pred_test = self.model.predict(self.X_test_nn,
                                           verbose=verbose_pred)

            train_preds.append(pred_train)
            val_preds.append(pred_val)
            test_preds.append(pred_test)

            print(f"{i + 1}/{num_it}", end="\r")

        file_weights = open(file, "wb")
        pickle.dump([weights, self.fit_hist], file_weights)
        file_weights.close()

        means_preds_train = np.mean(train_preds, axis=0)
        stds_preds_train = np.std(train_preds, axis=0)

        means_preds_val = np.mean(val_preds, axis=0)[0]
        stds_preds_val = np.std(val_preds, axis=0)[0]

        means_preds_test = np.mean(test_preds, axis=0)
        stds_preds_test = np.std(test_preds, axis=0)

        self.train_stats = [means_preds_train, stds_preds_train]
        self.val_stats = [means_preds_val, stds_preds_val]
        self.test_stats = [means_preds_test, stds_preds_test]

        self.train_preds = np.array(train_preds)
        self.val_preds = np.array(val_preds)
        self.test_preds = np.array(test_preds)
        
        self.NN_weights = weights

    def training_loss_plot(self):
        """
        Function that plots the training history of the Neural Network model
        """

        keys = self.fit_hist[0].history.keys()
        ncols = len(keys) // 2

        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharey=True,
                               figsize=(10, 5))

        for i, key in enumerate(keys):
            values = [hist.history[key] for hist in self.fit_hist]
            values = pd.DataFrame(values)

            index = i // ncols

            x = range(values.shape[1])
            y = values.mean(axis=0)
            yerr = values.std(axis=0, ddof=0)

            ax[index].plot(x, y, marker='.', ls='-',
                           label=f"Mean\n{key[0:10]}")

            ax[index].fill_between(x, y-yerr, y+yerr, alpha=0.25,
                                   label=rf"$\pm 1 \sigma$")

            if i % 2 == 1:
                ax[index].set_xlabel('Epochs')

                if i < 2:
                    ax[index].legend(ncol=1, loc='right')
                ax[index].grid(ls=':', alpha=0.4, zorder=0)

                if ncols > 1:
                    title = 'Validation set' if 'val' in key else 'Train set'
                    ax[index].set_title(title)

        ax[0].set_ylim(-0, 1)
        ax[0].set_ylabel('Accuracy / Loss')
        fig.suptitle(f"Train History")
        fig.subplots_adjust(wspace=0.05)

        folder = f"data_folder/images/train_loss_classifier_{self.name}.svg"
        fig.savefig(folder, transparent=True, bbox_inches='tight')

        return fig, ax

    def plot_roc_curve(self):
        """
        Function that plots the ROC curve for train, validation
        and test data
        """
        X_data = [self.train_preds, self.val_preds, self.test_preds]
        y_data = [self.y_train_nn, self.y_val_nn, self.y_test_nn]

        data_label = ['Train data', 'Validation data', 'Test data']

        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True,
                               figsize=(12, 4))

        for i, (X, y) in enumerate(zip(X_data, y_data)):
            roc_curves = [roc_curve(y_true=y, y_score=y_pred)
                          for y_pred in X]

            fpr_interp = np.linspace(0, 1, 100)
            tprs = []
            aucs = []

            for single_roc in roc_curves:
                fpr, tpr, threshold = single_roc
                aucs.append(auc(fpr, tpr))

                tpr_interp = np.interp(fpr_interp, fpr, tpr)
                tpr_interp[0] = 0

                tprs.append(tpr_interp)

                ax[i].plot([0, 1], [0, 1], 'r--')
                ax[i].plot(fpr, tpr, 'b--', alpha=0.1)

            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0, ddof=0)

            ax[i].plot(fpr_interp, mean_tpr, 'b', label='Mean ROC')
            ax[i].fill_between(x=fpr_interp,
                               y1=mean_tpr - std_tpr,
                               y2=mean_tpr + std_tpr,
                               color='grey', alpha=0.5,
                               label=r'$\pm 1 \sigma$',
                               zorder=0)

            ax[i].set(xlabel='False Positive rate', title=data_label[i])
            ax[i].grid(ls=':', alpha=0.3)

            mean_auc = np.mean(aucs, axis=0)
            std_auc = np.std(aucs, axis=0, ddof=0)

            if i == 2:
                ax[i].text(x=0.08, y=-0.02,
                           s=fr"AUC: {mean_auc:0.3f}$\pm${std_auc:0.3f}")
                ax[i].plot([0, 1], [0, 1], 'r--', label='Random\nclassifier')
                ax[i].legend(loc='lower right')

            else:
                ax[i].text(x=0.55, y=-0.02,
                           s=fr"AUC: {mean_auc:0.3f}$\pm${std_auc:0.3f}")

        ax[0].set_ylabel('True Positive rate')
        fig.suptitle('ROC Curve')
        fig.subplots_adjust(wspace=0.05)

        folder = f"data_folder/images/roc_curve_classifier_{self.name}.svg"
        fig.savefig(folder, transparent=True, bbox_inches='tight')

        return fig, ax

    def plot_confusion_matrix(self, normalize=False):
        """
        Function that plots the Confusion Matrix for train, validation and
        test data.

        Input
        =====
        normalize: bool (optional)
            if True, the confusion matrix will be normalized. Defaulte is False
        """
        classes = ['Ia', 'no Ia']

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
        data_label = ['Train data', 'Validation data', 'Test data']

        X_data = [self.train_preds, self.val_preds, self.test_preds]
        y_data = [self.y_train_nn, self.y_val_nn, self.y_test_nn]

        for i_l, label in enumerate(data_label):
            cms = [confusion_matrix(y_data[i_l], X_data[i_l][j].round(),
                                    labels=[1, 0])
                   for j in range(X_data[i_l].shape[0])]

            cm_mean = np.array(cms).mean(axis=0)
            cm_std = np.array(cms).std(axis=0, ddof=0)

            if normalize:
                norm = cm_mean.sum(axis=1)[:, np.newaxis]
                cm_mean = cm_mean / norm
                cm_std = cm_std / norm

            im = ax[i_l].imshow(cm_mean, interpolation='nearest',
                                cmap=plt.cm.Blues,
                                vmin=0, vmax=1 if normalize else None)

            fmt = lambda x, pos: f'{int(x):,}'.replace(',', '.')
            fmt_norm = lambda x, pos: f'{x:0.1f}'
            fig.colorbar(im, ax=ax[i_l], shrink=0.75,
                         format=fmt_norm if normalize else fmt)

            tick_marks = np.arange(len(classes))
            ax[i_l].set_xticks(tick_marks, classes, rotation=45)
            ax[i_l].set_yticks(tick_marks, classes)

            thresh = cm_mean.max() / 2.
            for i, j in itertools.product(range(cm_mean.shape[0]),
                                          range(cm_mean.shape[1])):
                if normalize:
                    text = rf"${cm_mean[i, j]:0.2f} \pm {cm_std[i, j]:0.2f}$"
                else:
                    text = rf"${int(cm_mean[i, j])} \pm {int(cm_std[i, j])}$"
                ax[i_l].text(j, i, text,
                             horizontalalignment="center",
                             color=("white" if cm_mean[i, j] > thresh
                                    else "black")
                             )

            ax[i_l].set(xlabel='Predicted label', title=label)

        ax[0].set_ylabel('True label')
        fig.suptitle('Confusion matrix')
        fig.subplots_adjust(wspace=0.05)

        file_name = f"{self.name}_norm" if normalize else self.name
        folder = ("data_folder/images/"
                  f"confusion_matrix_classifier_{file_name}.svg")
        fig.savefig(folder, transparent=True, bbox_inches='tight')

        return fig, ax


class ExternalData:
    """
    Class for handling external data for the Neural Network classifier.
    This class provides methods to preprocess, predict, and plot the
    classifier performance.

    Attributes
    ==========

    sn_class: list
        list of SN_data classes

    nn_class: NN_classifier
        Neural Network classifier class

    load_weights_model(i=0):
        Function that loads the weights of the Neural Network classifier
        calculated in the training process using model_statistics method
        from the nn_class

    y_pred: np.array
        numpy array with the predictions

    y_true: np.array (if available)
        numpy array with the true labels

    nan_index: np.array
        numpy array with the index of NaN values
    """
    def __init__(self, sn_class, nn_class):
        self.sn_class = sn_class
        self.nn_class = nn_class
    
    def load_weights_model(self, i=0):
        """
        Function that loads the weights of the Neural Network classifier
        calculated in the training process using model_statistics method

        Input
        =====
        i: int (optional, default=0)
            index of the weights to be loaded
        """
        file = ("./data_folder/weights/"
                f"classifier_weights_{self.nn_class.name}.pkl"
                )
        file_weights = open(file, "rb")
        weights, fit_hist = pickle.load(file_weights)

        if i > len(weights) - 1 or i < 0:
            print(f"Index {i} is out of range, the last index is "
                  f"{len(weights) - 1}")
            return None

        print(f"Weights loaded: {i} of {len(weights) - 1}")

        self.nn_class.model.set_weights(weights[i])
        file_weights.close()

    def model_prediction(self):
        """
        Function that predicts the labels for the external data considering
        the NN model in nn_class
        """

        data = pd.concat([sn_class.lc_fitted
                          for sn_class in self.sn_class])
        data = pd.concat([data, pd.DataFrame(columns=['g', 'r', 'i', 'z'])],
                         ignore_index=True)

        if 'sn_type' in data.columns:
            self.y_true = data['sn_type']
            data.drop(columns=['sn_type'], inplace=True)

        if data.isna().any().any():
            len_seq = data.days[0].shape[0]
            array = np.zeros(len_seq)
            data = data.map(lambda x: array
                            if np.array(pd.isnull(x)).any()
                            else x)

        data_reshaped = self.nn_class.NN_reshape(data_ext=data)

        y_pred = self.nn_class.model.predict(data_reshaped)
        self.y_pred = y_pred

    def df_lc_class_Ia(self, prob_Ia=0.85, confusion_info=True):
        if not hasattr(self, 'y_pred'):
            self.model_prediction()
        
        df_all_lc = pd.DataFrame()
        
        for i, sn_class in enumerate(self.sn_class):
            # obtain observations that were not discarded
            obs_list = sn_class.lc_df.index
            obs_mask = ~np.isin(obs_list, sn_class.obs_discarded)
            df_lc = sn_class.lc_df.loc[obs_mask].copy()
            
            # modify index to avoid duplicates
            max_index = df_all_lc.index.max() if not df_all_lc.empty else df_lc.index.max()
            df_lc.index = np.array(df_lc.index) + (max_index + 1)

            # concatenate all light curves
            df_all_lc = pd.concat([df_all_lc, df_lc])
        
        size_y_pred = self.y_pred.shape[0]
        size_df_lc = df_all_lc.index.unique().size
        print(f"Size of y_pred: {size_y_pred} - Size of fitted LC: {size_df_lc}")
        
        if size_y_pred != size_df_lc:
            print("The size of the prediction and the light curve data "
                  "do not match")
            return None
        
        index_classified_Ia = []
        
        for prob, obs in zip(self.y_pred, df_all_lc.index.unique()):
            if prob > prob_Ia:
                index_classified_Ia.append(obs)
        
        mask_index = np.isin(df_all_lc.index, index_classified_Ia)

        if confusion_info:
            if not hasattr(self, 'y_true'):
                print("True labels are not available. Confusion info cannot be generated")
            
            else:
                y_true = self.y_true
                y_pred = self.y_pred
                y_pred = y_pred.reshape(len(y_pred))

                confusion_df = pd.DataFrame({'true': y_true, 'predicted': y_pred.flatten()})
                filtered_df = confusion_df.loc[confusion_df['predicted'] > prob_Ia, 'true']
                filtered_df = filtered_df.replace({1: 'Ia', 0: 'noIa'})
                
                print(f"SN classified as Ia with probability > {prob_Ia} contains:", filtered_df.value_counts())

        return df_all_lc[mask_index]

    def nan_checker(self):
        """
        Function that checks if there are NaN values in the prediction
        """

        nan_index = np.argwhere(np.isnan(self.y_pred))
        mask = np.ones(self.y_pred.shape, dtype=bool)[:, 0]
        mask[nan_index] = False

        if not mask.all():
            print("There are NaN values in the prediction")

        self.nan_index = nan_index
        self.y_pred = self.y_pred[mask]
        
        if hasattr(self, 'y_true'):
            self.y_true = self.y_true[mask]

    def plot_confusion_matrix(self, normalize=False, prob_Ia=0):
        """
        Function that plots the Confusion Matrix for external data

        Input
        =====
        normalize: bool (optional)
            if True, the confusion matrix will be normalized. Defaulte is False
        """
        self.nan_checker()

        if not hasattr(self, 'y_true'):
            print("True labels are not available. Confusion matrix cannot be generated")
            return None

        classes = ['Ia', 'no Ia']

        fig, ax = plt.subplots(figsize=(5, 5))

        y_pred = self.y_pred.reshape(len(self.y_pred))

        cm = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred.round(),
                              labels=[1, 0])

        norm = cm.sum(axis=1)[:, np.newaxis]
        if normalize:
            cm = cm / norm

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0,
                       vmax=1 if normalize else None)
        fig.colorbar(im, ax=ax, shrink=0.8)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes, rotation=45)
        ax.set_yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            text = rf"${cm[i, j]:0.2f}$" if normalize else rf"{cm[i, j]}"
            ax.text(j, i, text,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set(xlabel='Predicted label', ylabel='True label')
        ax.set_title('Confusion matrix')
        return fig, ax


# Cosmology


def lc_fitter(data, zp=27.5, zpsys='AB', model='salt3', plot=True, mcmc=False,
              instrument='des', dust=True):
    """
    Function that fits a light curve using sncosmo library

    Input
    =====
    data: pd.DataFrame
        light curve data Frame

    model: str (optional, default='salt3')
        model used to fit the light curve

    plot: bool (optional, default=True)
        if it's True, the light curve is plotted

    mcmc: bool (optional, default=False)
        if it's True, the fit is done using MCMC

    instrument: str (optional, default='des')
        instrument used to obtain the data

    Output
    =====
    result: sncosmo.FitResult
        result of the fit

    fitted_model: sncosmo.Model
        fitted model
    """

    # Preprocessing data for snocosmo
    data = data.copy()

    zpsys = [zpsys] * data.shape[0]
    data['zpsys'] = zpsys
    data['ZEROPT'] = [zp] * data.shape[0]

    if 'FLT' in data.columns:
        data.drop(columns=['FLT'], inplace=True)

    data['BAND'] = data.BAND.apply(lambda band: instrument + band)
    data['flux'] = data.FLUXCAL
    data['fluxerr'] = data.FLUXCALERR

    data = Table.from_pandas(data)

    if dust:
        # dust model
        ebv = data['MWEBV'][0]
        dust = snc.CCM89Dust()
        dust.set(ebv=ebv)

        # model to fit
        model = snc.Model(source=model,
                          effects=[dust],
                          effect_names=['mw'],
                          effect_frames=['obs'])
        model.set(mwebv=ebv)

    else:
        # model to fit
        model = snc.Model(source=model)

    z = data['REDSHIFT_FINAL'][0]
    t0 = data['PEAKMJD'][0]
    model.set(z=z, t0=t0)

    func = snc.mcmc_lc if mcmc else snc.fit_lc
    result, fitted_model = func(data, model, ['x0', 'x1', 'c'],
                                modelcov=True)

    if plot:
        snc.plot_lc(data, model=fitted_model, errors=result.errors,
                    zp=zp)

    return result, fitted_model


def lc_fit_summary(data, band,
                   zpsys='AB', zp=27.5, model='salt3', mcmc=False,
                   instrument='des', dust=True):
    """
    Function that fits a light curve using sncosmo library and returns a
    summary of the fit for each observation in the data.

    Input
    =====
    data: pd.DataFrame
        light curve data Frame

    band: str or sncosmo bandpass
        band used to fit the light curve

    model: str (optional, default='salt3')
        model used to fit the light curve

    mcmc: bool (optional, default=False)
        if it's True, the fit is done using MCMC

    instrument: str (optional, default='des')
        instrument used to obtain the data

    Output
    =====
    data_summary: pd.DataFrame
        summary of the fit for each observation in the data

    obs_err: list
        list with the index of the observations that could not be fitted
    """

    # Predefine lists for results
    results = {'m_B': [], 'mabs_B': [], 'x0': [], 'x1': [], 'c': [],
               'cov_matrix': [], 'log_mass_host': [], 'log_mass_host_err': [],
               'G_host': [], 'z': [], 'z_err': [], 'vpec': [], 'vpec_err': []
               }
    obs_err = []

    mu_sim = 'SIM_DLMU' in data.columns
    mu_sim_values = []

    sntype = 'SNTYPE' in data.columns
    sntype_values = []

    # Predefine the Jacobian matrix function
    # https://github.com/sncosmo/sncosmo/issues/207#issuecomment-312023684
    jacobian_matrix = lambda x0: np.array([[-2.5 / (np.log(10) * x0), 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])

    # Iterate over unique indices in the data
    for i in data.index.unique():
        data_i = data.loc[i]

        try:
            result, fitted_model = lc_fitter(data_i, zp=zp, zpsys=zpsys,
                                             model=model, plot=False, mcmc=mcmc,
                                             instrument=instrument, dust=dust)
        except RuntimeError:
            try:
                result, fitted_model = lc_fitter(data_i, zp=zp, zpsys=zpsys,
                                                 model=model, plot=False,
                                                 mcmc=True,
                                                 instrument=instrument,
                                                 dust=dust)
            
            except Exception:
                obs_err.append(i)
                continue

        # Skip the current observation if the covariance matrix is not available
        if result.covariance is None:
            continue

        # Append simulation distance modulus if available   
        if mu_sim:
            mu_sim_values.append(data_i['SIM_DLMU'].iloc[0])
        
        # Append SNTYPE if available
        if sntype:
            sntype_values.append(data_i['SNTYPE'].iloc[0])

        # Extract fitted parameters
        results['m_B'].append(fitted_model.source_peakmag(band=band, magsys=zpsys))
        results['mabs_B'].append(fitted_model.source_peakabsmag(band, magsys=zpsys))
        results['x0'].append(fitted_model['x0'])
        results['x1'].append(fitted_model['x1'])
        results['c'].append(fitted_model['c'])

        # Compute covariance matrix
        j_matrix = jacobian_matrix(fitted_model['x0'])
        cov_matrix = j_matrix @ result.covariance @ j_matrix.T  # m_B, x1, c
        results['cov_matrix'].append(cov_matrix)

        # Host galaxy properties
        mass_i = data_i['HOSTGAL_LOGMASS'].iloc[0]
        results['log_mass_host'].append(mass_i)
        results['log_mass_host_err'].append(data_i['HOSTGAL_LOGMASS_ERR'].iloc[0])
        results['G_host'].append(0.5 if mass_i > 10 else -0.5)

        # Redshift and peculiar velocity
        results['z'].append(data_i['REDSHIFT_FINAL'].iloc[0])
        results['z_err'].append(data_i['REDSHIFT_FINAL_ERR'].iloc[0])
        results['vpec'].append(data_i['VPEC'].iloc[0])
        results['vpec_err'].append(data_i['VPEC_ERR'].iloc[0])

    # Create summary DataFrame
    data_summary = pd.DataFrame(results)

    # Add simulated distance modulus if available
    if mu_sim:
        data_summary['mu_sim'] = mu_sim_values
    
    # Add SNTYPE if available
    if sntype:
        data_summary['SNTYPE'] = sntype_values

    return data_summary, obs_err

@njit
def E(z, omega_m, omega_de, omega_r, w_0, w_a):
    """
    Function that calculates the normalized Hublle parameter as a function of
    redshift using the Friedmann equation

    Input
    =====
    z: float
        redshift

    omega_m: float
        matter density parameter

    omega_de: float
        dark energy density parameter

    omega_r: float
        radiation density parameter

    w_0, w_a: float
        dark energy equation of state parameters
        w(z) = w_0 + w_a * z / (1 + z)

    Output
    =====
    E(z): float
        normalized Hubble parameter
    """

    omega_k = 1 - omega_m - omega_de - omega_r

    term_w0 = (1 + w_0) * np.log(1 + z)
    term_wa = (1 / (1 + z) + np.log(1 + z) - 1) * w_a
    f_z = np.exp(3 * (term_w0 + term_wa))

    dark_energy = omega_de * f_z
    radiation = omega_r * (1 + z) ** 4
    matter = omega_m * (1 + z) ** 3
    curvature = omega_k * (1 + z) ** 2

    return np.sqrt(radiation + matter + dark_energy + curvature)


def Hubble_param(z, h, omega_m, omega_de, omega_r, w_0, w_a):
    """
    Function that calculates the Hubble parameter as a function of redshift
    using the Friedmann equation

    Input
    =====
    z: float
        redshift

    h: float
        Hubble constant in units of 100 km/s/Mpc

    omega_m: float
        matter density parameter

    omega_de: float
        dark energy density parameter

    omega_r: float
        radiation density parameter

    w_0, w_a: float
        dark energy equation of state parameters
        w(z) = w_0 + w_a * z / (1 + z)

    Output
    ======
    Hubble parameter at redshift z
    """

    H0 = h * 100  # km/s/Mpc
    return H0 * E(z, omega_m, omega_de, omega_r, w_0, w_a)

@njit
def S_k(omega_k, x):
    """
    Function that calculates the comoving distance as a function of the
    curvature parameter omega_k

    Input
    =====
    omega_k: float
        curvature parameter

    x: float
        comoving distance
    """
    if np.isclose(omega_k, 0):
        return x

    elif omega_k > 0:
        sqrt_omega_k = np.sqrt(omega_k)
        arg = sqrt_omega_k * x
        return np.sinh(arg) / sqrt_omega_k

    elif omega_k < 0:
        sqrt_omega_k = np.sqrt(-omega_k)
        arg = sqrt_omega_k * x
        return np.sin(arg) / sqrt_omega_k


def reduced_luminosity_distance(z, omega_m, omega_de, omega_r=0,
                                w_0=-1, w_a=0):
    """
    Function that calculates the c/H0 reduced luminosity distance as a function
    of redshift using the Friedmann equation

    Input
    =====
    z: float
        redshift

    omega_m: float
        matter density parameter

    omega_de: float
        dark energy density parameter

    omega_r: float (optional, default=0)
        radiation density parameter

    w_0, w_a: float (optional, default=-1, 0)
        dark energy equation of state parameters
        w(z) = w_0 + w_a * z / (1 + z)

    Output
    ======
    reduced luminosity distance at redshift z
    """
    args = (omega_m, omega_de, omega_r, w_0, w_a)

    def int_func(z):
        return integrate.quad(lambda z_i: 1 / E(z_i, *args), 0, z)[0]
    integral = np.vectorize(int_func)(z)

    omega_k = 1 - omega_m - omega_de - omega_r

    return (1 + z) * S_k(omega_k, integral)


def distance_modulus(z, omega_m, omega_de, omega_r=0, w_0=-1, w_a=0, h=None):
    """
    Function that calculates the distance modulus as a function of redshift
    using the Friedmann equation

    Input
    =====
    z: float
        redshift

    omega_m: float
        matter density parameter

    omega_de: float
        dark energy density parameter

    omega_r: float (optional, default=0)
        radiation density parameter

    w_0, w_a: float (optional, default=-1, 0)
        dark energy equation of state parameters
        w(z) = w_0 + w_a * z / (1 + z)

    h: float (optional, default=None)
        Hubble constant in units of 100 km/s/Mpc

        if None, the distance modulus is calculated using the c/H0 reduced
        luminous distance

    Output
    ======
    distance modulus at redshift z
    """
    args = (z, omega_m, omega_de, omega_r, w_0, w_a)

    lum_dist = reduced_luminosity_distance(*args)

    if h is not None:
        H0 = h * 100  # km/s/Mpc
        lum_dist *= c / H0

    return 5 * np.log10(lum_dist) + 25  # luminosity distance in Mpc

# MCMC

def plot_chains(sampler, labels):
    """
    Plot the MCMC chains
    """

    samples = sampler.get_chain()
    nsteps, nwalkers, ndim = np.shape(samples)

    tau = sampler.get_autocorr_time(tol=0)
    max_tau = np.nanmax(tau)
    discard = int(3 * max_tau)
    
    fig, ax = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 8), sharex=True)

    if ndim == 1:
        ax.plot(samples[:, :, 0], 'k', alpha=0.1, linewidth=0.5, ls='-')

        ax.set_ylabel(labels[0])
        ax.set_xlabel("step number")
        ax.grid(ls=':', alpha=0.3)

        ax.axvline(discard, color='red', ls='--',
                   label='discarded \n burn-in')

        ax.legend()
        
    else:
        for i, axis in enumerate(ax):
            axis.plot(samples[:, :, i], 'k', alpha=0.1, linewidth=0.5, ls='-')

            axis.set_ylabel(labels[i])
            axis.grid(ls=':', alpha=0.3)

            axis.axvline(discard, color='red', ls='--',
                         label='discarded \n burn-in')

        axis.set_xlabel("step number")
        ax[0].legend()
    return fig, ax

def summarize_mcmc(flat_samples, labels):
    """
    Summarize MCMC results, compute percentiles, and display formatted LaTeX.

    Inputs:
    =======

    flat_samples:
        Flattened MCMC chain samples of shape (n_samples, n_params)

    labels:
        List of LaTeX labels for each parameter

    Output:
    =======
    Array of computed mean and std for each parameter.
    """
    ndim = flat_samples.shape[1]
    # Compute 25th, 50th, 75th percentiles
    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0).T

    for i in range(ndim):
        difs = np.diff(mcmc[i])
        inf, sup = difs
        txt = (f"{labels[i]} = "
               f"${mcmc[i, 1]:.3f}_{{-{inf:.3f}}}^{{+{sup:.3f}}}")
        display(Math(txt))
    
    dict_output = {label[2:-1]: flat_samples[:, i].mean() for i, label in enumerate(labels)}
    dict_output.update({f"{label[2:-1]}_err": flat_samples[:, i].std()
                        for i, label in enumerate(labels)})

    return dict_output

def plot_corner(sampler, labels, discard=None, thin=None, truths=None, pretty=False):
    """
    Generate a corner plot for the posterior distribution
    """

    tau = sampler.get_autocorr_time(tol=0)
    max_tau = np.nanmax(tau)

    discard_tau = int(3 * max_tau) if not np.isnan(max_tau) else 0
    discard = discard if discard is not None else discard_tau

    thin_tau = int(max_tau / 2) if not np.isnan(max_tau) else 1
    thin = thin if thin is not None else thin_tau

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    figure = corner.corner(flat_samples, labels=labels,
                           show_titles=True, title_fmt="0.3f",
                           levels=(0.393, 0.864, 0.989),  # 1, 2, 3 sigma
                           title_quantiles=(0.16, 0.50, 0.84),
                           truths=truths, smooth=1)
    plt.show()

    if pretty:
        samples = MCSamples(samples=flat_samples, names=labels)
        samples.updateSettings({'contours': [0.393, 0.864, 0.989]})

        plot = plots.get_subplot_plotter()
        plot.settings.num_plot_contours = 3
        plot.triangle_plot(samples, filled=True, truth_vals=truths)
        plt.show()
    
    return summarize_mcmc(flat_samples, labels)
