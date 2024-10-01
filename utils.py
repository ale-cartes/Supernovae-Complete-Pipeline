import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import splrep, splev
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import itertools

import keras
from keras import optimizers, initializers
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, GRU, LSTM,\
                         Dense, Bidirectional, Dropout,\
                         Flatten, BatchNormalization,\
                         Input, concatenate, Add
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy
import pickle
import pyhopper
import multiprocessing as mp

import sncosmo as snc

seed = 42
keras.utils.set_random_seed(seed)

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

def lc_fitter(data, model='salt3', plot=True, mcmc=True):
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
    
    mcmc: bool (optional, default=True)
        if it's True, the MCMC method is used to fit the light curve
    
    Output
    =====
    
    result: sncosmo.FitResult
        result of the fit
    
    fitted_model: sncosmo.Model
        fitted model
    """
    
    data = data.copy()
    zpsys = ['ab'] * data.shape[0]
    data['zpsys'] = zpsys
    
    data['BAND'] =  data.BAND.apply(lambda x: 'des' + x)
    
    data['flux'] = data.FLUXCAL
    data['fluxerr'] = data.FLUXCALERR
    
    data = Table.from_pandas(data)
    
    model = snc.Model(source=model)
    
    z = data['REDSHIFT_FINAL'][0]
    t0 = data['PEAKMJD'][0]
    
    model.set(z=z, t0=t0)
    
    func = snc.mcmc_lc if mcmc else snc.fit_lc
        
    result, fitted_model = func(data, model, ['x0', 'x1', 'c'])
    
    if plot:
        snc.plot_lc(data, model=fitted_model, errors=result.errors)
    
    return result, fitted_model

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

        light_curves['BAND'] = light_curves[band_col].str.decode('utf-8').str.strip()
        light_curves.name = self.phot_file.split('/')[-1]

        self.lc_df = light_curves
        self.bands = light_curves.BAND.value_counts().keys()

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
        the moment of the peak. Head file with peak information should be given as
        attribute
        """
        if np.equal(self.head_file, None):
            print("This function only works if a Head file is provided")
            return None

        try: self.obs_info
        except: self.obs_summary()

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

        curves_fitted = pd.DataFrame.from_dict(dict_curves_fitted, orient='index')
        self.lc_fitted = curves_fitted
        self.obs_discarded = obs_discarded
        self.len_seq = len_seq
    
    def plotter(self, obs, days=True, fitted=False):
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
            try: self.lc_df.days
            except:
                if not np.equal(self.head_file, None):
                    self.peakmjd_to_days()

                else:
                    self.mjd_to_days()
        
        color = {'u': 'purple', 'g': 'green', 'r': 'red', 
                 'i': (150/255, 0, 0), 'z': (60/255, 0, 0)}

        if fitted:
            try: self.lc_fitted
            except: self.preprocess()

            data_obs = self.lc_fitted[self.lc_fitted.index == obs]
            if data_obs.empty:
                if obs in self.obs_discarded:
                    print(f"Obs: {obs} was discarded because was not possible to fit it")

                else:
                    print(f"Obs: {obs} was not found, try with another one")

                return None
            
            fig, ax = plt.subplots(figsize=(14, 8))
            data_obs = self.lc_fitted[self.lc_fitted.index == obs]
            
            for band in self.bands:
                ax.plot(*data_obs.days.values, *data_obs[band].values,
                        color=color[band], label=band)

            ax.set_xlabel('Days', fontsize=18)

        else:
            data_obs = self.lc_df[self.lc_df.index == obs]
            
            if data_obs.empty:
                print(f"Obs: {obs} was not found, try with another one")
                return None
            
            fig, ax = plt.subplots(figsize=(14, 8))
            for band in (data_obs['BAND'].value_counts()).index:
                data_to_plot = data_obs[data_obs['BAND'] == band]

                xdata_plot = data_to_plot.MJD
                xlabel = 'MJD'
                if days:
                    xlabel = 'Days'
                    xdata_plot = data_to_plot.days

                ax.errorbar(xdata_plot, data_to_plot.FLUXCAL,
                            yerr=data_to_plot.FLUXCALERR, marker='o',
                            ls='--', capsize=2, color=color[band], label=band)

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
    
    model_nl_creator_ph(n=None, rnns_i=[1, 1, 1], neurons=[8, 8, 8],
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
        self.data = self.data.sample(frac=frac, random_state=seed)
    
    def train_test_split(self, train_size=0.7, val_size=0.15, rand_state=42):
        data_wo_types = self.data.drop(columns=['sn_type', 'obs'])

        if data_wo_types.isna().any().any():
            len_seq = self.data.days[0].shape[0]
            array = np.zeros(len_seq)
            data_wo_types = data_wo_types.map(lambda x: array
                                              if np.array(pd.isnull(x)).any()
                                              else x)

        X_train, X_test, y_train, y_test = train_test_split(data_wo_types,
                                                            self.data.sn_type,
                                                            train_size=train_size,
                                                            random_state=rand_state)

        test_size = 1 - train_size - val_size
        test_size = test_size / (test_size + val_size)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                        test_size=test_size,
                                                        random_state=rand_state)

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
            self.X_train_nn, self.y_train_nn = func(self.X_train), func(self.y_train)
            self.X_val_nn, self.y_val_nn = func(self.X_val), func(self.y_val)
            self.X_test_nn, self.y_test_nn = func(self.X_test), func(self.y_test)

        else:
            return func(data_ext)

    def model_nl_creator_ph(self, n=None, rnns_i=[1, 1, 1], neurons=[8, 8, 8],
                            activations_i=[1, 1, 1],
                            init_weights_i=[0, 0, 0],
                            dropout=0.2,
                            optimizer=optimizers.Adam, lr=1e-3,
                            plot_model=False):

        n = len(rnns_i) if n==None else n
        
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
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        
        hist = self.model.fit(self.X_train_nn, self.y_train_nn,
                              validation_data=(self.X_val_nn, self.y_val_nn),
                              epochs=epochs, batch_size=batch_size,
                              callbacks=[early_stopping], verbose=verbose)
        
        self.fit_hist.append(hist)

        if plot:
            self.training_loss_plot()

    def best_hyp_pyhopper(self, search_params, model_creator,
                          time=None, steps=None, patience=15,
                          plot_loss=False, plot_bf=True, verbose=0,
                          epochs=250, nwrap=5, pruner=0.75,
                          n_jobs=1, save=False, load=False):
        
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
        cktp_file = folder if (save or load) else None

        if load:
            time = None
            steps = 0

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

            x = np.array(search_params.history.steps) + 1
            ax.scatter(x=x, y=search_params.history.fs,
                       label="Sampled")

            ax.plot(x, search_params.history.best_fs,
                    ls = '--', color="red",
                    label="Best so far", zorder=0)

            ax.grid(ls=':', alpha=0.4, zorder=0)
            ax.set(xlim=[0.5, len(search_params.history) + 0.5],
                   xlabel='Step',
                   ylabel='Validation Accuracy')

            ax.legend()
            
            folder = f"data_folder/images/pyhopper_opt_classifier_{self.name}.svg"
            fig.savefig(folder, transparent=True, bbox_inches='tight')

        return best_params, batch_size

    def model_statistics(self, num_it=10, batch_size=8, epochs=250,
                         verbose_fit=0, patience=15, load=False):
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

    def training_loss_plot(self):
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

            ax[index].plot(x, y, marker='.', ls='-', label=f"Mean\n{key[0:10]}")
            
            ax[index].fill_between(x, y-yerr, y+yerr, alpha=0.25,
                                   label=rf"$\pm 1 \sigma$")

            if i % 2 == 1:
                ax[index].set_xlabel('Epochs')

                if i<2:
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
                               label=f'$\pm 1 \sigma$',
                               zorder=0)

            ax[i].set(xlabel='False Positive rate', title=data_label[i])
            ax[i].grid(ls=':', alpha=0.3)

            mean_auc = np.mean(aucs, axis=0)
            std_auc = np.std(aucs, axis=0, ddof=0)

            if i==2:
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
            fig.colorbar(im, ax=ax[i_l], shrink=0.75)
            
            tick_marks = np.arange(len(classes))
            ax[i_l].set_xticks(tick_marks, classes, rotation=45)
            ax[i_l].set_yticks(tick_marks, classes)

            thresh = cm_mean.max() / 2.
            for i, j in itertools.product(range(cm_mean.shape[0]),
                                          range(cm_mean.shape[1])):
                if normalize:
                    text =  rf"${cm_mean[i, j]:0.2f} \pm {cm_std[i, j]:0.2f}$"
                else:
                    text = rf"${int(cm_mean[i, j])} \pm {int(cm_std[i, j])}$"
                ax[i_l].text(j, i, text,
                             horizontalalignment="center",
                             color="white" if cm_mean[i, j] > thresh else "black")

            ax[i_l].set(xlabel='Predicted label', title=label)

        ax[0].set_ylabel('True label')
        fig.suptitle('Confusion matrix')
        fig.subplots_adjust(wspace=0.05)

        folder = f"data_folder/images/confusion_matrix_classifier_{self.name}.svg"
        fig.savefig(folder, transparent=True, bbox_inches='tight')

        return fig, ax


class ExternalData:
    """
    Class for handling external data for the Neural Network classifier. This class
    provides methods to preprocess, predict, and plot the classifier performance.
    
    Attributes
    ==========
    
    sn_class: list
        list of SN_data classes
    
    nn_class: NN_classifier
        Neural Network classifier
    
    name: str
        name of the classifier
    
    y_pred: np.array
        numpy array with the predictions
    
    y_true: np.array
        numpy array with the true labels
    
    nan_index: np.array
        numpy array with the index of NaN values
    """
    def __init__(self, sn_class, nn_class, name=None):
        self.sn_class = sn_class
        self.nn_class = nn_class
        self.name = name

    def model_prediction(self):
        data = pd.concat([sn_class.lc_fitted
                          for sn_class in self.sn_class])
        data = pd.concat([data, pd.DataFrame(columns=['g', 'r', 'i', 'z'])],
                         ignore_index=True)
        
        data_wo_types = data.drop(columns=['sn_type'])
        
        if data_wo_types.isna().any().any():
            len_seq = data.days[0].shape[0]
            array = np.zeros(len_seq)
            data_wo_types = data_wo_types.map(lambda x: array
                                              if np.array(pd.isnull(x)).any()
                                              else x)

        data_reshaped = self.nn_class.NN_reshape(data_ext=data_wo_types)

        y_pred = self.nn_class.model.predict(data_reshaped)
        self.y_pred = y_pred

        self.y_true = data['sn_type']

    def nan_checker(self):
        nan_index = np.argwhere(np.isnan(self.y_pred))
        mask = np.ones(self.y_pred.shape, dtype=bool)[:, 0]
        mask[nan_index] = False

        if not mask.all():
            print("There are NaN values in the prediction")

        self.nan_index = nan_index
        self.y_pred = self.y_pred[mask]
        self.y_true = self.y_true[mask]

    def plot_confusion_matrix(self, normalize=False):
        """
        Function that plots the Confusion Matrix for external data

        Input
        =====
        normalize: bool (optional)
            if True, the confusion matrix will be normalized. Defaulte is False
        """
        self.nan_checker()

        classes = ['Ia', 'no Ia']

        fig, ax = plt.subplots(figsize=(5, 5))

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
        ax.set_title('Confusion matrix' if self.name == None
                     else f'Confusion matrix - {self.name}')
        return fig, ax

