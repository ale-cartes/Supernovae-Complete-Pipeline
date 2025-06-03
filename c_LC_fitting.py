"""
Module for fitting supernova light curves using classified data.

This script handles:
- Loading preprocessed SN data and classifier.
- Making predictions using the trained classifier.
- Selecting SN Ia candidates based on classification probabilities.
- Fitting the light curves of classified SN Ia.
- Saving and loading fitted light curve parameters.

Author: Alejandro Cartes
"""

from utils import *
from b_classifier import *

def load_nn_classifier(name):
    """
    Load the preprocessed SN data and initialize the trained classifier
    """

    nn_class = pretraining_class(name)
    best_params, batch_size = hyperparameter_tunning(nn_class, load=True)
    
    return nn_class, best_params, batch_size

def classify_external_data(classes, nn_class, prob_Ia=0.9, weight_file_index=0,
                           plot_confusion_matrix=False):
    """
    Load external SN data, apply classification, and filter SN Ia candidates
    """

    external_data = ExternalData(sn_class=classes, nn_class=nn_class)
    external_data.load_weights_model(i=weight_file_index)
    external_data.model_prediction()

    if plot_confusion_matrix:
        external_data.plot_confusion_matrix(normalize=True)
        external_data.plot_confusion_matrix(normalize=False)
    
    # Extract SN Ia candidates
    sn_df = external_data.df_lc_class_Ia(prob_Ia=prob_Ia)
    
    return sn_df

def fit_lightcurves(sn_df, band, name_file="lc_params.pkl", load=False):
    """
    Fit the light curves of classified SN Ia
    """

    if load:
        data = pd.read_pickle(f"data_folder/fitted_curves/{name_file}")

    else: 
        data, index_discarded = lc_fit_summary(sn_df, band)
        data.to_pickle(f"data_folder/fitted_curves/{name_file}")
    
    return data

def load_fitted_data(name_file="lc_params.pkl"):
    """
    Load the previously fitted light curve data
    """
    
    return pd.read_pickle(f"data_folder/fitted_curves/{name_file}")

if __name__ == "__main__":
    name = 'z_host'
    
    # Step 1: Load the classifier and best parameters
    nn_class, best_params, batch_size = load_nn_classifier(name)

    # Step 2: Classify external data and extract SN Ia candidates
    classes = load_classes_from_file(name)
    sn_df = classify_external_data(classes, nn_class, prob_Ia=0.9,
                                   plot_confusion_matrix=True)

    # Step 3: Fit light curves
    band = snc.get_bandpass('bessellb')
    data = fit_lightcurves(sn_df, band, load=True,
                           name_file="classified_Ia_0.9.pkl")   
