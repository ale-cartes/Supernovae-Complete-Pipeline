"""
Module for training a neural network classifier on preprocessed supernova data.

This script handles:
- Loading preprocessed SN data.
- Initializing the NN_classifier.
- Splitting data into training, validation, and test sets.
- Performing hyperparameter tuning with PyHopper.
- Training and evaluating the final model.

Author: Alejandro Cartes
"""

from utils import *
from a_preprocess import *

def pretraining_class(file_name, frac=1, seed=42, train_size=0.7, val_size=0.15):
    """Load preprocessed SN data and initialize the NN_classifier class"""

    print(f"Loading SN classes called {file_name}")

    classes = load_classes_from_file(file_name)

    # Initialize the NN_classifier class
    nn_classifier = NN_classifier(classes, name=file_name)
    nn_classifier.data_sample(frac, seed=seed)
    nn_classifier.train_test_split(train_size=train_size, val_size=val_size)

    # free memory
    del classes

    # Reshape the data for the neural network
    nn_classifier.NN_reshape()

    # Print the number of samples for each class
    print(nn_classifier.data.sn_type.value_counts())

    return nn_classifier

def hyperparameter_tunning(nn_class, search_nl=None, epochs=150, load=False,
                           time='72h', steps=None, nwrap=4):
    """Perform hyperparameter tuning using PyHopper."""

    best_params, batch_size = nn_class.best_hyp_pyhopper(nn_class.model_nl_creator,
                                                         search_params=search_nl,
                                                         verbose=1,
                                                         time=time,
                                                         steps=steps,
                                                         nwrap=nwrap,
                                                         epochs=epochs,
                                                         save=True, load=load)

    nn_class.model_nl_creator(**best_params, plot_model=True)

    return best_params, batch_size

def training_statistics(nn_class, num_it=25, epochs=150, batch_size=512, patience=15, load=False):
    """Train and evaluate multiple times the model"""

    nn_class.model_statistics(num_it=num_it, epochs=epochs, batch_size=batch_size,
                              patience=patience, load=load)
    return nn_class

def evaluation_plots(nn_class):
    """Generate evaluation plots for model performance."""

    nn_class.training_loss_plot()
    nn_class.plot_roc_curve()
    nn_class.plot_confusion_matrix(normalize=False)
    nn_class.plot_confusion_matrix(normalize=True)

    return None

if __name__ == "__main__":
    # Step 1: Initialize the NN_classifier class
    frac, seed = 1, 42
    name = 'z_host'
    nn_class_mb = pretraining_class(file_name=name, frac=frac, seed=seed)

    # Step 2: Hyperparameter tuning

    n_max = 5  # max number of layers to be considered
    search_nl = pyhopper.Search(n = pyhopper.int(1, n_max),
                                rnns_i = pyhopper.int(0, 2, shape=n_max),
                                neurons = pyhopper.int(2, 128, power_of=2, shape=n_max),
                                activations_i = pyhopper.int(0, 3, shape=n_max),
                                init_weights_i = pyhopper.int(0, 2, shape=n_max),
                                dropout = pyhopper.float(0, 0.4, precision=2),
                                lr = pyhopper.float(1e-5, 1e-3, log=True, precision=1),
                                batch_size = pyhopper.int(512*frac, 8192*frac, power_of=2)
                                )
    
    epochs = 150
    best_params, batch_size = hyperparameter_tunning(nn_class_mb, search_nl=search_nl,
                                                     epochs=epochs, load=True)
    
    # Step 3: Training and evaluation
    training_statistics(nn_class_mb, epochs=epochs, batch_size=batch_size, load=True)

    # Step 4: Evaluation plots
    evaluation_plots(nn_class_mb)
