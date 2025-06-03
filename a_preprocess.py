"""
Module for loading and preprocessing lightcurve data from FITS files.

This script handles:
- Loading FITS files for Ia and non-Ia supernovae.
- Preprocessing the loaded lightcurves.
- Saving and loading preprocessed data.

Author: Alejandro Cartes
"""

from utils import *

def generate_combinations(bands):
    """Generate all combinations of bands"""

    combinations = []
    for r in range(1, len(bands) + 1):
        for subset in itertools.combinations(bands, r):
            combinations.append(list(subset))
    return combinations

def create_SNclasses(bands, folder_Ia, folder_noIa):
    """Create SN classes (Ia and non-Ia) from FITS files."""

    combinations = generate_combinations(bands)
    
    Ia_classes = []
    noIa_classes = []
    
    for comb in combinations:
        # file paths for Ia classes
        file_Ia = folder_Ia + ("/DES_Ia_" + ''.join(comb)) * 2
        head_Ia = file_Ia + "_HEAD.FITS"
        phot_Ia = file_Ia + "_PHOT.FITS"
        Ia_classes.append(SN_data(head_file=head_Ia, phot_file=phot_Ia))

        # file paths for non-Ia classes
        file_noIa = folder_noIa + ("/DES_noIa_" + ''.join(comb)) * 2
        head_noIa = file_noIa + "_HEAD.FITS"
        phot_noIa = file_noIa + "_PHOT.FITS"
        noIa_classes.append(SN_data(head_file=head_noIa, phot_file=phot_noIa))

    # Combine Ia and non-Ia classes
    classes = np.append(Ia_classes, noIa_classes)
    return classes

def preprocess_classes(classes, z_host=True, band_col='BAND',
                       normalize=True, len_seq=50):
    """Preprocess the lightcurve data for each class"""
    for i, sn_class in enumerate(classes):
        sn_class.reader(band_col=band_col)
        sn_class.preprocess(z_host=z_host, normalize=normalize, len_seq=len_seq)
    return classes

def save_classes(classes, name):
    """Save the preprocessed classes to a file"""
    np.save(f"data_folder/classes/{name}.npy", classes)

def load_classes_from_file(name):
    """Load the preprocessed classes from a file"""
    return np.load(f"data_folder/classes/{name}.npy", allow_pickle=True)

if __name__ == "__main__":
    bands = ['g', 'r', 'i', 'z']
    folder_Ia = "Lightcurves/curves/Ia/DES/SNANA_aug"
    folder_noIa = "Lightcurves/curves/nonIa/DES/SNANA_aug"
    name = 'z_host'

    # Step 1: Load classes
    classes = create_SNclasses(bands, folder_Ia, folder_noIa)
    
    # Step 2: Preprocess the loaded classes
    classes = preprocess_classes(classes)

    # Step 3: Save the preprocessed classes
    save_classes(classes, name)
