import os
from utils import *

aug_input = (input("Do data augmentation? [y/n]") == 'y')
save_input = (input("save data? [y/n]") == 'y')

path = os.getcwd()

folder = 'Supernovae-Complete-Pipeline'
path_back = os.path.split(path)[0]
curves_dir = os.path.join(path_back, 'curves')

Ia_DES_file = os.path.join(curves_dir, 'Ia', 'DES', 'DES_Ia_PHOT.FITS')
nonIa_DES_file = os.path.join(curves_dir, 'nonIa', 'DES', 'DES_nonIa_PHOT.FITS')

Ia_dump = os.path.join(path, 'Lightcurves', 'Ia', 'DES', 'DES_Ia.DUMP')
nonIa_dump = os.path.join(path, 'Lightcurves', 'nonIa', 'DES', 'DES_nonIa.DUMP')
nonIa_summary = summary(nonIa_dump)

print('nonIa types:', nonIa_summary.value_counts(), sep='\n')

if aug_input:
    Ia_fitted = curves_augmentation(preprocess(Ia_DES_file, Ia_dump))
    nonIa_fitted = curves_augmentation(preprocess(nonIa_DES_file, nonIa_dump))

else:
    Ia_fitted = preprocess(Ia_DES_file, Ia_dump)
    nonIa_fitted = preprocess(nonIa_DES_file, nonIa_dump)

curves_fitted = pd.concat((Ia_fitted, nonIa_fitted), ignore_index=True)
curves_fitted = replace_nan_array(curves_fitted)

# one-hot encoder: 1 -> Ia, 0 -> nonIa
Type = [1 if j < len(Ia_fitted) else 0 for j in range(len(curves_fitted))]
curves_fitted['Type'] = Type

curves_RNN, types_RNN = RNN_reshape(curves_fitted)

if save_input:
    file_RNN = 'curves_RNN_fitted.npy'
    file_types_RNN = 'types_RNN.npy'

    if aug_input:
        file_RNN = 'curves_RNN_aug.npy'
        file_types_RNN = 'types_RNN_aug.npy'

    np.save(file_RNN, curves_RNN)
    np.save(file_types_RNN, types_RNN)