import os
from utils import *

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

Ia_fitted = curves_augmentation(preprocess(Ia_DES_file, Ia_dump))
nonIa_fitted = curves_augmentation(preprocess(nonIa_DES_file, nonIa_dump))

curves_fitted = pd.concat((Ia_fitted, nonIa_fitted), ignore_index=True)
curves_fitted = replace_nan_array(curves_fitted)

# one-hot encoder: 1 -> Ia, 0 -> nonIa
Type = [1 if j < len(Ia_fitted) else 0 for j in range(len(curves_fitted))]
curves_fitted['Type'] = Type

file_fitted = './curves_fitted.pkl'
curves_fitted.to_pickle('./curves_fitted.pkl')