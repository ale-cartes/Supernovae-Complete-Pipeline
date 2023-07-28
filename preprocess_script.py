import os
from utils import *

# requesting for data augmentation and save
normalize = (input("Do data normalization?[y/n] ") == 'y')
aug_input = (input("Do data augmentation? [y/n] ") == 'y')
save_input = (input("save data? [y/n] ") == 'y')

# searching for files and folders
path = os.getcwd()
curves_dir = "Lightcurves/curves"

Ia_DES_file = os.path.join(curves_dir, 'Ia', 'DES', 'DES_Ia_PHOT.FITS')
nonIa_DES_file = os.path.join(curves_dir, 'nonIa', 'DES', 'DES_nonIa_PHOT.FITS')

Ia_head = os.path.join(curves_dir, 'Ia', 'DES', 'DES_Ia_HEAD.FITS')
nonIa_head = os.path.join(curves_dir, 'nonIa', 'DES', 'DES_nonIa_HEAD.FITS')
nonIa_summary = summary(nonIa_head)

print('nonIa types:', nonIa_summary['SNTYPE'].value_counts(), sep='\n')

# preprocess the data
if aug_input:
    Ia_fitted = curves_augmentation(preprocess(Ia_DES_file, head_file=Ia_head,
                                               normalize=normalize))
    nonIa_fitted = curves_augmentation(preprocess(nonIa_DES_file,
                                                  head_file=nonIa_head,
                                                  normalize=normalize))

else:
    Ia_fitted = preprocess(Ia_DES_file, head_file=Ia_head, normalize=normalize)
    nonIa_fitted = preprocess(nonIa_DES_file, head_file=nonIa_head,
                              normalize=normalize)

curves_fitted = pd.concat((Ia_fitted, nonIa_fitted), ignore_index=True)
curves_fitted = replace_nan_array(curves_fitted)

# one-hot encoder: 1 -> Ia, 0 -> nonIa
types = [1 if j < len(Ia_fitted) else 0 for j in range(len(curves_fitted))]
curves_fitted['Type'] = types

# give Neural Network format to the data
curves_RNN, types_RNN = RNN_reshape(curves_fitted)

# save the data
if save_input:
    file_name = './data_folder/curves_RNN'
    file_types = './data_folder/types_RNN'

    if normalize:
        file_name += '_norm'
        file_types += '_norm'

    if aug_input:
        file_name += '_aug'
        file_types += '_aug'
    
    file_name += '.npy'
    file_types += '.npy'

    np.save(file_name, curves_RNN)
    np.save(file_types, types_RNN)
