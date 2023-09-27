import os
from utils import *

# requesting for data augmentation and save
normalize = (input("Do data normalization?[y/n] ") == 'y')
save_input = (input("save data? [y/n] ") == 'y')

# searching for files and folders
path = os.getcwd()
curves_dir = "Lightcurves/curves"

Ia_DES_file = os.path.join(curves_dir, 'Ia', 'DES', 'DES_Ia_PHOT.FITS')
nonIa_DES_file = os.path.join(
    curves_dir, 'nonIa', 'DES', 'DES_nonIa_PHOT.FITS')

Ia_head = os.path.join(curves_dir, 'Ia', 'DES', 'DES_Ia_HEAD.FITS')
nonIa_head = os.path.join(curves_dir, 'nonIa', 'DES', 'DES_nonIa_HEAD.FITS')

Ia_summary = summary(Ia_head)
nonIa_summary = summary(nonIa_head)

print('n° Ia curves:', summary(Ia_head)['SNTYPE'].size)
print('nonIa types:', nonIa_summary['SNTYPE'].value_counts(), sep='\n')

# preprocessing
w_power = 2  # eror weight power
Ia_preproccesed = preprocess(Ia_DES_file, head_file=Ia_head,
                             normalize=normalize, w_power=w_power)
nonIa_preproccesed = preprocess(nonIa_DES_file, head_file=nonIa_head,
                                normalize=normalize, w_power=w_power)

curves_preproccesed = pd.concat((Ia_preproccesed, nonIa_preproccesed),
                                ignore_index=True)

# one-hot encoder: 1 -> Ia, 0 -> nonIa
types = [1 if j < len(Ia_preproccesed) else 0 
         for j in range(len(curves_preproccesed))]

curves_preproccesed['Type'] = types

# save the data
if save_input:
    file_name = './data_folder/curves_preprocessed'

    if normalize:
        file_name += '_norm'

    file_name += '.parquet'

    curves_preproccesed.to_parquet(file_name)