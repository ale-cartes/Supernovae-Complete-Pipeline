import os
from utils import *

path = os.getcwd()

folder = 'Supernovae-Complete-Pipeline'
path_back = os.path.split(path)[0]
curves_dir = os.path.join(path_back, 'curves')

# read curves data
Ia_DES_curves = reader(os.path.join(curves_dir, 'Ia', 'DES',
                                    'DES_Ia_PHOT.FITS'))

nonIa_DES_curves = reader(os.path.join(curves_dir, 'nonIa', 'DES',
                                       'DES_nonIa_PHOT.FITS'))

# .DUMP files
Ia_dump = os.path.join(os.path.join(path, 'Lightcurves', 'Ia',
                                    'DES', 'DES_Ia.DUMP'))
nonIa_dump = os.path.join(os.path.join(path, 'Lightcurves', 'nonIa',
                                       'DES', 'DES_nonIa.DUMP'))

# nonIa types
nonIa_summary = summary(nonIa_dump)

# change MJD to days
peakmjd_to_days(Ia_DES_curves, Ia_dump, inplace=True, output=False)
peakmjd_to_days(nonIa_DES_curves, nonIa_dump, inplace=True, output=False)

# add nonIa as new observations
nonIa_DES_curves.obs += max(Ia_DES_curves.obs)

# one-hot encoder: 1 -> Ia, 0 -> nonIa
nonIa_DES_curves['Type'] = [0] * nonIa_DES_curves.shape[0]
Ia_DES_curves['Type'] = [1] * Ia_DES_curves.shape[0]

columns = ['obs', 'MJD', 'Days', 'BAND', 'FLUXCAL', 'FLUXCALERR', 'Type']
curves = pd.concat((Ia_DES_curves[columns], nonIa_DES_curves[columns]))

del Ia_DES_curves, nonIa_DES_curves  # free-up memory
