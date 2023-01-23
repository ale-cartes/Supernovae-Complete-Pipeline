import os
from utils import *

path = os.getcwd()

folder = 'Supernovae-Complete-Pipeline'
path_back = path.split(sep=folder)[0]
curves_dir = path_back + 'curves'

Ia_DES_curves = reader(f"{curves_dir}\\Ia\\DES\\DES_Ia_PHOT.FITS")

nonIa_DES_curves = reader(f"{curves_dir}\\nonIa\\DES\\DES_nonIa_PHOT.FITS")
nonIa_summary = summary(f"{path}\\Lightcurves\\nonIa\\DES\\DES_nonIa.DUMP")

mjd_to_days(Ia_DES_curves, inplace=True, output=False)
mjd_to_days(nonIa_DES_curves, inplace=True, output=False)

nonIa_DES_curves['Type'] = ['nonIa'] * nonIa_DES_curves.shape[0]
Ia_DES_curves['Type'] = ['Ia'] * Ia_DES_curves.shape[0]

columns = ['obs', 'MJD', 'Days',
           'BAND', 'FLUXCAL', 'FLUXCALERR', 'Type']
curves = pd.concat((Ia_DES_curves[columns], 
                    nonIa_DES_curves[columns]))

del Ia_DES_curves, nonIa_DES_curves  # free-up memory