import os
from utils import *

if 'path' not in globals():
    path = os.getcwd()


folder = 'Supernovae-Complete-Pipeline'
path_back = path.split(sep=folder)[0]
curves_dir = path_back + 'curves'

Ia_DES_curves = reader(f"{curves_dir}\\Ia\\DES\\DES_Ia_PHOT.FITS")

nonIa_DES_curves = reader(f"{curves_dir}\\nonIa\\DES\\DES_nonIa_PHOT.FITS")
nonIa_summary = summary(f"{path}\\Lightcurves\\nonIa\\DES\\DES_nonIa.DUMP")

os.chdir(path)

lightcurve_columns = ['obs', 'MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR']

mjd_to_days(Ia_DES_curves, inplace=True, output=False)
mjd_to_days(nonIa_DES_curves, inplace=True, output=False)
