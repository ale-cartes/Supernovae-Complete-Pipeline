import os
from utils import *
from scipy.interpolate import splrep, splev

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

# merging Ia-nonIa data
columns = ['obs', 'MJD', 'Days', 'BAND', 'FLUXCAL', 'FLUXCALERR', 'Type']
curves_nonfiltered = pd.concat((Ia_DES_curves[columns],
                                nonIa_DES_curves[columns]))


# discard light curves with few observations in a band
min_obs = 5
curves_nonfilt_group = curves_nonfiltered.groupby('obs')
curves_band_counts = curves_nonfilt_group.BAND.value_counts()

obs_discard = []
for obs in curves_nonfiltered.obs.unique():
    if not (curves_band_counts[obs] > min_obs).all():
        obs_discard.append(obs)

curves = curves_nonfiltered[~curves_nonfiltered.obs.isin(obs_discard)]

# fitting curves using B-splines
bands = ['g ', 'r ', 'i ', 'z ']
curves_group = curves.groupby('obs')
dict_curves_fitted = {}

for obs, curve in curves_group:
    x_new = np.linspace(curve.Days.min(), curve.Days.max(), 100)
    dict_curve_fitted = {"Days": x_new}

    for band in bands:
        x = curve[curve.BAND == band].Days
        y = curve[curve.BAND == band].FLUXCAL
        yerr = curve[curve.BAND == band].FLUXCALERR

        spl = splrep(x, y, w=1/yerr, k=min_obs)

        y_new = splev(x_new, spl)

        dict_curve_fitted[band] = y_new
    
    dict_curves_fitted[obs] = dict_curve_fitted

curves_fitted = pd.DataFrame(dict_curves_fitted).transpose()


# del Ia_DES_curves, nonIa_DES_curves  # free-up memory
