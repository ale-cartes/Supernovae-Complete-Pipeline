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

# merging Ia-nonIa data
columns = ['obs', 'MJD', 'Days', 'BAND', 'FLUXCAL', 'FLUXCALERR']
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
    t_ev = np.linspace(curve.Days.min(), curve.Days.max(), 100)
    dict_curve_fitted = {"Days": t_ev}

    for band in bands:
        flux_fitted = fitter_Bspline(curve, band, t_ev, order=min_obs)
        dict_curve_fitted[band] = flux_fitted
    
    dict_curves_fitted[obs] = dict_curve_fitted

curves_fitted = pd.DataFrame(dict_curves_fitted).transpose()

# one-hot encoder: 1 -> Ia, 0 -> nonIa
Type = []
for obs in curves_fitted.index:
    if obs <= 100_000:
        Type.append(1)
    else:
        Type.append(0)

curves_fitted['Type'] = Type
