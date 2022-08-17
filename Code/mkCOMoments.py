#make moment maps of 22" CO TP cube

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import pandas as pd
from makeMomentsfromSpectra import make_moments_from_spectra

fname_co = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed'
#fname_co = '../Data/Ancillary_Data/M82.CO.30m.map'


co_cube = SpectralCube.read(fname_co+'.fits').with_spectral_unit(u.km/u.s)
co_cube.allow_huge_operations = True

co_mom0 = co_cube.moment(order=0)
co_mom0.write(fname_co+'_mom0.fits',overwrite=True)

co_mom1 = co_cube.moment(order=1)
co_mom1.write(fname_co+'_mom1.fits',overwrite=True)

co_peak = co_cube.max(axis=0)
co_peak.write(fname_co+'_peak.fits',overwrite=True)


#now get the moments of the outflow pointings
n_pointings = 2
n_pixels = 7

vmin=-100.
vmax=400.

for i in range(n_pointings):	

	for j in range(n_pixels):
		co_dat = pd.read_csv('../Data/Outflow_Pointings/CO_TP/outflow'+str(i+1)+'_pix'+str(j)+'_smo.csv')
		this_vel = co_dat['Velocity_kms'].values
		this_spec = co_dat['Intensity_K'].values
		this_vel_og = this_vel.copy()
		this_spec_og = this_spec.copy()
		idx = np.where((this_vel >= vmin) & (this_vel <= vmax))[0]
		this_spec = this_spec[idx]
		this_vel = this_vel[idx]

		dv = np.abs(this_vel[0]-this_vel[1]) #km/s
		tab = make_moments_from_spectra(this_vel,this_spec,dv,nsig=2.,vel_all=this_vel_og,spec_all=this_spec_og,vmin=vmin,vmax=vmax)
		this_df = pd.DataFrame.from_dict(tab)
		this_df.to_csv('../Data/Outflow_Pointings/CO_TP_Moments/outflow'+str(i+1)+'_pix'+str(j)+'_momentparams.csv',index=False)


