#make moment maps of 17" HI cube

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import pandas as pd
from makeMomentsfromSpectra import make_moments_from_spectra


fname_hi = '../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor'

hi_cube = SpectralCube.read(fname_hi+'.fits').with_spectral_unit(u.km/u.s)
hi_cube.allow_huge_operations = True

hi_beam_area = np.pi*hi_cube.header['BMAJ']*hi_cube.header['BMIN']*u.deg**2/(4*np.log(2))
equiv = u.brightness_temperature(hi_cube.header['RESTFRQ']*u.Hz,beam_area=hi_beam_area)

hi_cube_K = hi_cube.to(u.K, equivalencies=equiv)
hi_cube_K.write(fname_hi+'_K.fits',overwrite=True)

#mask out the regions with absorption
#from Martini+2018 Sect 2.3
hi_mask_lev = 7E19/1.823E18/10*u.K #K 
print('Masking out I_HI < %.2f K' %hi_mask_lev.value)
mask = (hi_cube_K > hi_mask_lev)
hi_cube_masked = hi_cube_K.with_mask(mask)

hi_mom0 = hi_cube_masked.moment(order=0)
hi_mom0.write(fname_hi+'_mom0.fits',overwrite=True)

hi_mom1 = hi_cube_masked.moment(order=1)
hi_mom1.write(fname_hi+'_mom1.fits',overwrite=True)

#now get the moments of the outflow pointings
n_pointings = 2
n_pixels = 7

vmin=-100.
vmax=400.

for i in range(n_pointings):	

	for j in range(n_pixels):
		hi_dat = pd.read_csv('../Data/Outflow_Pointings/HI/outflow'+str(i+1)+'_pix'+str(j)+'_smo.csv')
		this_vel = hi_dat['Velocity_kms'].values
		this_spec = hi_dat['Intensity_K'].values
		this_vel_og = this_vel.copy()
		this_spec_og = this_spec.copy()
		idx = np.where((this_vel >= vmin) & (this_vel <= vmax))[0]
		this_spec = this_spec[idx]
		this_vel = this_vel[idx]

		dv = np.abs(this_vel[0]-this_vel[1]) #km/s
		tab = make_moments_from_spectra(this_vel,this_spec,dv,nsig=2.,vel_all=this_vel_og,spec_all=this_spec_og,vmin=vmin,vmax=vmax)
		this_df = pd.DataFrame.from_dict(tab)
		this_df.to_csv('../Data/Outflow_Pointings/HI_Moments/outflow'+str(i+1)+'_pix'+str(j)+'_momentparams.csv',index=False)


