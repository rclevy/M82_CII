this_script = __file__

#Extract CO and HI spectra in the same apertures as the SOFIA CII single pointings

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import upGREATUtils
import pandas as pd
from photutils.aperture import SkyCircularAperture,aperture_photometry
from spectral_cube import SpectralCube

gal_center = SkyCoord('09h55m52.72s 69d40m45.8s')
dist = 3.5*u.Mpc
pix_diameter = 14.1*u.arcsec
n_pointings=7

#load the CII spectra from each outflow pointing
cii_outflow1 = pd.DataFrame(columns=['Pixel_Number','Pix_Center','Velocity_kms','Intensity_K','Header'])
cii_outflow2 = pd.DataFrame(columns=['Pixel_Number','Pix_Center','Velocity_kms','Intensity_K','Header'])
for i in range(0,n_pointings):
	this_spec, this_vax, this_hdr, _ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow1_pix'+str(i)+'.fits')
	#get RA and Dec of pixel center
	_,_,this_pix_cen = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, this_hdr, dist)
	cii_outflow1 = cii_outflow1.append({'Pixel_Number': i, 'Pix_Center': this_pix_cen, 'Velocity_kms': this_vax, 'Intensity_K': this_spec, 'Header': this_hdr}, ignore_index=True)

	this_spec, this_vax, this_hdr, _ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow2_pix'+str(i)+'.fits')
	_,_,this_pix_cen = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, this_hdr, dist)
	cii_outflow2 = cii_outflow2.append({'Pixel_Number': i, 'Pix_Center': this_pix_cen, 'Velocity_kms': this_vax, 'Intensity_K': this_spec, 'Header': this_hdr}, ignore_index=True)


#Now open the CO cube
fname_co = '../Data/Ancillary_Data/M82-NOEMA-30m-mask-merged.clean.fits'
co_cube = fits.open(fname_co)
co_cube_hdr = co_cube[0].header
co_cube = co_cube[0].data
co_cube_K = 1.222E6*co_cube/((co_cube_hdr['RESTFREQ']/1E9)**2*(co_cube_hdr['BMAJ']*3600)*(co_cube_hdr['BMIN']*3600))
co_wcs = WCS(co_cube_hdr)
co_vel_axis = SpectralCube.read(fname_co).with_spectral_unit(u.km/u.s).spectral_axis.value

#Now open the HI cube
fname_hi = '../Data/Ancillary_Data/m82_hi_24as_feathered_nhi.fits'
hi_cube = fits.open(fname_hi)
hi_cube_hdr = hi_cube[0].header
hi_cube = hi_cube[0].data
hi_wcs = WCS(hi_cube_hdr)
hi_vel_axis = SpectralCube.read(fname_hi).with_spectral_unit(u.km/u.s).spectral_axis.value
	
#make apertures at the locations
apers_outflow1 = ['']*n_pointings
apers_outflow2 = ['']*n_pointings
for i in range(0,n_pointings):
	apers_outflow1[i] = SkyCircularAperture(cii_outflow1['Pix_Center'].values[i],r=pix_diameter/2)
	apers_outflow2[i] = SkyCircularAperture(cii_outflow2['Pix_Center'].values[i],r=pix_diameter/2)

#now extract those pixels from the cubes
co_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_cm-2'])
hi_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_cm-2'])
for i in range(0,n_pointings):
	co_mask_1 = apers_outflow1[i].to_pixel(co_wcs.celestial).to_mask(method='center')
	co_mask_2 = apers_outflow2[i].to_pixel(co_wcs.celestial).to_mask(method='center')
	co_spec_1 = np.zeros((co_cube.shape[0],))*np.nan
	co_spec_2 = np.zeros((co_cube.shape[0],))*np.nan
	for j in range(co_cube.shape[0]):
		co_spec_1[j] = np.nansum(co_mask_1.multiply(co_cube_K[j,:,:],fill_value=np.nan))
		co_spec_2[j] = np.nansum(co_mask_2.multiply(co_cube_K[j,:,:],fill_value=np.nan))
	co_outflow1 = co_outflow1.append({'Velocity_kms': co_vel_axis, 'Intensity_K': co_spec_1}, ignore_index=True)
	co_outflow2 = co_outflow2.append({'Velocity_kms': co_vel_axis, 'Intensity_K': co_spec_2}, ignore_index=True)

	hi_mask_1 = apers_outflow1[i].to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_mask_2 = apers_outflow2[i].to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_spec_1 = np.zeros((hi_cube.shape[0],))*np.nan
	hi_spec_2 = np.zeros((hi_cube.shape[0],))*np.nan
	for j in range(hi_cube.shape[0]):
		hi_spec_1[j] = np.nansum(hi_mask_1.multiply(hi_cube[j,:,:],fill_value=np.nan))
		hi_spec_2[j] = np.nansum(hi_mask_2.multiply(hi_cube[j,:,:],fill_value=np.nan))
	hi_outflow1 = hi_outflow1.append({'Velocity_kms': hi_vel_axis, 'Intensity_cm-2': hi_spec_1}, ignore_index=True)
	hi_outflow2 = hi_outflow2.append({'Velocity_kms': hi_vel_axis, 'Intensity_cm-2': hi_spec_2}, ignore_index=True)


#now write the CO and HI spectra to files
for i in range(0,n_pointings):
	tab = {'Velocity_kms': co_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': co_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow2_pix'+str(i)+'.csv',index=False)

	tab = {'Velocity_kms': hi_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_cm-2': hi_outflow1.iloc[[i]]['Intensity_cm-2'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_cm-2': hi_outflow2.iloc[[i]]['Intensity_cm-2'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'.csv',index=False)







