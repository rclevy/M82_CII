this_script = __file__

#Extract CO and HI spectra in the same apertures as the SOFIA CII single pointings
#extract from the CENTRAL PIXEL
#note this doesn't do any beam matching


import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.convolution import Gaussian1DKernel,convolve
import upGREATUtils
import pandas as pd
from photutils.aperture import SkyCircularAperture,aperture_photometry
from spectral_cube import SpectralCube
from scipy.interpolate import interp1d


gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
dist = 3.63*u.Mpc
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


# #Now open the CO cube
# fname_co = '../Data/Ancillary_Data/M82-NOEMA-30m-mask-merged.clean_smoothedCII.fits'
# co_cube = fits.open(fname_co)
# co_cube_hdr = co_cube[0].header
# co_cube = co_cube[0].data
# co_cube_K = 1.222E6*co_cube/((co_cube_hdr['RESTFREQ']/1E9)**2*(co_cube_hdr['BMAJ']*3600)*(co_cube_hdr['BMIN']*3600))
# co_wcs = WCS(co_cube_hdr)
# co_vel_axis = SpectralCube.read(fname_co).with_spectral_unit(u.km/u.s).spectral_axis.value


#open the CO TP cube
fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed.fits'
#fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.map.fits' #has some weird absoprtion
co_tp_cube = fits.open(fname_co_tp)
co_tp_cube_hdr = co_tp_cube[0].header
co_tp_cube = co_tp_cube[0].data
co_tp_wcs = WCS(co_tp_cube_hdr)
co_tp_vel_axis = SpectralCube.read(fname_co_tp).with_spectral_unit(u.km/u.s).spectral_axis.value


#Now open the HI cube
fname_hi = '../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor.fits'
hi_cube = fits.open(fname_hi) #Jy/beam
hi_cube_hdr = hi_cube[0].header
hi_cube = hi_cube[0].data
hi_wcs = WCS(hi_cube_hdr)
hi_vel_axis = SpectralCube.read(fname_hi).with_spectral_unit(u.km/u.s).spectral_axis.value
#convert HI cube from Jy/beam to K
hi_cube = 1.222E6*hi_cube/((hi_cube_hdr['RESTFRQ']/1E9)**2*hi_cube_hdr['BMAJ']*hi_cube_hdr['BMIN']*3600**2)

# #now open the PACS CII cube
# fname_pacs = '../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_14asGaussPSF_contsub.fits'
# pacs_cube = fits.open(fname_pacs) #Jy/pixel
# wl0 = pacs_cube[0].header['MEANWAVE']*u.micron
# z = pacs_cube[0].header['REDSHFTV']
# vframe = pacs_cube[0].header['VFRAME']
# rest_wl = wl0/(1+z)
# pacs_cube_hdr = pacs_cube[1].header
# pacs_cube_hdr['CTYPE3'] = 'WAVE'
# pacs_cube = pacs_cube[1].data
# pacs_wcs = WCS(pacs_cube_hdr)
# sc = SpectralCube(data=pacs_cube,wcs=pacs_wcs)
# pacs_vel_axis = sc.with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=rest_wl).spectral_axis.value
# pacs_vel_axis = pacs_vel_axis-vframe

#make apertures at the locations
apers_outflow1 = ['']*n_pointings
apers_outflow2 = ['']*n_pointings
for i in range(0,n_pointings):
	apers_outflow1[i] = SkyCircularAperture(cii_outflow1['Pix_Center'].values[i],r=pix_diameter/2)
	apers_outflow2[i] = SkyCircularAperture(cii_outflow2['Pix_Center'].values[i],r=pix_diameter/2)

#now extract those pixels from the cubes
co_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
pacs_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_Jy'])
pacs_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_Jy'])
co_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
pacs_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_Jy'])
pacs_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_Jy'])
co_tp_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
for i in range(0,n_pointings):
	# co_mask_1 = apers_outflow1[i].to_pixel(co_wcs.celestial).to_mask(method='center')
	# co_mask_2 = apers_outflow2[i].to_pixel(co_wcs.celestial).to_mask(method='center')
	# co_spec_1 = np.zeros((co_cube.shape[0],))*np.nan
	# co_spec_2 = np.zeros((co_cube.shape[0],))*np.nan
	# for j in range(co_cube.shape[0]):
	# 	co_spec_1[j] = np.nanmedian(co_mask_1.multiply(co_cube_K[j,:,:],fill_value=np.nan))
	# 	co_spec_2[j] = np.nanmedian(co_mask_2.multiply(co_cube_K[j,:,:],fill_value=np.nan))
	# co_outflow1 = co_outflow1.append({'Velocity_kms': co_vel_axis, 'Intensity_K': co_spec_1}, ignore_index=True)
	# co_outflow2 = co_outflow2.append({'Velocity_kms': co_vel_axis, 'Intensity_K': co_spec_2}, ignore_index=True)

	hi_mask_1 = apers_outflow1[i].to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_mask_2 = apers_outflow2[i].to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_spec_1 = np.zeros((hi_cube.shape[0],))*np.nan
	hi_spec_2 = np.zeros((hi_cube.shape[0],))*np.nan
	for j in range(hi_cube.shape[0]):
		hi_spec_1[j] = np.nanmedian(hi_mask_1.multiply(hi_cube[j,:,:],fill_value=np.nan))
		hi_spec_2[j] = np.nanmedian(hi_mask_2.multiply(hi_cube[j,:,:],fill_value=np.nan))
	hi_outflow1 = hi_outflow1.append({'Velocity_kms': hi_vel_axis, 'Intensity_K': hi_spec_1}, ignore_index=True)
	hi_outflow2 = hi_outflow2.append({'Velocity_kms': hi_vel_axis, 'Intensity_K': hi_spec_2}, ignore_index=True)


	# pacs_mask_1 = apers_outflow1[i].to_pixel(pacs_wcs.celestial).to_mask(method='center')
	# pacs_mask_2 = apers_outflow2[i].to_pixel(pacs_wcs.celestial).to_mask(method='center')
	# pacs_spec_1 = np.zeros((pacs_cube.shape[0],))*np.nan
	# pacs_spec_2 = np.zeros((pacs_cube.shape[0],))*np.nan
	# for j in range(pacs_cube.shape[0]):
	# 	pacs_spec_1[j] = np.nanmedian(pacs_mask_1.multiply(pacs_cube[j,:,:],fill_value=np.nan))
	# 	pacs_spec_2[j] = np.nanmedian(pacs_mask_2.multiply(pacs_cube[j,:,:],fill_value=np.nan))
	# pacs_outflow1 = pacs_outflow1.append({'Velocity_kms': pacs_vel_axis, 'Intensity_Jy': pacs_spec_1}, ignore_index=True)
	# pacs_outflow2 = pacs_outflow2.append({'Velocity_kms': pacs_vel_axis, 'Intensity_Jy': pacs_spec_2}, ignore_index=True)

	co_tp_mask_1 = apers_outflow1[i].to_pixel(co_tp_wcs.celestial).to_mask(method='center')
	co_tp_mask_2 = apers_outflow2[i].to_pixel(co_tp_wcs.celestial).to_mask(method='center')
	co_tp_spec_1 = np.zeros((co_tp_cube.shape[0],))*np.nan
	co_tp_spec_2 = np.zeros((co_tp_cube.shape[0],))*np.nan
	for j in range(co_cube.shape[0]):
		co_tp_spec_1[j] = np.nanmedian(co_tp_mask_1.multiply(co_tp_cube[j,:,:],fill_value=np.nan))
		co_tp_spec_2[j] = np.nanmedian(co_tp_mask_2.multiply(co_tp_cube[j,:,:],fill_value=np.nan))
	co_tp_outflow1 = co_tp_outflow1.append({'Velocity_kms': co_tp_vel_axis, 'Intensity_K': co_tp_spec_1}, ignore_index=True)
	co_tp_outflow2 = co_tp_outflow2.append({'Velocity_kms': co_tp_vel_axis, 'Intensity_K': co_tp_spec_2}, ignore_index=True)


	#smooth the the same velocity resolution as the CII
	fwhm_factor = np.sqrt(8*np.log(2))
	cii_velres = 10. * u.km/u.s

	# co_velres = co_cube_hdr['CDELT3']/1E3 * u.km/u.s
	# co_gaussian_width = ((cii_velres**2-co_velres**2)**0.5/co_velres/fwhm_factor)
	# co_kernel = Gaussian1DKernel(co_gaussian_width.value)
	# co_spec_1_smo = convolve(co_spec_1,co_kernel)
	# co_vel_smo = np.arange(co_vel_axis[0],co_vel_axis[-1],cii_velres.value)
	# co_1_int = interp1d(co_vel_axis,co_spec_1)
	# co_2_int = interp1d(co_vel_axis,co_spec_2)
	# co_spec_1_smo_int = co_1_int(co_vel_smo)
	# co_spec_2_smo_int = co_2_int(co_vel_smo)
	# co_outflow1_smo = co_outflow1_smo.append({'Velocity_kms': co_vel_smo, 'Intensity_K': co_spec_1_smo_int}, ignore_index=True)
	# co_outflow2_smo = co_outflow2_smo.append({'Velocity_kms': co_vel_smo, 'Intensity_K': co_spec_2_smo_int}, ignore_index=True)

	hi_velres = np.abs(hi_cube_hdr['CDELT3']/1E3 * u.km/u.s)
	hi_gaussian_width = ((cii_velres**2-hi_velres**2)**0.5/hi_velres/fwhm_factor)
	hi_kernel = Gaussian1DKernel(hi_gaussian_width.value)
	hi_spec_1_smo = convolve(hi_spec_1,hi_kernel)
	hi_vel_smo = np.arange(hi_vel_axis[0],hi_vel_axis[-1],-cii_velres.value)
	hi_1_int = interp1d(hi_vel_axis,hi_spec_1)
	hi_2_int = interp1d(hi_vel_axis,hi_spec_2)
	hi_spec_1_smo_int = hi_1_int(hi_vel_smo)
	hi_spec_2_smo_int = hi_2_int(hi_vel_smo)
	#for the HI, replace the channel at 20 km/s with ave of neighbors since it appears to be bad in all pixels
	idx = np.where((hi_vel_smo > 15.) & (hi_vel_smo < 25.))[0]
	hi_spec_1_smo_int[idx] = np.nanmean([hi_spec_1_smo_int[idx-1],hi_spec_1_smo_int[idx+1]])
	hi_outflow1_smo = hi_outflow1_smo.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_1_smo_int}, ignore_index=True)
	hi_outflow2_smo = hi_outflow2_smo.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_2_smo_int}, ignore_index=True)


	co_tp_velres = co_tp_cube_hdr['CDELT3']/1E3 * u.km/u.s
	co_tp_gaussian_width = ((cii_velres**2-co_tp_velres**2)**0.5/co_tp_velres/fwhm_factor)
	co_tp_kernel = Gaussian1DKernel(co_tp_gaussian_width.value)
	co_tp_spec_1_smo = convolve(co_tp_spec_1,co_tp_kernel)
	co_tp_vel_smo = np.arange(co_tp_vel_axis[0],co_tp_vel_axis[-1],cii_velres.value)
	co_tp_1_int = interp1d(co_tp_vel_axis,co_tp_spec_1)
	co_tp_2_int = interp1d(co_tp_vel_axis,co_tp_spec_2)
	co_tp_spec_1_smo_int = co_tp_1_int(co_tp_vel_smo)
	co_tp_spec_2_smo_int = co_tp_2_int(co_tp_vel_smo)
	co_tp_outflow1_smo = co_tp_outflow1_smo.append({'Velocity_kms': co_tp_vel_smo, 'Intensity_K': co_tp_spec_1_smo_int}, ignore_index=True)
	co_tp_outflow2_smo = co_tp_outflow2_smo.append({'Velocity_kms': co_tp_vel_smo, 'Intensity_K': co_tp_spec_2_smo_int}, ignore_index=True)


	#pacs is already 10 km/s
	# pacs_outflow1_smo = pacs_outflow1
	# pacs_outflow2_smo = pacs_outflow2





#now write the spectra to files
for i in range(0,n_pointings):
	# tab = {'Velocity_kms': co_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow1_pix'+str(i)+'.csv',index=False)
	# tab = {'Velocity_kms': co_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow2_pix'+str(i)+'.csv',index=False)

	tab = {'Velocity_kms': hi_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'.csv',index=False)

	# tab = {'Velocity_kms': pacs_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_Jy': pacs_outflow1.iloc[[i]]['Intensity_Jy'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/PACS/outflow1_pix'+str(i)+'.csv',index=False)
	# tab = {'Velocity_kms': pacs_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_Jy': pacs_outflow2.iloc[[i]]['Intensity_Jy'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/PACS/outflow2_pix'+str(i)+'.csv',index=False)

	# tab = {'Velocity_kms': co_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow1_smo.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow1_pix'+str(i)+'_smo.csv',index=False)
	# tab = {'Velocity_kms': co_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_outflow2_smo.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO/outflow2_pix'+str(i)+'_smo.csv',index=False)

	tab = {'Velocity_kms': hi_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'_smo.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'_smo.csv',index=False)

	# tab = {'Velocity_kms': pacs_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_Jy': pacs_outflow1_smo.iloc[[i]]['Intensity_Jy'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/PACS/outflow1_pix'+str(i)+'_smo.csv',index=False)
	# tab = {'Velocity_kms': pacs_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_Jy': pacs_outflow2_smo.iloc[[i]]['Intensity_Jy'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/PACS/outflow2_pix'+str(i)+'_smo.csv',index=False)

	tab = {'Velocity_kms': co_tp_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': co_tp_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow2_pix'+str(i)+'.csv',index=False)

	tab = {'Velocity_kms': co_tp_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow1_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(i)+'_smo.csv',index=False)
	tab = {'Velocity_kms': co_tp_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow2_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow2_pix'+str(i)+'_smo.csv',index=False)


