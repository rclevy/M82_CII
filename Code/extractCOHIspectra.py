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
from reproject import reproject_interp
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


#open the CO TP cube
fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed.fits'
#fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.map.fits' #has some weird absoprtion
co_tp_cube_fits = fits.open(fname_co_tp)
co_tp_cube_hdr = co_tp_cube_fits[0].header
co_tp_cube = co_tp_cube_fits[0].data
co_tp_wcs = WCS(co_tp_cube_hdr)
co_tp_vel_axis = SpectralCube.read(fname_co_tp).with_spectral_unit(u.km/u.s).spectral_axis.value

#Now open the HI cube
fname_hi = '../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor.fits'
hi_cube_fits = fits.open(fname_hi) #Jy/beam
hi_cube_hdr = hi_cube_fits[0].header
hi_cube = hi_cube_fits[0].data
hi_wcs = WCS(hi_cube_hdr)
hi_vel_axis = SpectralCube.read(fname_hi).with_spectral_unit(u.km/u.s).spectral_axis.value
#convert HI cube from Jy/beam to K
hi_cube = 1.222E6*hi_cube/((hi_cube_hdr['RESTFRQ']/1E9)**2*hi_cube_hdr['BMAJ']*hi_cube_hdr['BMIN']*3600**2)
hi_cube_fits[0].data = hi_cube


# #regrid the CO and HI cubes to have 14" pixels
# cii_pix_area = np.pi*(pix_diameter/2)**2
# cii_pix_sz_square = np.sqrt(cii_pix_area).to(u.deg).value
# n_pix_x_new = int(hi_cube_hdr['NAXIS1']*np.abs(hi_cube_hdr['CDELT1']/cii_pix_sz_square))
# n_pix_y_new = int(hi_cube_hdr['NAXIS2']*np.abs(hi_cube_hdr['CDELT2']/cii_pix_sz_square))
# cen_pix_x_new = hi_cube_hdr['CRPIX1']/hi_cube_hdr['NAXIS1']*n_pix_x_new
# cen_pix_y_new = hi_cube_hdr['CRPIX2']/hi_cube_hdr['NAXIS2']*n_pix_y_new
# hi_hdr_new = hi_cube_hdr.copy()
# hi_hdr_new['NAXIS1']=n_pix_x_new
# hi_hdr_new['NAXIS2']=n_pix_y_new
# hi_hdr_new['CDELT2']=cii_pix_sz_square
# hi_hdr_new['CDELT1']=-cii_pix_sz_square
# hi_hdr_new['CRPIX1']=cen_pix_x_new
# hi_hdr_new['CRPIX2']=cen_pix_y_new
# hi_cube_repro,_ = reproject_interp(hi_cube_fits,hi_hdr_new)
# fits.writeto('../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor_K_14as.fits',hi_cube_repro,hi_hdr_new,overwrite=True)

# n_pix_x_new = int(co_tp_cube_hdr['NAXIS1']*np.abs(co_tp_cube_hdr['CDELT1']/cii_pix_sz_square))
# n_pix_y_new = int(co_tp_cube_hdr['NAXIS2']*np.abs(co_tp_cube_hdr['CDELT2']/cii_pix_sz_square))
# cen_pix_x_new = co_tp_cube_hdr['CRPIX1']/co_tp_cube_hdr['NAXIS1']*n_pix_x_new
# cen_pix_y_new = co_tp_cube_hdr['CRPIX2']/co_tp_cube_hdr['NAXIS2']*n_pix_y_new
# co_tp_hdr_new = co_tp_cube_hdr.copy()
# co_tp_hdr_new['NAXIS1']=n_pix_x_new
# co_tp_hdr_new['NAXIS2']=n_pix_y_new
# co_tp_hdr_new['CDELT2']=cii_pix_sz_square
# co_tp_hdr_new['CDELT1']=-cii_pix_sz_square
# co_tp_hdr_new['CRPIX1']=cen_pix_x_new
# co_tp_hdr_new['CRPIX2']=cen_pix_y_new
# co_tp_cube_repro,_ = reproject_interp(co_tp_cube_fits,co_tp_hdr_new)
# fits.writeto('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_14as.fits',co_tp_cube_repro,co_tp_hdr_new,overwrite=True)



#now extract the central pixel of each upGREAT pointing from the cubes
hi_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
# hi_outflow1_smo_14 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
# hi_outflow2_smo_14 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow1_smo_sum = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
hi_outflow2_smo_sum = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow1 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow2 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow1_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
co_tp_outflow2_smo = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
# co_tp_outflow1_smo_14 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])
# co_tp_outflow2_smo_14 = pd.DataFrame(columns=['Velocity_kms','Intensity_K'])

for i in range(0,n_pointings):
	#get central pixel
	cen_pix_1_hi = hi_wcs.celestial.all_world2pix(cii_outflow1['Pix_Center'][i].ra.value,cii_outflow1['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	cen_pix_1_co_tp = co_tp_wcs.celestial.all_world2pix(cii_outflow1['Pix_Center'][i].ra.value,cii_outflow1['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	cen_pix_2_hi = hi_wcs.celestial.all_world2pix(cii_outflow2['Pix_Center'][i].ra.value,cii_outflow2['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	cen_pix_2_co_tp = co_tp_wcs.celestial.all_world2pix(cii_outflow2['Pix_Center'][i].ra.value,cii_outflow2['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	# cen_pix_1_hi_14 = WCS(hi_hdr_new).celestial.all_world2pix(cii_outflow1['Pix_Center'][i].ra.value,cii_outflow1['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	# cen_pix_1_co_tp_14 = WCS(co_tp_hdr_new).celestial.all_world2pix(cii_outflow1['Pix_Center'][i].ra.value,cii_outflow1['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	# cen_pix_2_hi_14 = WCS(hi_hdr_new).celestial.all_world2pix(cii_outflow2['Pix_Center'][i].ra.value,cii_outflow2['Pix_Center'][i].dec.value,1,ra_dec_order=True)
	# cen_pix_2_co_tp_14 = WCS(co_tp_hdr_new).celestial.all_world2pix(cii_outflow2['Pix_Center'][i].ra.value,cii_outflow2['Pix_Center'][i].dec.value,1,ra_dec_order=True)

	#extact central pixel spectrum
	hi_spec_1 = hi_cube[:,int(np.round(cen_pix_1_hi[1])),int(np.round(cen_pix_1_hi[0]))]
	hi_spec_2 = hi_cube[:,int(np.round(cen_pix_2_hi[1])),int(np.round(cen_pix_2_hi[0]))]
	co_tp_spec_1 = co_tp_cube[:,int(np.round(cen_pix_1_co_tp[1])),int(np.round(cen_pix_1_co_tp[0]))]
	co_tp_spec_2 = co_tp_cube[:,int(np.round(cen_pix_2_co_tp[1])),int(np.round(cen_pix_2_co_tp[0]))]
	# hi_spec_1_14 = hi_cube_repro[:,int(np.round(cen_pix_1_hi_14[1])),int(np.round(cen_pix_1_hi_14[0]))]
	# hi_spec_2_14 = hi_cube_repro[:,int(np.round(cen_pix_2_hi_14[1])),int(np.round(cen_pix_2_hi_14[0]))]
	# co_tp_spec_1_14 = co_tp_cube_repro[:,int(np.round(cen_pix_1_co_tp_14[1])),int(np.round(cen_pix_1_co_tp_14[0]))]
	# co_tp_spec_2_14 = co_tp_cube_repro[:,int(np.round(cen_pix_2_co_tp_14[1])),int(np.round(cen_pix_2_co_tp_14[0]))]


	#for the HI column density, we need the summed spectrum in the same region
	aper_outflow1 = SkyCircularAperture(cii_outflow1['Pix_Center'].values[i],r=pix_diameter/2)
	aper_outflow2 = SkyCircularAperture(cii_outflow2['Pix_Center'].values[i],r=pix_diameter/2)
	hi_mask_1 = aper_outflow1.to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_mask_2 = aper_outflow2.to_pixel(hi_wcs.celestial).to_mask(method='center')
	hi_spec_sum_1 = np.zeros((hi_cube.shape[0],))*np.nan
	hi_spec_sum_2 = np.zeros((hi_cube.shape[0],))*np.nan
	for j in range(hi_cube.shape[0]):
		hi_spec_sum_1[j] = np.nansum(hi_mask_1.multiply(hi_cube[j,:,:],fill_value=np.nan))
		hi_spec_sum_2[j] = np.nansum(hi_mask_2.multiply(hi_cube[j,:,:],fill_value=np.nan))

	#append to data frame
	hi_outflow1 = hi_outflow1.append({'Velocity_kms': hi_vel_axis, 'Intensity_K': hi_spec_1}, ignore_index=True)
	hi_outflow2 = hi_outflow2.append({'Velocity_kms': hi_vel_axis, 'Intensity_K': hi_spec_2}, ignore_index=True)
	co_tp_outflow1 = co_tp_outflow1.append({'Velocity_kms': co_tp_vel_axis, 'Intensity_K': co_tp_spec_1}, ignore_index=True)
	co_tp_outflow2 = co_tp_outflow2.append({'Velocity_kms': co_tp_vel_axis, 'Intensity_K': co_tp_spec_2}, ignore_index=True)

	#smooth the the same velocity resolution as the CII
	fwhm_factor = np.sqrt(8*np.log(2))
	cii_velres = 10. * u.km/u.s

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


	hi_spec_1_smo_sum = convolve(hi_spec_sum_1,hi_kernel)
	hi_1_int_sum = interp1d(hi_vel_axis,hi_spec_sum_1)
	hi_2_int_sum = interp1d(hi_vel_axis,hi_spec_sum_2)
	hi_spec_1_smo_int_sum = hi_1_int_sum(hi_vel_smo)
	hi_spec_2_smo_int_sum = hi_2_int_sum(hi_vel_smo)
	#for the HI, replace the channel at 20 km/s with ave of neighbors since it appears to be bad in all pixels
	idx = np.where((hi_vel_smo > 15.) & (hi_vel_smo < 25.))[0]
	hi_spec_1_smo_int_sum[idx] = np.nanmean([hi_spec_1_smo_int_sum[idx-1],hi_spec_1_smo_int_sum[idx+1]])
	hi_outflow1_smo_sum = hi_outflow1_smo_sum.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_1_smo_int_sum}, ignore_index=True)
	hi_outflow2_smo_sum = hi_outflow2_smo_sum.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_2_smo_int_sum}, ignore_index=True)


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


	# hi_spec_1_smo_14 = convolve(hi_spec_1_14,hi_kernel)
	# hi_1_int_14 = interp1d(hi_vel_axis,hi_spec_1_14)
	# hi_2_int_14 = interp1d(hi_vel_axis,hi_spec_2_14)
	# hi_spec_1_smo_int_14 = hi_1_int_14(hi_vel_smo)
	# hi_spec_2_smo_int_14 = hi_2_int_14(hi_vel_smo)
	# #for the HI, replace the channel at 20 km/s with ave of neighbors since it appears to be bad in all pixels
	# idx = np.where((hi_vel_smo > 15.) & (hi_vel_smo < 25.))[0]
	# hi_spec_1_smo_int_14[idx] = np.nanmean([hi_spec_1_smo_int_14[idx-1],hi_spec_1_smo_int_14[idx+1]])
	# hi_outflow1_smo_14 = hi_outflow1_smo_14.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_1_smo_int_14}, ignore_index=True)
	# hi_outflow2_smo_14 = hi_outflow2_smo_14.append({'Velocity_kms': hi_vel_smo, 'Intensity_K': hi_spec_2_smo_int_14}, ignore_index=True)

	# co_tp_spec_1_smo_14 = convolve(co_tp_spec_1_14,co_tp_kernel)
	# co_tp_1_int_14 = interp1d(co_tp_vel_axis,co_tp_spec_1_14)
	# co_tp_2_int_14 = interp1d(co_tp_vel_axis,co_tp_spec_2_14)
	# co_tp_spec_1_smo_int_14 = co_tp_1_int_14(co_tp_vel_smo)
	# co_tp_spec_2_smo_int_14 = co_tp_2_int_14(co_tp_vel_smo)
	# co_tp_outflow1_smo_14 = co_tp_outflow1_smo_14.append({'Velocity_kms': co_tp_vel_smo, 'Intensity_K': co_tp_spec_1_smo_int_14}, ignore_index=True)
	# co_tp_outflow2_smo_14 = co_tp_outflow2_smo_14.append({'Velocity_kms': co_tp_vel_smo, 'Intensity_K': co_tp_spec_2_smo_int_14}, ignore_index=True)



#now write the spectra to files
for i in range(0,n_pointings):

	tab = {'Velocity_kms': hi_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'.csv',index=False)

	tab = {'Velocity_kms': hi_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'_smo.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'_smo.csv',index=False)

	tab = {'Velocity_kms': hi_outflow1_smo_sum.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1_smo_sum.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'_smo_summed.csv',index=False)
	tab = {'Velocity_kms': hi_outflow2_smo_sum.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2_smo_sum.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'_smo_summed.csv',index=False)


	# tab = {'Velocity_kms': hi_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow1_smo_14.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(i)+'_smo_14as.csv',index=False)
	# tab = {'Velocity_kms': hi_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': hi_outflow2_smo_14.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/HI/outflow2_pix'+str(i)+'_smo_14as.csv',index=False)


	tab = {'Velocity_kms': co_tp_outflow1.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow1.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(i)+'.csv',index=False)
	tab = {'Velocity_kms': co_tp_outflow2.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow2.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow2_pix'+str(i)+'.csv',index=False)

	tab = {'Velocity_kms': co_tp_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow1_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(i)+'_smo.csv',index=False)
	tab = {'Velocity_kms': co_tp_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow2_smo.iloc[[i]]['Intensity_K'].values[0]}
	pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow2_pix'+str(i)+'_smo.csv',index=False)


	# tab = {'Velocity_kms': co_tp_outflow1_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow1_smo_14.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(i)+'_smo_14as.csv',index=False)
	# tab = {'Velocity_kms': co_tp_outflow2_smo.iloc[[i]]['Velocity_kms'].values[0], 'Intensity_K': co_tp_outflow2_smo_14.iloc[[i]]['Intensity_K'].values[0]}
	# pd.DataFrame.from_dict(tab).to_csv('../Data/Outflow_Pointings/CO_TP/outflow2_pix'+str(i)+'_smo_14as.csv',index=False)


