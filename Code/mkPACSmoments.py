#make moment maps of the native PACS data
#do this by fitting Gaussians since the lines are Gaussian and we want to remove the continuum level


import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from spectral_cube import SpectralCube
from scipy.optimize import curve_fit


fits_cube = fits.open('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube.fits')
hdr_cube = fits_cube[1].header
hdr_cube['CTYPE3']='WAVE'
wl0 = fits_cube[0].header['MEANWAVE']*u.micron
z = fits_cube[0].header['REDSHFTV']
rest_wl = wl0/(1+z)
cube = fits_cube[1].data #Jy/pixel
cube = SpectralCube(data=cube*u.Jy/u.pixel,wcs=WCS(hdr_cube)).with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=rest_wl)
cube.allow_huge_operations=True

def gauss(x,a,b,c,d):
	return a*np.exp((-(x-b)**2/(2*c**2)))+d

peak = np.zeros((cube.shape[1],cube.shape[2]))
bkgd = np.zeros((cube.shape[1],cube.shape[2]))
intinten = np.zeros((cube.shape[1],cube.shape[2]))
vel_axis = cube.spectral_axis.value #km/s
for i in range(cube.shape[1]):
	for j in range(cube.shape[2]):
		this_spec = cube[:,i,j].value
		if np.all(np.isnan(this_spec))==True:
			peak[i,j] = np.nan
			bkgd[i,j] = np.nan
			intinten[i,j] = np.nan
		else:
			idx = np.where(np.isnan(this_spec)==False)
			init = [np.nanmax(this_spec[idx]),250.,100.,0.]
			popt,_ = curve_fit(gauss,vel_axis[idx],this_spec[idx],p0=init)
			peak[i,j]=popt[0]
			bkgd[i,j]=popt[-1]
			intinten[i,j] = popt[0]*popt[2]*np.sqrt(np.pi/2)


#now save the peak map
wcs_peak = WCS(hdr_cube).celestial
hdr_peak = wcs_peak.to_header()
hdr_peak['BUNIT'] = hdr_cube['BUNIT']
hdu = fits.PrimaryHDU(data=peak,header=hdr_peak)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_peak_contsub.fits',overwrite=True)

#now save integrated intensity map
wcs_intinten = WCS(hdr_cube).celestial
hdr_intinten = wcs_intinten.to_header()
hdr_intinten['BUNIT'] = hdr_cube['BUNIT']+' km s-1'
hdu = fits.PrimaryHDU(data=intinten,header=hdr_intinten)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_intinten_contsub.fits',overwrite=True)


#save the background subtracted cube
cube_contsub = np.zeros_like(cube)
for i in range(cube.shape[0]):
	cube_contsub[i,:,:] = cube[i,:,:].value-bkgd
hdu0 = fits.PrimaryHDU(data=fits_cube[0].data,header=fits_cube[0].header)
hdu = fits.ImageHDU(data=cube_contsub,header=hdr_cube)
hdul = fits.HDUList([hdu0,hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_contsub.fits',overwrite=True)



#now don't do a continuum subtraction
mom0 = cube.moment(order=0)
mom0.write('../Data/Ancillary_Data/Herschel_PACS_158um_mom0.fits',overwrite=True)
mmax = cube.max(axis=0)
mmax.write('../Data/Ancillary_Data/Herschel_PACS_158um_max.fits',overwrite=True)





