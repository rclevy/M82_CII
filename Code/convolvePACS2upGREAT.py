#convolve the PACS cube to a 14" Gaussian
#uses the convolution kernels from Aniano+2011

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
import astropy.units as u
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube
from makeMomentsfromSpectra import make_moments_from_spectra

#open the PACS cube
cube = fits.open('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube.fits')
image = cube[1].data
hdr = cube[1].header

#use kernel from Aniano+2011
kernel = fits.open('../Data/Ancillary_Data/Kernels/Kernel_LowRes_PACS_160_to_Gauss_14.0.fits')
hdr_kern = kernel[0].header

#match the kernel pixel scale to the image
px_sc_im = hdr['CDELT2']*3600 #arcsec
px_sc_kern = hdr_kern['CD2_2']*3600 #arcsec
sf = px_sc_im/px_sc_kern
naxis_kern = np.array([hdr_kern['NAXIS1'], hdr_kern['NAXIS2']])
naxis_kern_res = np.round(naxis_kern/sf).astype('int')
if np.mod(naxis_kern_res[0],2)==0: #kernel must be odd
	naxis_kern_res[0]+=1
if np.mod(naxis_kern_res[1],2)==0:
	naxis_kern_res[1]+=1
zm = naxis_kern_res/naxis_kern
kernel_res = zoom(kernel[0].data,zm)
kernel_res[kernel_res==-0.]=0.

#but the kernel doesn't need to be this large in extent
#and this slows down the convolution a lot
cen = np.array([(kernel_res.shape[0]-1)/2,(kernel_res.shape[1]-1)/2]).astype('int')
crop = np.array([(image.shape[1]-1)/2,(image.shape[2]-1)/2]).astype('int')
kernel_res_crop = kernel_res[cen[0]-crop[0]:cen[0]+crop[0]+1,cen[1]-crop[1]:cen[1]+crop[1]+1]

#now do the convolution
#this is slow!
image_conv = np.zeros_like(image)
for i in range(image_conv.shape[0]):
	image_conv[i,:,:] = convolve(image[i,:,:],kernel_res_crop,preserve_nan=True)

#save the convolved cube
hdr_conv = hdr.copy()
hdu0 = fits.PrimaryHDU(data=cube[0].data,header=cube[0].header)
hdu = fits.ImageHDU(data=image_conv,header=hdr_conv)
hdul = fits.HDUList([hdu0,hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_14asGaussPSF.fits',overwrite=True)

#make peak and integrated intensity maps from the background subtracted cube
wl0 = cube[0].header['MEANWAVE']*u.micron
z = cube[0].header['REDSHFTV']
rest_wl = wl0/(1+z)
hdr_conv['CTYPE3']='WAVE'
c = SpectralCube(data=image_conv*u.Jy/u.pixel,wcs=WCS(hdr_conv)).with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=rest_wl)
c.allow_huge_operations = True

#only compute the moments over the same velocity range as the [CII] data for consistency
cii_cube = SpectralCube.read('../Data/Disk_Map/M82_CII_map_cube.fits').with_spectral_unit(u.km/u.s)*u.K
cii_v_all = cii_cube.spectral_axis.value #km/s
vmin = -100.*u.km/u.s
vmax = 400.*u.km/u.s
idx = np.where((cii_v_all>=vmin.value) & (cii_v_all<=vmax.value))
cii_v = cii_v_all[idx]
c_all = c.spectral_slab(np.nanmin(cii_v)*u.km/u.s,np.nanmax(cii_v)*u.km/u.s)
c = c.spectral_slab(vmin,vmax)
dv = 10. #km/s


#now fit with a Gaussian at each pixel
#and find the moments at each pixel
#remove continuum/baseline and measure [CII] peak intensity
#and so peak and intinten are measiured in the same way as the SOFIA data
def gauss(x,a,b,c,d):
	return a*np.exp((-(x-b)**2/(2*c**2)))+d

peak = np.zeros((image_conv.shape[1],image_conv.shape[2]))
bkgd = np.zeros((image_conv.shape[1],image_conv.shape[2]))
intinten = np.zeros((image_conv.shape[1],image_conv.shape[2]))
mmax = np.zeros((image_conv.shape[1],image_conv.shape[2]))
mom0 = np.zeros((image_conv.shape[1],image_conv.shape[2]))
chans =  np.flip(cii_v)
bounds = [[0.,chans[0],0.,0.],[np.inf,chans[-1],np.inf,np.inf]]
for i in range(image_conv.shape[1]):
	for j in range(image_conv.shape[2]):
		this_spec = c[:,i,j].value
		if np.all(np.isnan(this_spec))==True:
			peak[i,j] = np.nan
			bkgd[i,j] = np.nan
			intinten[i,j] = np.nan
		else:
			idx = np.where(np.isnan(this_spec)==False)
			init = [np.nanmax(this_spec[idx]),250.,100.,0.]
			popt,_ = curve_fit(gauss,chans[idx],this_spec[idx],bounds=bounds)
			peak[i,j]=popt[0]
			bkgd[i,j]=popt[-1]
			intinten[i,j] = popt[0]*popt[2]*np.sqrt(np.pi/2)
		tab = make_moments_from_spectra(chans,this_spec,dv,2.)
		mom0[i,j] = tab['Mom0'].values[0]
		mmax[i,j] = tab['Max'].values[0]

#now save the peak map
wcs_peak = WCS(hdr).celestial
hdr_peak = wcs_peak.to_header()
hdr_peak['BUNIT'] = hdr['BUNIT']
hdu = fits.PrimaryHDU(data=peak,header=hdr_peak)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_peak_14asGaussPSF_contsub.fits',overwrite=True)
hdu = fits.PrimaryHDU(data=mmax,header=hdr_peak)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_max_14asGaussPSF_contsub.fits',overwrite=True)



wcs_intinten = WCS(hdr).celestial
hdr_intinten = wcs_intinten.to_header()
hdr_intinten['BUNIT'] = hdr['BUNIT']+' km s-1'
hdu = fits.PrimaryHDU(data=intinten,header=hdr_intinten)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_intinten_14asGaussPSF_contsub.fits',overwrite=True)
hdu = fits.PrimaryHDU(data=mom0,header=hdr_intinten)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_mom0_14asGaussPSF_contsub.fits',overwrite=True)


#save the background subtracted cube
image_conv_contsub = np.zeros_like(image_conv)
for i in range(image_conv.shape[0]):
	image_conv_contsub[i,:,:] = image_conv[i,:,:]-bkgd
hdu = fits.ImageHDU(data=image_conv_contsub,header=hdr_conv)
hdul = fits.HDUList([hdu0,hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_14asGaussPSF_contsub.fits',overwrite=True)


# peak = c.max(axis=0)
# peak.write('../Data/Ancillary_Data/Herschel_PACS_158um_peak_14asGaussPSF_contsub.fits',overwrite=True)

# #save mom0
# cc = c.unmasked_data[:].value-bkgd
# dv = chans[0]-chans[1]
# mom0 = np.nansum(cc,axis=0)*dv
# hdu = fits.PrimaryHDU(data=mom0,header=hdr_intinten)
# hdul = fits.HDUList([hdu])
# hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_mom0_14asGaussPSF_contsub.fits',overwrite=True)




