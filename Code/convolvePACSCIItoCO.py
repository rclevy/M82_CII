#convolve the PACS cube to a 22" Gaussian matching CO
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
from reproject import reproject_adaptive

#open the PACS cube
cube = fits.open('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube.fits')
image = cube[1].data
hdr = cube[1].header

#use kernel from Aniano+2011
kernel = fits.open('../Data/Ancillary_Data/Kernels/Kernel_LowRes_PACS_160_to_Gauss_22.0.fits')
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
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_22asGaussPSF.fits',overwrite=True)

#make peak and integrated intensity maps from the background subtracted cube
wl0 = cube[0].header['MEANWAVE']*u.micron
z = cube[0].header['REDSHFTV']
rest_wl = wl0/(1+z)
hdr_conv['CTYPE3']='WAVE'
c = SpectralCube(data=image_conv*u.Jy/u.pixel,wcs=WCS(hdr_conv)).with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=rest_wl)
c.allow_huge_operations = True

#only compute the moments over the same velocity range as the [CII] data for consistency
# cii_cube = SpectralCube.read('../Data/Disk_Map/M82_CII_map_cube.fits').with_spectral_unit(u.km/u.s)*u.K
# cii_v_all = cii_cube.spectral_axis.value #km/s
cii_v_all = c.spectral_axis.value #km/s
vmin = -104.*u.km/u.s
vmax = 400.*u.km/u.s
idx = np.where((cii_v_all>=vmin.value) & (cii_v_all<=vmax.value))
cii_v = cii_v_all[idx]
c_all = c.spectral_slab(np.nanmin(cii_v)*u.km/u.s,np.nanmax(cii_v)*u.km/u.s)
c = c.spectral_slab(vmin,vmax)
dv = 10. #km/s


#now find the moments at each pixel
mom0 = np.zeros((image_conv.shape[1],image_conv.shape[2]))
chans = cii_v.copy()
bounds = [[0.,chans[0],0.,0.],[np.inf,chans[-1],np.inf,np.inf]]
for i in range(image_conv.shape[1]):
	for j in range(image_conv.shape[2]):
		this_spec = c[:,i,j].value
		tab = make_moments_from_spectra(chans,this_spec,dv,2.)
		mom0[i,j] = tab['Mom0'][0]#.values[0]

wcs_intinten = WCS(hdr).celestial
hdr_intinten = wcs_intinten.to_header()
hdu = fits.PrimaryHDU(data=mom0,header=hdr_intinten)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_mom0_22asGaussPSF.fits',overwrite=True)


#now reproject the PACS CII onto the matched CII/CO grid
#open the CO data
co_matched = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits')[0]
co_mom0 = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0.fits')[0]
co_hdr = co_mom0.header
co_mom0 = co_mom0.data
hdr_matched = co_matched.header
hdr_matched['NAXIS1'] = co_hdr['NAXIS1']
hdr_matched['NAXIS2'] = co_hdr['NAXIS2']

cii_mom0_matched,_ = reproject_adaptive(hdu,co_matched.header,conserve_flux=True)
co_mom0_matched,_ = reproject_adaptive(fits.PrimaryHDU(data=co_mom0,header=co_hdr),co_matched.header,conserve_flux=False)

#make new header
hdr_matched['BMAJ'] = 22./3600
hdr_matched['BMIN'] = 22./3600
hdr_matched['BPA'] = 0.
hdr_matched['BUNIT'] = 'Jy/pixel km/s'
fits.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_mom0_22asGaussPSF_matchedPACSCIICO.fits',
	data=cii_mom0_matched,header=hdr_matched,overwrite=True)
fits.writeto('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedPACSCIICO.fits',
	data=co_mom0_matched,header=hdr_matched,overwrite=True)




