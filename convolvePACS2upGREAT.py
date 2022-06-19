#convolve the PACS cube to a 14" Gaussian
#uses the convolution kernels from Aniano+2011

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from scipy.ndimage import zoom
from scipy.optimize import curve_fit

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


#now fit with a Gaussian at each pixel
#remove continuum/baseline and measure [CII] peak intensity
def gauss(x,a,b,c,d):
	return a*np.exp((-(x-b)**2/(2*c**2)))+d

peak = np.zeros((image_conv.shape[1],image_conv.shape[2]))
bkgd = np.zeros((image_conv.shape[1],image_conv.shape[2]))
#make a fake spectral axis, since we only want the peak, it doesn't matter
chans = np.arange(1,image_conv.shape[0]+1,1)
bounds = [[0.,chans[0],0.,0.],[np.inf,chans[-1],np.inf,np.inf]]
for i in range(image_conv.shape[1]):
	for j in range(image_conv.shape[2]):
		this_spec = image_conv[:,i,j]
		if np.all(np.isnan(this_spec))==True:
			peak[i,j] = np.nan
			bkgd[i,j] = np.nan
		else:
			idx = np.where(np.isnan(this_spec)==False)
			popt,_ = curve_fit(gauss,chans[idx],this_spec[idx],bounds=bounds)
			peak[i,j]=popt[0]
			bkgd[i,j]=popt[-1]

#save the background subtracted cube
image_conv_contsub = np.zeros_like(image_conv)
for i in range(image_conv.shape[0]):
	image_conv_contsub[i,:,:] = image_conv[i,:,:]-bkgd
hdu = fits.ImageHDU(data=image_conv_contsub,header=hdr_conv)
hdul = fits.HDUList([hdu0,hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_spec_158um_cube_14asGaussPSF_contsub.fits',overwrite=True)


#now save the peak map
# peak = np.nanmax(image_conv,axis=0)
wcs_peak = WCS(hdr).celestial
hdr_peak = wcs_peak.to_header()
hdr_peak['BUNIT'] = hdr['BUNIT']
hdu = fits.PrimaryHDU(data=peak,header=hdr_peak)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/Herschel_PACS_158um_peak_14asGaussPSF_contsub.fits',overwrite=True)

