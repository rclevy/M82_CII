#make a FITS image of the [CII] data from SHINING available in table format

#table can be accessed and downloaded here:
#http://tapvizier.u-strasbg.fr/adql/?%20J/ApJ/861/94/table7

import numpy as np
import pandas as pd
from reproject import reproject_interp
from reproject.mosaicking import (find_optimal_celestial_wcs, reproject_and_coadd)
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from scipy.ndimage import zoom

#open the SHINING table
tab = pd.read_csv('../Data/Ancillary_Data/SHINING_M82_CII.csv')
RA = tab['RAJ2000'].values
Dec = tab['DEJ2000'].values
FCII_Wm2 = tab['CII158'].values

#define some properties of the image that are the same for all pixels
pxscale = 47/5 #47"X47" FOV / 5x5 pixels; Section 3 of Herrera-Camus+2018a


hdus = []
for i in range(len(RA)):

	pad_data = np.zeros((3,3))
	pad_data[1,1] = FCII_Wm2[i]

	this_hdu = fits.PrimaryHDU(data=pad_data)
	del this_hdu.header['EXTEND']

	#fill in the header info
	this_hdu.header['OBJECT'] = 'M82'
	this_hdu.header['BUNIT'] = 'W m-2'
	this_hdu.header['CRVAL1'] = RA[i]
	this_hdu.header['CRVAL2'] = Dec[i]
	this_hdu.header['CTYPE1'] = 'RA---TAN' #based on PACS fits file from archive
	this_hdu.header['CTYPE2'] = 'DEC--TAN'
	this_hdu.header['CUNIT1'] = 'deg'
	this_hdu.header['CUNIT2'] = 'deg'
	this_hdu.header['CDELT1'] = -pxscale/3600 #deg
	this_hdu.header['CDELT2'] = pxscale/3600 #deg
	this_hdu.header['CRPIX1'] = 2.
	this_hdu.header['CRPIX2'] = 2.

	hdus.append(this_hdu)

#now use astropy.reproject.mosaic to stitch these together correctly
#based on pltNICMOSData.py
hdrs = [hdu.header for hdu in hdus]
wcs_out, shape_out = find_optimal_celestial_wcs(hdus, auto_rotate=False)
array, footprint = reproject_and_coadd(hdus,wcs_out,shape_out=shape_out,reproject_function=reproject_interp,match_background=False)
array[array==0.]=np.nan
hdu = fits.PrimaryHDU(array,header=wcs_out.to_header())
hdul = fits.HDUList(hdu)
hdul.writeto('../Data/Ancillary_Data/SHINING_M82_CII.fits',overwrite=True)


#now convolve the image to 14"
#based on convolvePACS2upGREAT.py

#use kernel from Aniano+2011
kernel = fits.open('../Data/Ancillary_Data/Kernels/Kernel_LowRes_PACS_160_to_Gauss_14.0.fits')
hdr_kern = kernel[0].header

#match the kernel pixel scale to the image
hdr = hdu.header
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
cen = np.array([(kernel_res.shape[0]-1)/2,(kernel_res.shape[1]-1)/2]).astype('int')
crop = np.array([(array.shape[0]-1)/2,(array.shape[1]-1)/2]).astype('int')
kernel_res_crop = kernel_res[cen[0]-crop[0]:cen[0]+crop[0]+1,cen[1]-crop[1]:cen[1]+crop[1]+1]

array_conv = convolve(array,kernel_res_crop,preserve_nan=True)

hdr_conv = hdr.copy()
hdu = fits.PrimaryHDU(data=array_conv,header=hdr_conv)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/SHINING_M82_CII_14asGaussPSF.fits',overwrite=True)






