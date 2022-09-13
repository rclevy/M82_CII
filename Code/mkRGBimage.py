#make an RGB image of M82 using multicolorfits
#h/t to Liz Watkins for alerting me to this software!

#closely follows this tutorial
#https://github.com/pjcigan/multicolorfits/blob/master/examples/wlm.md


import numpy as np
from astropy.io import fits
import multicolorfits as mcf

#load the HI, CO, and CII images
hi,hdr_hi = fits.getdata('../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor_mom0.fits',header=True)
co,hdr_co = fits.getdata('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0.fits',header=True)
cii,hdr_cii = fits.getdata('../Data/Disk_Map/Moments/M82_CII_map_mom0_maskSNR2.fits',header=True)

#use the CO as the reference header
ref_header = mcf.makesimpleheader(hdr_co)
fits.writeto('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_reproject2co.fits',
	co,ref_header,overwrite=True)

#need to reproject the HI and CII into this frame and pixel scale
hi_repr = mcf.reproject2D(hi,mcf.makesimpleheader(hdr_hi),ref_header)
fits.writeto('../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor_mom0_reproject2co.fits',
	hi_repr,ref_header,overwrite=True)
cii_repr = mcf.reproject2D(cii,mcf.makesimpleheader(hdr_cii),ref_header)
fits.writeto('../Data/Disk_Map/Moments/M82_CII_map_mom0_maskSNR2_reproject2co.fits',
	cii_repr,ref_header,overwrite=True)

#this is really cool, but tbh it doesn't look as nice as the contour image I already have