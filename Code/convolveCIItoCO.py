#convolve the CII map in the disk to the CO spatial resolution and gridding
#spectrally smooth to 10 km/s

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel,Gaussian2DKernel,convolve
import astropy.units as u
from reproject import reproject_interp
from spectral_cube import SpectralCube
import radio_beam


#open the CO cube
co = SpectralCube.read('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed.fits')*u.K

#open the CII cube
cii = SpectralCube.read('../Data/Disk_Map/M82_CII_map_fullysampled.fits')

#first spectrally smooth the CO to 10 km/s (like the CII)
fwhm_factor = np.sqrt(8*np.log(2))
sf = np.abs(cii.header['CDELT3']/(co.header['CDELT3']/1E3))
kernel_spec = Gaussian1DKernel(sf/fwhm_factor)
velax_cii = cii.spectral_axis
co_specsmooth = co.spectral_smooth(kernel_spec).spectral_interpolate(velax_cii,suppress_smooth_warning=True).with_spectral_unit(u.km/u.s)

#now convolve to the CO beam size
beam_co = radio_beam.Beam(major=co.header['BMAJ']*u.deg,minor=co.header['BMIN']*u.deg,pa=co.header['BPA']*u.deg)
beam_cii = radio_beam.Beam(major=cii.header['BMAJ']*u.deg,minor=cii.header['BMIN']*u.deg,pa=cii.header['BPA']*u.deg)
cii_conv = cii.convolve_to(beam_co)

#finally reproject the CO cube onto the CII pixels (since those are larger)
#but we want the WCS orientation of the CO cube
co_hdr = co_specsmooth.header
cii_hdr = cii_conv.header
#in the new header, replace the pixel size and start info with the CII
new_hdr = co_hdr.copy()
new_hdr['NAXIS1'] = cii_hdr['NAXIS1']
new_hdr['NAXIS2'] = cii_hdr['NAXIS2']
new_hdr['CRPIX1'] = cii_hdr['CRPIX1']
new_hdr['CRPIX2'] = cii_hdr['CRPIX2']
new_hdr['CRVAL1'] = cii_hdr['CRVAL1']
new_hdr['CRVAL2'] = cii_hdr['CRVAL2']
new_hdr['CDELT1'] = cii_hdr['CDELT1']
new_hdr['CDELT2'] = cii_hdr['CDELT2']
co_specsmooth_reproj = co_specsmooth.reproject(new_hdr)
new_hdr['LINE']='CII'
new_hdr['RESTFRQ'] = cii_hdr['RESTFRQ']
cii_conv_reproj = cii_conv.reproject(new_hdr)

#write these matched cubes to a file
cii_conv_reproj.write('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits',overwrite=True)
co_specsmooth_reproj.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_matchedCIICO.fits',overwrite=True)

#make mom0 maps
max_cii = cii_conv_reproj.max(axis=0)
max_co = co_specsmooth_reproj.max(axis=0)
mom0_cii = cii_conv_reproj.moment(order=0)
mom0_co = co_specsmooth_reproj.moment(order=0)
mom1_cii = cii_conv_reproj.moment(order=1)
mom1_co = co_specsmooth_reproj.moment(order=1)

#write to file
max_cii.write('../Data/Disk_Map/Moments/M82_CII_map_max_matchedCIICO.fits',overwrite=True)
max_co.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_max_matchedCIICO.fits',overwrite=True)
mom0_cii.write('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits',overwrite=True)
mom0_co.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits',overwrite=True)
mom1_cii.write('../Data/Disk_Map/Moments/M82_CII_map_mom1_matchedCIICO.fits',overwrite=True)
mom1_co.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom1_matchedCIICO.fits',overwrite=True)


