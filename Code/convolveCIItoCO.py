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
#co_specsmooth_reproj = co_specsmooth.reproject(cii_conv.header)
cii_conv_reproj = cii_conv.reproject(co_specsmooth.header)

#write these matched cubes to a file
cii_conv_reproj.write('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits',overwrite=True)
co_specsmooth.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_matchedCIICO.fits',overwrite=True)

#make mom0 maps
mom0_cii = cii_conv_reproj.moment(order=0)
mom0_co = co_specsmooth.moment(order=0)

#write to file
mom0_cii.write('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits',overwrite=True)
mom0_co.write('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits',overwrite=True)