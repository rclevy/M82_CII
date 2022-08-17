from spectral_cube import SpectralCube
import radio_beam
import astropy.units as u
from astropy.convolution import Gaussian1DKernel
import numpy as np

fname_co = '../Data/Ancillary_Data/M82-NOEMA-30m-mask-merged.clean'
co_cube = SpectralCube.read(fname_co+'.fits')
cii_beam = radio_beam.Beam(major=14.1*u.arcsec,minor=14.1*u.arcsec,pa=0*u.deg)
co_cube.allow_huge_operations=True
co_cube_smo = co_cube.convolve_to(cii_beam)
co_cube_smo.write(fname_co+'_smoothedCII.fits',overwrite=True)

# #now smooth in velocity
# fwhm_factor = np.sqrt(8*np.log(2))
# current_resolution = co_cube.header['CDELT3']/1E3 * u.km/u.s
# target_resolution = 10.* u.km/u.s
# pixel_scale = current_resolution
# gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 /
#                   pixel_scale / fwhm_factor)
# kernel = Gaussian1DKernel(gaussian_width.value)
# co_cube_smo_vel = co_cube_smo.spectral_smooth(kernel)
# co_cube_smo_vel.write(fname_co+'_smoothedCII.fits',overwrite=True)