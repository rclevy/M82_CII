#make moment maps of 17" HI cube

from spectral_cube import SpectralCube
import astropy.units as u
from numpy import pi, log

fname_hi = '../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor'

hi_cube = SpectralCube.read(fname_hi+'.fits').with_spectral_unit(u.km/u.s)
hi_cube.allow_huge_operations = True

hi_beam_area = pi*hi_cube.header['BMAJ']*hi_cube.header['BMIN']*u.deg**2/(4*log(2))
equiv = u.brightness_temperature(hi_cube.header['RESTFRQ']*u.Hz,beam_area=hi_beam_area)

hi_cube_K = hi_cube.to(u.K, equivalencies=equiv)
hi_cube_K.write(fname_hi+'_K.fits',overwrite=True)

#mask out the regions with absorption
#from Martini+2018 Sect 2.3
hi_mask_lev = 7E19/1.823E18/10*u.K #K 
print('Masking out I_HI < %.2f K' %hi_mask_lev.value)
mask = (hi_cube_K > hi_mask_lev)
hi_cube_masked = hi_cube_K.with_mask(mask)

hi_mom0 = hi_cube_masked.moment(order=0)
hi_mom0.write(fname_hi+'_mom0.fits',overwrite=True)

hi_mom1 = hi_cube_masked.moment(order=1)
hi_mom1.write(fname_hi+'_mom1.fits',overwrite=True)
