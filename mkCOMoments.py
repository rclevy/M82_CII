#make moment maps of 22" CO TP cube

from spectral_cube import SpectralCube
import astropy.units as u
from numpy import pi, log

fname_co = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed'
#fname_co = '../Data/Ancillary_Data/M82.CO.30m.map'


co_cube = SpectralCube.read(fname_co+'.fits').with_spectral_unit(u.km/u.s)
co_cube.allow_huge_operations = True

co_mom0 = co_cube.moment(order=0)
co_mom0.write(fname_co+'_mom0.fits',overwrite=True)

co_mom1 = co_cube.moment(order=1)
co_mom1.write(fname_co+'_mom1.fits',overwrite=True)

co_peak = co_cube.max(axis=0)
co_peak.write(fname_co+'_peak.fits',overwrite=True)
