#based on notebooks at https://github.com/mpound/pdrtpy-nb/tree/master/notebooks

from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import numpy as np
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot

#let's start with just pixel 6 first
m6_cii = Measurement(data=23.3,uncertainty = StdDevUncertainty(0.3),identifier="CII_158",unit="K km/s",
	bmaj=14.1*u.arcsec,bmin=14.1*u.arcsec,bpa=0.*u.degree)
m6_co = Measurement(data=334.8,uncertainty = StdDevUncertainty(12.5),identifier="CO_10",unit="K km/s",
	bmaj=14.1*u.arcsec,bmin=14.1*u.arcsec,bpa=0.*u.degree)


#start with the Wolfire & Kaufman 2020 model at solar metallicity
ms = ModelSet('wk2020',z=1)
p = LineRatioFit(ms,[m6_cii,m6_co])
p.run()