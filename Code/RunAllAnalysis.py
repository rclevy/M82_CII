#run all analysis steps for CII in M82

import os

os.system('python smoothCOcube.py') #This is VERY slow
os.system('python extractCOHIspectra.py')
os,system('python calcColumnDensity.py')
os.system('python fitCIIspectra.py')
os.system('python pltCIICOHIspectra.py')
os.system('python mkCIIMapMoments.py --mask_level 2')
os.system('python pltCIImap.py "max" --mask_level 2 --norm asinh --clim 0.07 8.')
os.system('python pltCIImap.py "mom0" --mask_level 2 --norm asinh --clim 0 1250')
os.system('python pltCIImap.py "mom1" --mask_level 2 --clim 85 335') #symmetric about Vsys=210
os.system('python pltCIImap.py "mom2" --mask_level 2 --clim 0 175')
os.system('python pltCompositeImage.py')
os.system('python compareLitFluxes.py')
os.system('python calcCIIPropertiesDisk.py')
os.system('python convolveCIItoCO.py')
os.system('python pltCIICOspectradisk.py')
os.system('python calcRadiationField.py')
os.system('python writeTableTracerRatios.py')
