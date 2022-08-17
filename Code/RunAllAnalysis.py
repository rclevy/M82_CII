#run all analysis steps for CII in M82

import os

os.system('python smoothCOcube.py') #This is VERY slow
os.system('python extractCOHIspectra.py')
os,system('python calcColumnDensity.py')
os.system('python fitCIIspectra.py')
os.system('python pltCIICOHIspectra.py')
os.system('python mkCIIMapMoments.py --mask_level 2')
os.system('python pltCIImap.py "max" --mask_level 2 --norm log --clim 0.04 8.')
os.system('python pltCIImap.py "mom0" --mask_level 2 --norm asinh --clim 0 1250')
os.system('python pltCIImap.py "mom1" --mask_level 2 --clim 85 335') #symmetric about Vsys=210
os.system('python pltCIImap.py "mom2" --mask_level 2 --clim 0 175')
os.system('python pltCompositeImage.py')
os.system('python convolvePACS2upGREAT.py')
os.system('python compareHerschel.py')
os.system('python calcCIIPropertiesDisk.py')
