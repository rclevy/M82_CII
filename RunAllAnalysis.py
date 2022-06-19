#run all analysis steps for CII in M82

import os

os.system('python smoothCOcube.py') #This is VERY slow
os.system('python extractCOHIspectra.py')
os.system('python pltCIIspectra.py --plot_fit True')
os.system('python pltCIICOHIspectra.py')
os.system('python mkCIIMapMoments.py --mask_level 5')
os.system('python pltCIImap.py "peak" --mask_level 5 --norm log --clim 0 9.7')
os.system('python pltCIImap.py "intinten" --mask_level 5 --norm asinh --clim 0 800')
os.system('python pltCIImap.py "vcen_cgrad" --mask_level 5 --clim 0 350')
os.system('python pltCIImap.py "fwhm" --mask_level 5 --clim 0 200')
os.system('python pltCompositeImage.py')