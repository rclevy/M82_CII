#run all analysis steps for CII in M82

import os

os.system('python smoothCOcube.py') #This is VERY slow
os.system('python extractCOHIspectra.py')
os.system('python pltCIIspectra.py --plot_fit True')
os.system('python pltCIICOHIspectra.py')
os.system('python mkCIIMapMoments.py --mask_level 2')
os.system('python pltCIImap.py "max" --mask_level 2 --norm log --clim 0 9.7')
os.system('python pltCIImap.py "mom0" --mask_level 2 --norm asinh --clim 0 800')
os.system('python pltCIImap.py "mom1" --mask_level 2 --clim 0 350')
os.system('python pltCIImap.py "mom2" --mask_level 2 --clim 0 200')
os.system('python pltCompositeImage.py')