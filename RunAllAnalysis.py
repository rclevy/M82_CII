#run all analysis steps for CII in M82

import os

os.system('python extractCOHIspectra.py')
os.system('python pltCIIspectra.py')
os.system('python mkCIIMapMoments.py') #set some mask level probably
os.system('python pltCIImap.py "peak" --norm asinh') #set some mask level
os.system('python pltCIImap.py "mom0" --norm asinh --cmap gray_r --clim 0 1000') #set some mask level
os.system('python pltCIImap.py "vcen_cgrad" --cmap RdYlBu_r --clim 0 350')
os.system('python pltCIImap.py "fwhm" --cmap viridis --clim 0 200')
os.system('python pltCompositeImage.py')