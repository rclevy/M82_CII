#run all analysis steps for CII in M82

import os

os.system('python smoothCOcube.py') #This is VERY slow
os.system('python extractCOHIspectra.py')
os.system('python pltCIIspectra.py --plot_fit True')
os.system('python pltCIICOHIspectra.py')
os.system('python mkCIIMapMoments.py') #set some mask level probably
os.system('python pltCIImap.py "peak"') #set some mask level
os.system('python pltCIImap.py "intinten" --cmap gray_r --clim 0 800') #set some mask level
os.system('python pltCIImap.py "vcen_cgrad" --cmap RdYlBu_r --clim 0 350')
os.system('python pltCIImap.py "fwhm" --cmap viridis --clim 0 200')
os.system('python pltCompositeImage.py')