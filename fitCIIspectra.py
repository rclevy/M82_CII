#Fit CII spectra with Gaussians

import numpy as np
from scipy.optimize import curve_fit
import upGREATUtils
import pandas as pd

def gauss(x,a,b,c):
	return a*np.exp((-(x-b)**2/(2*c**2)))

def gauss2(x,a1,b1,c1,a2,b2,c2):
	return a1*np.exp((-(x-b1)**2/(2*c1**2)))+a2*np.exp((-(x-b2)**2/(2*c2**2)))


n_pointings = 2
n_pixels = 7

for i in range(n_pointings):
	for j in range(n_pixels):
		#load spectrum
		this_spec,this_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(j)+'.fits')
		chan_width = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s

		#remove NaNs
		idx = np.where(np.isnan(this_spec)==False)
		this_vel = this_vel[idx]
		this_spec = this_spec[idx]

		#define fitting bounds
		bounds_lower = [0., np.nanmin(this_vel), 2*chan_width]
		bounds_upper = [np.inf, np.nanmax(this_vel), 200.]
		bounds = (bounds_lower, bounds_upper)

		bounds_lower = [0., np.nanmin(this_vel), 2*chan_width, 0., np.nanmin(this_vel), 2*chan_width]
		bounds_upper = [np.inf, np.nanmax(this_vel), 200., np.inf, np.nanmax(this_vel), 200.]
		bounds2 = (bounds_lower, bounds_upper)


		#define starting guess
		p_init = [np.nanmax(this_spec),this_vel[np.nanargmax(this_spec)], 100.]
		p_init2 = [np.nanmax(this_spec),this_vel[np.nanargmax(this_spec)], 100., np.nanmax(this_spec),this_vel[np.nanargmax(this_spec)], 100.]

		#fit with a Gaussian
		popt,pcov = curve_fit(gauss,this_vel,this_spec,p0=p_init,bounds=bounds)
		epopt = np.sqrt(np.diag(pcov))
		popt2,pcov2 = curve_fit(gauss2,this_vel,this_spec,p0=p_init2,bounds=bounds2)
		epopt2 = np.sqrt(np.diag(pcov2))

		#find the RMS of the spectrum
		#grab spectrum > 3 sigma from peak
		vrng_2sig = [popt[1]-3*popt[2],popt[1]+3*popt[2]]
		rms = np.sqrt(np.nanmean(this_spec[(this_vel < vrng_2sig[0]) | (this_vel > vrng_2sig[1])]**2))

		#compute reduced X^2
		chi2r = np.nansum((this_spec-gauss(this_vel,*popt))**2/rms)/(len(this_spec)-len(popt))
		chi2r2 = np.nansum((this_spec-gauss2(this_vel,*popt2))**2/rms)/(len(this_spec)-len(popt2))

		#calc integrated intensity and uncert
		intinten = popt[0]*popt[2]*np.sqrt(np.pi)
		eintinten = np.sqrt((epopt[0]*popt[2]*np.sqrt(np.pi))**2+(popt[0]*epopt[2]*np.sqrt(np.pi))**2)

		#save fits to file
		tab = {'IntInten': intinten, 'eIntInten': eintinten, 'Peak': [popt[0]], 'ePeak': [epopt[0]], 'Vcen': [popt[1]], 'eVcen': [epopt[1]], 'FWHM': [popt[2]*2.355], 'eFWHM': [epopt[2]*2.355], 'Chi2r': [chi2r]}
#		tab2 = {'Peak': [popt2[0],popt2[3]], 'ePeak': [epopt2[0],epopt2[3]], 'Vcen': [popt2[1],popt2[4]], 'eVcen': [epopt2[1],epopt2[4]], 'FWHM': [popt2[2]*2.355,popt2[5]*2.355], 'eFWHM': [epopt2[2]*2.355,epopt2[5]*2.355], 'Chi2r': [chi2r2,chi2r2]}

		df = pd.DataFrame.from_dict(tab)
#		df2 = pd.DataFrame.from_dict(tab2)

		df.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_pix'+str(j)+'_gaussianfitparams.csv',index=False)
#		df2.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_pix'+str(j)+'_gaussian2fitparams.csv',index=False)







