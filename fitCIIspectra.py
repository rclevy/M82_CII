#Fit CII spectra with Gaussians

import numpy as np
from scipy.optimize import curve_fit
import upGREATUtils
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

def gauss(x,a,b,c):
	return a*np.exp((-(x-b)**2/(2*c**2)))

def gauss2(x,a1,b1,c1,a2,b2,c2):
	return a1*np.exp((-(x-b1)**2/(2*c1**2)))+a2*np.exp((-(x-b2)**2/(2*c2**2)))


gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
distance = 3.63*u.Mpc
 
n_pointings = 2
n_pixels = 7

for i in range(n_pointings):
	frames = []
	RAs = []
	Decs = []
	
	for j in range(n_pixels):
		#load spectrum
		this_spec,this_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(j)+'.fits')
		chan_width = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s
		_,_,pix_center = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)
		RAs.append(pix_center.ra.to_string(u.hour,format='latex'))
		Decs.append(pix_center.dec.to_string(u.degree, alwayssign=True,format='latex'))

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

		frames.append(df)

	#concat all the pixels into one table
	df = pd.concat(frames)
	df.insert(0,'Pixel Number',range(n_pixels),True)
	df.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_gaussianfitparams.csv',index=False)

	#write this table to a tex file
	tex_tab_file = '../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_gaussianfitparams.tex'
	with open(tex_tab_file,'w') as fp:
		fp.write('\\begin{deluxetable*}{cccccccc}\n')
		fp.write('\\tablecaption{Properties of \\CII\\ spectra in the outflow \\label{tab:outflowfits}}\n')
		fp.write('\\tablehead{Pixel Number & R.A. (J2000 hours) & Decl. (J2000 degrees) & ${\\rm I_{peak}}$ (K) & ${\\rm V_0}$ (km~s$^{-1}$) & $\\sigma_{\\rm V, FWHM}$ (km~s$^{-1}$) & ${\\rm I_{int}}$ (K~km~s$^{-1}$)}') 
		fp.write('\\startdata\n')

		for j in range(n_pixels):
			this_epeak = df['ePeak'].values[j]
			if np.round(this_epeak,2) <= 0.01:
				this_epeak = 0.01

			fp.write('%i & %s & %s & %.2f $\\pm$ %.2f & %1.f $\\pm$ %1.f & %1.f $\\pm$ %1.f & %1.f $\\pm$ %1.f \\\\\n' 
				%(df['Pixel Number'].values[j],RAs[j],Decs[j],df['Peak'].values[j],this_epeak,df['Vcen'].values[j],df['eVcen'].values[j],df['FWHM'].values[j],df['eFWHM'].values[j],df['IntInten'].values[j],df['eIntInten'].values[j]))


		fp.write('\\enddata\n')
		fp.write('\\tablecomments{The R.A. and Decl. of the center of each pixel are given. The other columns show the results of the Gaussian fits to the spectra including the peak intensity (${\\rm I_{peak}}$), the velocity centroid (${\\rm V_0}$), the FWHM velocity dispersion ($\\sigma_{\\rm V, FWHM}$), and corresponding uncertainties. ${\\rm I_{int}}$ is the integrated intensity calculated from the Gaussian fit (and the propagated uncertainty).}\n\n')
		fp.write('\\end{deluxetable*}\n')






