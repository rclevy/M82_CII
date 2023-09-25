#Fit CII spectra with Moments and Gaussians

import numpy as np
from scipy.optimize import curve_fit
import upGREATUtils
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from makeMomentsfromSpectra import make_moments_from_spectra

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
	dfs = []
	RAs = []
	Decs = []

	if i==0:
		M_cii = pd.read_csv('../Data/Outflow_Pointings/CII/outflow1_Mass_ColumnDensities.csv')['Mass_Cp_Msun'].values
		eM_cii = pd.read_csv('../Data/Outflow_Pointings/CII/outflow1_Mass_ColumnDensities.csv')['e_Mass_Cp_Msun_upper'].values
		pix_ord = pd.read_csv('../Data/Outflow_Pointings/CII/outflow1_Mass_ColumnDensities.csv')['Pixel Number'].values.tolist()
	else:
		M_cii = np.nan*np.ones((n_pixels,))
		eM_cii = np.nan*np.ones((n_pixels,))
		pix_ord = np.arange(0,n_pixels,1).tolist()
	
	for j in range(n_pixels):
		#load spectrum
		this_spec,this_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(j)+'.fits')
		Vclip_red,Vclip_blue = upGREATUtils.get_bad_edge_chans(i,j)
		this_vel,this_spec = upGREATUtils.rm_bad_edge_chans(this_vel,this_spec,which_end='both',Vred=Vclip_red,Vblue=Vclip_blue)
		chan_width = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s
		_,_,pix_center = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)
		RAs.append(pix_center.ra.to_string(u.hour,precision=1,format='latex'))
		Decs.append(pix_center.dec.to_string(u.degree,precision=1,alwayssign=True,format='latex'))

		#mask to -100-400 km/s
		this_vel_og = this_vel.copy()
		this_spec_og = this_spec.copy()
		vmin = -100. #km/s
		vmax = 400. #km/s
		this_spec[this_vel > vmax] = np.nan
		this_spec[this_vel < vmin] = np.nan


		#find rms outside velocity range
		rms = np.sqrt(np.nanmean(this_spec_og[(this_vel_og < vmin) | (this_vel_og > vmax)]**2))


		#remove NaNs
		idx = np.where(np.isnan(this_spec)==False)
		this_vel = this_vel[idx]
		this_spec = this_spec[idx]

		#get the C+ mass for this pixel
		idx = pix_ord.index(j)
		this_mass = M_cii[idx]
		this_emass = eM_cii[idx]
		
		#calc moments of the spectra
		dv = this_vel[0]-this_vel[1] #km/s
		tab = make_moments_from_spectra(this_vel,this_spec,dv,nsig=2.,vmin=vmin,vmax=vmax)
		tab['rms'] = [rms] #replace with fixed velocity window
		tab['Mass'] = [this_mass]
		tab['eMass'] = [this_emass]


		#open the HI and CO mom0
		hi_moms = pd.read_csv('../Data/Outflow_Pointings/HI_Moments/outflow'+str(i+1)+'_pix'+str(j)+'_momentparams.csv')
		tab['Mom0_HI'] = [hi_moms['Mom0'].values[0]]
		tab['eMom0_HI'] = [hi_moms['eMom0'].values[0]]
		co_moms = pd.read_csv('../Data/Outflow_Pointings/CO_TP_Moments/outflow'+str(i+1)+'_pix'+str(j)+'_momentparams.csv')
		tab['Mom0_CO'] = [co_moms['Mom0'].values[0]]
		tab['eMom0_CO'] = [co_moms['eMom0'].values[0]]

		this_df = pd.DataFrame.from_dict(tab)
		this_df.to_csv('../Data/Outflow_Pointings/CII_Moments/outflow'+str(i+1)+'_pix'+str(j)+'_momentparams.csv',index=False)
		
		dfs.append(this_df)

		


	#concat all the pixels into one table
	df = pd.concat(dfs)
	df.insert(0,'Pixel Number',range(n_pixels),True)
	df.set_index('Pixel Number',drop=False,inplace=True)

	#remove low SNR values
	idx = np.where((df['Mom2']/df['eMom2']).values < 0.1)[0].tolist()
	for bad_el in idx:
		df.at[bad_el,'Max'] = np.nan
		df.at[bad_el,'eMax'] = np.nan
		df.at[bad_el,'Mom0'] = np.nan
		df.at[bad_el,'eMom0'] = np.nan
		df.at[bad_el,'Mom1'] = np.nan
		df.at[bad_el,'eMom1'] = np.nan
		df.at[bad_el,'Mom2'] = np.nan
		df.at[bad_el,'eMom2'] = np.nan
		df.at[bad_el,'Mass'] = np.nan

	#get the CII/HI and CII/CO ratios
	df = df.assign(ratio_Mom0_HI_CII=df['Mom0_HI']/df['Mom0'])
	df = df.assign(eratio_Mom0_HI_CII=df['ratio_Mom0_HI_CII']*np.sqrt((df['eMom0']/df['Mom0'])**2+(df['eMom0_HI']/df['Mom0_HI'])**2))
	df = df.assign(ratio_Mom0_CO_CII=df['Mom0_CO']/df['Mom0'])
	df = df.assign(eratio_Mom0_CO_CII=df['ratio_Mom0_CO_CII']*np.sqrt((df['eMom0']/df['Mom0'])**2+(df['eMom0_CO']/df['Mom0_CO'])**2))


	df.to_csv('../Data/Outflow_Pointings/CII_Moments/outflow'+str(i+1)+'_momentparams.csv',index=False)

	#write this table to a tex file
	tex_tab_file = '../Data/Outflow_Pointings/CII_Moments/outflow'+str(i+1)+'_momentparams.tex'
	with open(tex_tab_file,'w') as fp:
		fp.write('\\begin{deluxetable*}{cccccccccc}\n')
		fp.write('\\tablecaption{Properties of \\CII\\ spectra in the outflow \\label{tab:outflowfits}}\n')
		fp.write('\\tablehead{Pixel Number & R.A. & Decl. & ${\\rm I_{peak}}$ & ${\\rm I_{int}}$ & ${\\rm V_0}$ & $\\sigma_{\\rm V}$ & rms & ${\\rm log_{10}M_{C^{+}}}$\\\\') 
		fp.write('& (J2000 hours) & (J2000 degrees) & (mK) & (K~km~s$^{-1}$) & (km~s$^{-1}$) & (km~s$^{-1}$) & (mK) & (${\\rm log_{10}M_\\odot}$)}')
		fp.write('\\startdata\n')

		for j in range(n_pixels):
			#convert to strings
			if j in idx:
				pk_str = '---'
				mom0_str = '---'
				mom1_str = '---'
				mom2_str = '---'
				mass_str = '---'
			else:
				pk_str = str(np.round(df['Max'].values[j]*1E3,1))+' $\\pm$ '+str(np.round(df['eMax'].values[j]*1E3,1))
				mom0_str = str(np.round(df['Mom0'].values[j],1))+' $\\pm$ '+str(np.round(df['eMom0'].values[j],1))
				mom1_str = str(np.round(df['Mom1'].values[j],1))+' $\\pm$ '+str(np.round(df['eMom1'].values[j],1))
				mom2_str = str(np.round(df['Mom2'].values[j],1))+' $\\pm$ '+str(np.round(df['eMom2'].values[j],1))
				mass_str = '%.1f $\\pm %.1f$' %(np.log10(df['Mass'].values[j]),df['eMass'].values[j]/(df['Mass'].values[j]*np.log(10)))

			fp.write('%i & %s & %s & %s & %s & %s & %s & %.1f & %s\\\\\n' 
				%(df['Pixel Number'].values[j],RAs[j],Decs[j],pk_str,mom0_str,mom1_str,mom2_str,df['rms'].values[j]*1E3,mass_str))


		fp.write('\\enddata\n')
		fp.write('\\tablecomments{The R.A. and Decl. of the center of each pixel are given. The other columns show the moments of the spectra including the peak intensity (${\\rm I_{peak}}$), the integrated intensity (moment 0; ${\\rm I_{int}}$), the mean velocity (moment 1; ${\\rm V_0}$), the linewidth (moment 2, $\\sigma_{\\rm V}$), and corresponding uncertainties. The moments are calculated over a fixed velocity range spanning -$100-400$~\\kms. rms is the root-mean-square noise of the spectrum calculated outside of the velocity range used for the moments. If the moments cannot be calculated because the line is not detected, then the rms is calculated over the entire bandpass. All temperatures refer to T$_{\\rm mb}$. ${\\rm M_{C^{+}}}$ is the mass in C$^+$ inferred from the spectrum considering only collisions with the atomic gas; see Appendix \\ref{app:density_calc} for details of this calculation.} \n\n')
		fp.write('\\end{deluxetable*}\n')

	#write this table to a tex file
	tex_tab_file = '../Data/Outflow_Pointings/CII_Moments/outflow'+str(i+1)+'_momentparams_COHI.tex'
	with open(tex_tab_file,'w') as fp:
		fp.write('\\begin{deluxetable}{ccc}\n')
		fp.write('\\tablecaption{Ratios of the \\CII, \\HI, and CO integrated intensities in the outflow.\\label{tab:outflowfits_HICO}}\n')
		# fp.write('\\tablehead{Pixel Number & ${\\rm log_{10}(I_{int,HI}/I_{int,[CII]})}$ & ${\\rm log_{10}(I_{int,CO}/I_{int,[CII]})}$}')
		fp.write('\\tablehead{Pixel Number & ${\\rm I_{int,HI}/I_{int,[CII]}}$ & ${\\rm I_{int,CO}/I_{int,[CII]}}$}')
		fp.write('\\startdata\n')

		for j in range(n_pixels):
			#convert to strings
			if j in idx:
				ratio_hi_str = '---'
				ratio_co_str = '---'
			else:
				# ratio_hi_str = '%.2f $\\pm %.2f$' %(np.log10(df['ratio_Mom0_HI_CII'].values[j]),df['eratio_Mom0_HI_CII'].values[j]/(df['ratio_Mom0_HI_CII'].values[j]*np.log(10)))
				# ratio_co_str = '%.2f $\\pm %.2f$' %(np.log10(df['ratio_Mom0_CO_CII'].values[j]),df['eratio_Mom0_CO_CII'].values[j]/(df['ratio_Mom0_CO_CII'].values[j]*np.log(10)))
				ratio_hi_str = '%.1f $\\pm$ %.1f' %(df['ratio_Mom0_HI_CII'].values[j],df['eratio_Mom0_HI_CII'].values[j])
				ratio_co_str = '%.1f $\\pm$ %.1f' %(df['ratio_Mom0_CO_CII'].values[j],df['eratio_Mom0_CO_CII'].values[j])

			fp.write('%i & %s & %s\\\\\n' 
				%(df['Pixel Number'].values[j],ratio_hi_str,ratio_co_str))


		fp.write('\\enddata\n')
		fp.write('\\tablecomments{The integrated intensity ratios are in K~km~s$^{-1}$ units.} \n\n')
		fp.write('\\end{deluxetable}\n')







		#### THIS IS CALCULATED BUT DEPRECATED IN THE ANALYSIS ####
	for j in range(n_pixels):
		#now fit with Gaussian
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
		#grab spectrum > 2 sigma from peak
		vrng_2sig = [popt[1]-2*popt[2],popt[1]+2*popt[2]]
		rms = np.sqrt(np.nanmean(this_spec[(this_vel < vrng_2sig[0]) | (this_vel > vrng_2sig[1])]**2))

		#compute reduced X^2
		chi2r = np.nansum((this_spec-gauss(this_vel,*popt))**2/rms)/(len(this_spec)-len(popt))
		chi2r2 = np.nansum((this_spec-gauss2(this_vel,*popt2))**2/rms)/(len(this_spec)-len(popt2))

		#calc integrated intensity and uncert
		intinten = popt[0]*popt[2]*np.sqrt(np.pi)
		eintinten = np.sqrt((epopt[0]*popt[2]*np.sqrt(np.pi))**2+(popt[0]*epopt[2]*np.sqrt(np.pi))**2)


		#convert the peak intensity from Tmb (K) to Jy
		#https://www-sofia.atlassian.net/wiki/spaces/OHFC8/pages/1703955/6.+GREAT#6.1.2.1---Imaging-Sensitivities
		eta_mb = 0.67
		eta_fss = 0.97
		Tr = popt[0]*eta_mb
		eTr = epopt[0]*eta_mb
		Ta = Tr*eta_fss
		eTa = eTr*eta_fss
		S = 971.*Ta #Jy
		eS = 971.*eTa #Jy

		#save fits to file
		tab = {'IntInten': intinten, 'eIntInten': eintinten, 'Peak': [popt[0]], 'ePeak': [epopt[0]], 'Peak_Jy': [S], 'ePeak_Jy': [eS], 'Vcen': [popt[1]], 'eVcen': [epopt[1]], 'FWHM': [popt[2]*2.355], 'eFWHM': [epopt[2]*2.355], 'Chi2r': [chi2r], 'rms': [rms]}
#   tab2 = {'Peak': [popt2[0],popt2[3]], 'ePeak': [epopt2[0],epopt2[3]], 'Vcen': [popt2[1],popt2[4]], 'eVcen': [epopt2[1],epopt2[4]], 'FWHM': [popt2[2]*2.355,popt2[5]*2.355], 'eFWHM': [epopt2[2]*2.355,epopt2[5]*2.355], 'Chi2r': [chi2r2,chi2r2]}

		df = pd.DataFrame.from_dict(tab)
#   df2 = pd.DataFrame.from_dict(tab2)

		df.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_pix'+str(j)+'_gaussianfitparams.csv',index=False)
#   df2.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_pix'+str(j)+'_gaussian2fitparams.csv',index=False)

		frames.append(df)

	#concat all the pixels into one table
	df = pd.concat(frames)
	df.insert(0,'Pixel Number',range(n_pixels),True)
	df.to_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_gaussianfitparams.csv',index=False)

	#write this table to a tex file
	tex_tab_file = '../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_gaussianfitparams.tex'
	with open(tex_tab_file,'w') as fp:
		fp.write('\\begin{deluxetable*}{cccccccccc}\n')
		fp.write('\\tablecaption{Properties of \\CII\\ spectra in the outflow \\label{tab:outflowfits}}\n')
		fp.write('\\tablehead{Pixel Number & R.A. & Decl. & \\multicolumn{2}{c}{${\\rm I_{peak}}$} & ${\\rm V_0}$ & $\\sigma_{\\rm V, FWHM}$ & ${\\rm I_{int}}$ & rms \\\\') 
		fp.write('& (J2000 hours) & (J2000 degrees) & (K) & (Jy) & (km~s$^{-1}$) & (km~s$^{-1}$) & (K~km~s$^{-1}$) & (K)}')
		fp.write('\\startdata\n')

		for j in range(n_pixels):
			this_epeak = df['ePeak'].values[j]
			this_epeak_Jy = df['ePeak_Jy'].values[j]
			if np.round(this_epeak,2) <= 0.01:
				this_epeak = 0.01
				eTr = this_epeak*eta_mb
				eTa = eTr*eta_fss
				this_epeak_Jy = 971.*eTa #Jy

			fp.write('%i & %s & %s & %.2f $\\pm$ %.2f & %1.f $\\pm$ %1.f & %1.f $\\pm$ %1.f & %1.f $\\pm$ %1.f & %1.f $\\pm$ %1.f & %.2f \\\\\n' 
				%(df['Pixel Number'].values[j],RAs[j],Decs[j],df['Peak'].values[j],this_epeak,df['Peak_Jy'].values[j],this_epeak_Jy,df['Vcen'].values[j],df['eVcen'].values[j],df['FWHM'].values[j],df['eFWHM'].values[j],df['IntInten'].values[j],df['eIntInten'].values[j],df['rms'].values[j]))


		fp.write('\\enddata\n')
		fp.write('\\tablecomments{The R.A. and Decl. of the center of each pixel are given. The other columns show the results of the Gaussian fits to the spectra including the peak intensity (${\\rm I_{peak}}$), the velocity centroid (${\\rm V_0}$), the FWHM velocity dispersion ($\\sigma_{\\rm V, FWHM}$), and corresponding uncertainties. ${\\rm I_{int}}$ is the integrated intensity calculated from the Gaussian fit (and the propagated uncertainty). rms is the root-mean-square noise of the spectrum calculated 2-$\\sigma$ from the \\CII\\ peak. All temperatures in K refer to T$_{\\rm mb}$. The conversion from K (T$_{\\rm mb}$) to Jy was done following Section 6.1.2.1 of the SOFIA Observer\'s Handbook for Cycle 8\\footnote{\\url{https://www-sofia.atlassian.net/wiki/spaces/OHFC8}}.} \n\n')
		fp.write('\\end{deluxetable*}\n')

