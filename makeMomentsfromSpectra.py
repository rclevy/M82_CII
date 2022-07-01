import numpy as np

def make_moments_from_spectra(vel,spec,dv,nsig=2.,vel_all=np.nan,spec_all=np.nan):

	#
	if np.all(np.isnan(vel_all))==True:
		vel_all = vel.copy()
	if np.all(np.isnan(spec_all))==True:
		spec_all = spec.copy()

	#get max, mom0, mom1, mom2 from spectrum itself
	mmax = np.nanmax(spec)
	mom0 = dv*np.nansum(spec) #K km/s
	mom1 = np.nansum(vel*spec)/np.nansum(spec) #km/s
	mom2 = np.sqrt(np.nansum((vel-mom1)**2*spec)/np.nansum(spec)) #km/s

	idx_nosig = np.where((vel_all <= mom1-nsig*mom2/2.355) | (vel_all >= mom1+nsig*mom2/2.355))
	if np.isnan(mom2) == False:
		rms = np.sqrt(np.nanmean(spec_all[idx_nosig]**2)) #K
	else:
		rms = np.sqrt(np.nanmean(spec_all**2)) #K

	emmax = rms #K
	emom0 = dv*rms #K km/s
	emom1 = dv/mom0*np.sqrt(mom0**2+emom0**2+(mom1*rms)**2) #km/s
	emom2 = dv/(2*mom2*mom0)*np.sqrt((2*mom1*emom1*mom0/dv+np.nansum(vel-mom1)**2*rms)**2+(rms*mom2**2)**2)

	if np.isnan(mom2) == True:
		print('WARNING: All noise! Setting values to NaN. rms calculated over entire bandpass.')
		mmax = np.nan
		emmax = np.nan
		mom0 = np.nan
		emom0 = np.nan
		mom1 = np.nan
		emom1 = np.nan
		mom2 = np.nan
		emom2 = np.nan


	tab = {'Mom0': [mom0], 'eMom0': [emom0], 'Max': [mmax], 'eMax': [emmax], 'Mom1': [mom1], 'eMom1': [emom1], 'Mom2': [mom2], 'eMom2': [emom2], 'rms': [rms]}
	return tab


