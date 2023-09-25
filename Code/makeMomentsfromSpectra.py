import numpy as np

def make_moments_from_spectra(vel,spec,dv,vmin=np.nan,vmax=np.nan,nsig=2):

	#calculate moments over specified velocity range
	if (np.isnan(vmax) == False) & (np.isnan(vmin)==False):
		idx = np.where((vel >= vmin) & (vel <= vmax))[0]
		vel = vel[idx]
		spec = spec[idx]

	#get max, mom0, mom1, mom2
	mmax = np.nanmax(spec)
	N = np.count_nonzero(~np.isnan(spec))
	mom0 = dv*np.nansum(spec) #K km/s
	mom1 = np.nansum(vel*spec)/np.nansum(spec) #km/s
	mom2 = np.sqrt(np.abs(np.nansum((vel-mom1)**2*spec)/np.nansum(spec))) #km/s


	idx_nosig = np.where((vel <= mom1-nsig*mom2/2.355) | (vel >= mom1+nsig*mom2/2.355))
	if np.isnan(mom2) == False:
		rms = np.sqrt(np.nanmean(spec[idx_nosig]**2)) #K
	else:
		rms = np.sqrt(np.nanmean(spec**2)) #K


	#get propagated uncertainties
	emmax = rms #K
	emom0 = dv*rms*np.sqrt(N) #K km/s
	emom1 = dv/mom0*np.sqrt(mom0**2+emom0**2+(mom1*rms)**2) #km/s
	emom2 = dv/(2*mom2*mom0)*np.sqrt((2*mom1*emom1*mom0/dv+np.nansum(vel-mom1)**2*rms)**2+(rms*mom2**2)**2)


	tab = {'Mom0': [mom0], 'eMom0': [emom0], 'Max': [mmax], 'eMax': [emmax], 'Mom1': [mom1], 'eMom1': [emom1], 'Mom2': [mom2], 'eMom2': [emom2], 'rms': [rms]}
	return tab


