#make moment maps of CII in disk
#do this both as traditional moments and by fitting Gaussians to each pixel

import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import medfilt2d
from scipy.ndimage import convolve

def make_moments(cube,filepath,outsuff):
	mom0 = cube.moment(order=0)
	mom1 = cube.moment(order=1)
	mom2 = cube.linewidth_fwhm()
	mom0.write(filepath+'_mom0_'+outsuff+'.fits',overwrite=True)
	mom1.write(filepath+'_mom1_'+outsuff+'.fits',overwrite=True)
	mom2.write(filepath+'_mom2_'+outsuff+'.fits',overwrite=True)
	return mom0,mom1,mom2

def save_gaussfits(data,hdr,filepath,maptype,outsuff):
	hdu = fits.PrimaryHDU(data=data,header=hdr)
	hdul = fits.HDUList(hdu)
	hdul.writeto(filepath+'_'+maptype+'_'+outsuff+'.fits',overwrite=True)
	return

filepath = '../Data/Disk_Map/M82_CII_map'

#load the cube
cube = SpectralCube.read(filepath+'_cube.fits').with_spectral_unit(u.km/u.s)*u.K

#make traditional moment maps with no masking
make_moments(cube,filepath,'nomask')

#now mask based on the SNR
rms = np.sqrt(np.nanmean(cube)**2)*u.K
level = 1.
outsuff = 'maskSNR'+str(int(level))
cube_masked = cube.with_mask(cube > level*rms)
m0m,_,_=make_moments(cube_masked,filepath,outsuff)




#now make "moments" by fitting a Gaussian to each spectrum
def gauss(x,a,b,c):
	return a*np.exp(-(x-b)**2/(2*c**2))

peak = np.nan*np.ones((cube.shape[1],cube.shape[2]))
epeak = peak.copy()
vel = peak.copy()
evel = peak.copy()
fwhm = peak.copy()
efwhm = peak.copy()

#based on mkvelmap.m
iy,ix = np.where(np.isnan(m0m.value)==False)
v = cube_masked.spectral_axis.value #km/s
v_span = np.nanmax(v)-np.nanmin(v)
dv = np.abs(cube_masked.header['CDELT3']) #km/s

for i in range(len(ix)):
	s = cube_masked[:,iy[i],ix[i]].value
	s[np.isnan(s)==True]=0.
	try:
		p,pcov = curve_fit(gauss,v,s,
			p0=(np.nanmax(s),np.mean([np.nanmax(v),np.nanmin(v)]),5*dv),
			bounds=([0,np.nanmin(v),dv],[np.inf,np.nanmax(v),v_span/2]))
		ep = np.sqrt(np.diag(pcov))
	except RuntimeError:
		p = np.array([np.nan, np.nan, np.nan])
		ep = np.array([np.nan, np.nan, np.nan])


	peak[iy[i],ix[i]] = p[0]
	vel[iy[i],ix[i]] = p[1]
	fwhm[iy[i],ix[i]] = p[2]*2.355
	epeak[iy[i],ix[i]] = ep[0]
	evel[iy[i],ix[i]] = ep[1]
	efwhm[iy[i],ix[i]] = ep[2]*2.355


hdr = m0m.header
hdr['BUNIT']='K'
save_gaussfits(peak,hdr,filepath,'peak',outsuff)
save_gaussfits(epeak,hdr,filepath,'epeak',outsuff)
hdr['BUNIT']='km/s'
save_gaussfits(vel,hdr,filepath,'vcen',outsuff)
save_gaussfits(evel,hdr,filepath,'evcen',outsuff)
save_gaussfits(fwhm,hdr,filepath,'fwhm',outsuff)
save_gaussfits(efwhm,hdr,filepath,'efwhm',outsuff)

#now do some masking of the vcen field
maxdev = 40. #km/s
fm5 = medfilt2d(vel,[5,5])
fm3 = medfilt2d(vel,[3,3])
um = np.ones(vel.shape)
um[np.isnan(vel)==True]=0.
pc5 = convolve(um,np.ones((5,5)),mode='nearest') # count how many neighbors a pixel has that are not NaN
ix3 = np.where((pc5<25) & (pc5>=9) & (np.abs(fm3-vel)>maxdev))
ix5 = np.where((pc5==25) & (np.abs(fm5-vel)>maxdev))
vel[ix3] = fm3[ix3] # filter out things sticking out in the unsharp masking
vel[ix5] = fm5[ix5] # filter out things sticking out in the unsharp masking
#print('Replaced %d for 3x3 median and %d for 5x5 median. %.2f%% of pixels replaced.\n'
#	%(len(ix3[0]),len(ix5[0]),(len(ix3[0])+len(ix5[0]))/len(vel[0])))

#replace single NaN pixels with the median of its neighbors
nneigh = 1
sp_count = 0
for i in range(1,vel.shape[0]-1):
	for j in range(1,vel.shape[1]-1):
		if np.isnan(vel[i,j])==True:
			neighbors = np.append(np.append(np.append(vel[i-nneigh,j-nneigh:j+nneigh+1],vel[i+nneigh,j-nneigh:j+nneigh+1]),vel[i,j+nneigh]),vel[i,j-nneigh])
			vel[i,j]=np.median(neighbors)
			if np.isnan(vel[i,j])==False:
				sp_count+=1
# print('Replaced %d single NaN pixels with median of neighbors. %.2f%% of pixels replaced.\n' 
# 	%(sp_count,sp_count/len(vel)))

save_gaussfits(vel,hdr,filepath,'vcen_cgrad',outsuff)

