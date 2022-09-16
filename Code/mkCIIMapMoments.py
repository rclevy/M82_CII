#make moment maps of CII in disk
#do this both as traditional moments and by fitting Gaussians to each pixel

import argparse

parser=argparse.ArgumentParser(
    description='''Make moment maps of [CII] in disk of M82. This makes traditional moment maps and maps based on Gaussian fitting. A masking level can be specified, based on the rms of the cube.''',
    epilog='''Author: R. C. Levy - Last updated: 2022-04-05''')
parser.add_argument('--mask_level', type=float, help='SNR level to mask maps (default: no masking)')

args=parser.parse_args()

import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import medfilt2d
from scipy.ndimage import convolve

def save_to_fits(data,hdr,filepath,maptype,outsuff):
	hdu = fits.PrimaryHDU(data=data,header=hdr)
	hdul = fits.HDUList(hdu)
	hdul.writeto(filepath+'_'+maptype+'_'+outsuff+'.fits',overwrite=True)
	return

def mask_out_undersampled(cube,filepath):
	#mask out parts of the cube that were undersampled
	#leaving only the fully sampled map region
	#do this first based on the original AOR
	#may be better to do this based on some coverage map or intgration time per pointing
	from astropy.coordinates import SkyCoord
	from astropy.wcs import WCS
	center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
	map_angle = -20. #deg, but the cube is already rotated by this much
	fullsamp_x = 185.4 #arcsec
	fullsamp_y = 65.4 #arcsec
	arcsec2pix = cube.header['CDELT2']*3600
	wcs = WCS(cube.header)
	xcen_pix,ycen_pix = wcs.celestial.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
	fullsamp_x_pix = fullsamp_x/arcsec2pix
	fullsamp_y_pix = fullsamp_y/arcsec2pix
	mask_x = np.array([np.floor(xcen_pix-fullsamp_x_pix/2), np.ceil(xcen_pix+fullsamp_x_pix/2)]).astype(int)
	mask_y = np.array([np.floor(ycen_pix-fullsamp_y_pix/2), np.ceil(ycen_pix+fullsamp_y_pix/2)]).astype(int)

	#mask out pixels outside this box
	data = cube.unmasked_data[:]
	data[:,:,0:mask_x[0]]=np.nan
	data[:,:,mask_x[1]:]=np.nan
	data[:,0:mask_y[0],:]=np.nan
	data[:,mask_y[1]:,:]=np.nan

	masked_cube = SpectralCube(data=data,wcs=wcs,header=cube.header)
	masked_cube.write(filepath+'_fullysampled.fits',overwrite=True)
	return masked_cube

def make_moments(cube,filepath,outsuff):
	mom0 = cube.moment(order=0)
	mom1 = cube.moment(order=1)
	mom2 = cube.linewidth_fwhm()
	mmax = cube.max(axis=0)
	mom0.write(filepath+'_mom0_'+outsuff+'.fits',overwrite=True)
	mom1.write(filepath+'_mom1_'+outsuff+'.fits',overwrite=True)
	mom2.write(filepath+'_mom2_'+outsuff+'.fits',overwrite=True)
	mmax.write(filepath+'_max_'+outsuff+'.fits',overwrite=True)
	return mom0,mom1,mom2,mmax

def mk_rms_map(cube,mom1,mom2):
	#make a map of the cube rms
	vmins = mom1-2*mom2/2.355
	vmaxs = mom1+2*mom2/2.355
	vel_axis = cube.spectral_axis
	cube_nosig = cube.unmasked_data[:].value
	for i in range(vmins.shape[0]):
		for j in range(vmins.shape[1]):
			idx_min = np.argmin(np.abs(vmins[i,j].value-vel_axis.value))
			idx_max = np.argmin(np.abs(vmaxs[i,j].value-vel_axis.value))
			cube_nosig[idx_max:idx_min,i,j] = np.nan
	rms_map = np.sqrt(np.nanmean(cube_nosig**2,axis=0))
	return rms_map

def make_emoments(mom0,mom1,mom2,mmax,rms_map,dv,filepath,outsuff):
	e_mmax = rms_map.copy()
	e_mom0 = dv*rms_map
	e_mom1 = dv/mom0.value*np.sqrt(mom0.value**2+e_mom0**2+(mom1.value*rms_map)**2)
	e_mom2 = rms_map.copy() ####WRONG

	e_mmax_hdr = cube.wcs.celestial.to_header()
	e_mmax_hdr['BUNIT'] = 'K'

	save_to_fits(e_mom0,mom0.header,filepath,'emom0',outsuff)
	save_to_fits(e_mom1,mom1.header,filepath,'emom1',outsuff)
	save_to_fits(e_mom2,mom2.header,filepath,'emom2',outsuff)
	save_to_fits(e_mmax,e_mmax_hdr,filepath,'emax',outsuff)
	return e_mom0,e_mom1,e_mom2,e_mmax


filepath = '../Data/Disk_Map/M82_CII_map'
filepath_mom = '../Data/Disk_Map/Moments/M82_CII_map'

#load the cube
cube = SpectralCube.read(filepath+'_cube.fits').with_spectral_unit(u.km/u.s)*u.K
hdr = cube.header
wcs = cube.wcs

#mask to the fully sampled region
cube = mask_out_undersampled(cube,filepath)

#now mask based on the SNR
if args.mask_level:
	level = args.mask_level
else:
	level = np.nan
rms_full = np.sqrt(np.nanmean(cube**2))*u.K
cube_rmsignal = cube.with_mask(cube < 2*rms_full)
rms = np.sqrt(np.nanmean(cube_rmsignal**2))*u.K
print('rms away from signal in 10 km/s channels = %.1f mK' %(rms.value*1000))

#shift the cube by 1 pixel along the major axis
cube = np.roll(cube,-1,axis=2)
cube = SpectralCube(data=cube*u.K,wcs=wcs,header=hdr)

#now limit to -100-400 km/s
vmin = -100.*u.km/u.s
vmax = 400.*u.km/u.s
cube_vmask = cube.spectral_slab(vmin,vmax)

if np.isnan(level) == True:
	outsuff = 'nomask'
	cube_masked = cube_vmask
else:
	outsuff = 'maskSNR'+str(int(level))
	cube_masked = cube_vmask.with_mask(cube_vmask > level*rms)
mom0,mom1,mom2,mmax=make_moments(cube_masked,filepath_mom,outsuff)

#now get propagated uncertainties on the moment maps based on the RMS
rms_map = mk_rms_map(cube,mom1,mom2)
dv = 10. #km/s
e_mom0,e_mom1,e_mom2,e_mmax = make_emoments(mom0,mom1,mom2,mmax,rms_map,dv,filepath_mom,outsuff)


# #now make "moments" by fitting a Gaussian to each spectrum
# def gauss(x,a,b,c):
# 	return a*np.exp(-(x-b)**2/(2*c**2))

# peak = np.nan*np.ones((cube.shape[1],cube.shape[2]))
# epeak = peak.copy()
# vel = peak.copy()
# evel = peak.copy()
# fwhm = peak.copy()
# efwhm = peak.copy()

# #based on mkvelmap.m
# iy,ix = np.where(np.isnan(mom0.value)==False)
# v = cube_masked.spectral_axis.value #km/s
# v_span = np.nanmax(v)-np.nanmin(v)
# dv = np.abs(cube_masked.header['CDELT3']) #km/s

# for i in range(len(ix)):
# 	s = cube_masked[:,iy[i],ix[i]].value
# 	s[np.isnan(s)==True]=0.
# 	try:
# 		p,pcov = curve_fit(gauss,v,s,
# 			p0=(np.nanmax(s),np.mean([np.nanmax(v),np.nanmin(v)]),5*dv),
# 			bounds=([0,np.nanmin(v),dv],[np.inf,np.nanmax(v),v_span/2]))
# 		ep = np.sqrt(np.diag(pcov))
# 	except RuntimeError:
# 		p = np.array([np.nan, np.nan, np.nan])
# 		ep = np.array([np.nan, np.nan, np.nan])


# 	peak[iy[i],ix[i]] = p[0]
# 	vel[iy[i],ix[i]] = p[1]
# 	fwhm[iy[i],ix[i]] = p[2]*2.355
# 	epeak[iy[i],ix[i]] = ep[0]
# 	evel[iy[i],ix[i]] = ep[1]
# 	efwhm[iy[i],ix[i]] = ep[2]*2.355


# #get integrated intensity based on Gaussian fits
# iint = peak*fwhm/2.355*np.sqrt(np.pi)
# eiint = np.sqrt((epeak*fwhm/2.355*np.sqrt(np.pi))**2+(peak*efwhm/2.355*np.sqrt(np.pi))**2)

# hdr = mom0.header
# hdr['BUNIT']='K'
# save_to_fits(peak,hdr,filepath_mom,'peak',outsuff)
# save_to_fits(epeak,hdr,filepath_mom,'epeak',outsuff)
# hdr['BUNIT']='km/s'
# save_to_fits(vel,hdr,filepath_mom,'vcen',outsuff)
# save_to_fits(evel,hdr,filepath_mom,'evcen',outsuff)
# save_to_fits(fwhm,hdr,filepath_mom,'fwhm',outsuff)
# save_to_fits(efwhm,hdr,filepath_mom,'efwhm',outsuff)
# hdr['BUNIT']='K km/s'
# save_to_fits(iint,hdr,filepath_mom,'intinten',outsuff)
# save_to_fits(iint,hdr,filepath_mom,'eintinten',outsuff)

# #now do some masking of the vcen field
# maxdev = 40. #km/s
# fm5 = medfilt2d(vel,[5,5])
# fm3 = medfilt2d(vel,[3,3])
# um = np.ones(vel.shape)
# um[np.isnan(vel)==True]=0.
# pc5 = convolve(um,np.ones((5,5)),mode='nearest') # count how many neighbors a pixel has that are not NaN
# ix3 = np.where((pc5<25) & (pc5>=9) & (np.abs(fm3-vel)>maxdev))
# ix5 = np.where((pc5==25) & (np.abs(fm5-vel)>maxdev))
# vel[ix3] = fm3[ix3] # filter out things sticking out in the unsharp masking
# vel[ix5] = fm5[ix5] # filter out things sticking out in the unsharp masking
# #print('Replaced %d for 3x3 median and %d for 5x5 median. %.2f%% of pixels replaced.\n'
# #	%(len(ix3[0]),len(ix5[0]),(len(ix3[0])+len(ix5[0]))/len(vel[0])))

# #replace single NaN pixels with the median of its neighbors
# nneigh = 1
# sp_count = 0
# for i in range(1,vel.shape[0]-1):
# 	for j in range(1,vel.shape[1]-1):
# 		if np.isnan(vel[i,j])==True:
# 			neighbors = np.append(np.append(np.append(vel[i-nneigh,j-nneigh:j+nneigh+1],vel[i+nneigh,j-nneigh:j+nneigh+1]),vel[i,j+nneigh]),vel[i,j-nneigh])
# 			vel[i,j]=np.median(neighbors)
# 			if np.isnan(vel[i,j])==False:
# 				sp_count+=1
# # print('Replaced %d single NaN pixels with median of neighbors. %.2f%% of pixels replaced.\n' 
# # 	%(sp_count,sp_count/len(vel)))

# save_to_fits(vel,hdr,filepath_mom,'vcen_cgrad',outsuff)

