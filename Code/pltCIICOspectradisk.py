#plot the CII and CO spectra in the disk
# at the center and on the red and blueshifted sides

this_script = __file__

#import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Rectangle,Shadow)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.patheffects as pe
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,Angle
import astropy.units as u
from astropy.visualization import (ManualInterval, AsinhStretch, ImageNormalize, LogStretch)
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import rotate
import seaborn as sns
import copy
from mkGREATfootprint import mk_great_footprint
from regions import Regions,CircleSkyRegion
from photutils.aperture import SkyCircularAperture,aperture_photometry
from spectral_cube import SpectralCube

#load the CO
fname_co = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_peak.fits'
co = fits.open(fname_co)
hdr_co = co[0].header
wcs_co = WCS(hdr_co)
co = co[0].data
co[co<0]=0.
#co = 1.222E6*co/((hdr_co['RESTFRQ']/1E9)**2*hdr_co['BMAJ']*hdr_co['BMIN']*3600**2)
dist = 3.63E3 #kpc
bmaj_co = hdr_co['BMAJ']*3600 #arcsec
bmin_co = hdr_co['BMIN']*3600 #arcsec
bpa_co = hdr_co['BPA'] #deg
#trim the edges of the TP map
co[0:10,:]=np.nan
co[:,0:10]=np.nan
co[-10:,:]=np.nan
co[:,-10:]=np.nan
#set colorbar limits
vmin_co = 0.
vmax_co = 3.
cbreak=0.0998 #default ds9 scaling
#norm=ImageNormalize(co,stretch=LogStretch(),interval=ManualInterval(vmax=vmax,vmin=vmin))
norm_co=ImageNormalize(co,stretch=AsinhStretch(cbreak),interval=ManualInterval(vmax=vmax_co,vmin=vmin_co))
cmap_co = cm.get_cmap('gist_yarg')
levels_co = np.logspace(np.log10(0.12),np.log10(vmax_co),10)
arcsec2pix = hdr_co['CDELT2']*3600


#open the SOFIA upgreat CII map
fname_cii = '../Data/Disk_Map/Moments/M82_CII_map_mom0_maskSNR2.fits'
cii = fits.open(fname_cii)
hdr_cii = cii[0].header
wcs_cii = WCS(hdr_cii)
cii = cii[0].data
#print(np.nanmax(cii))
cmap_cii = sns.color_palette('crest_r',as_cmap=True)
# vmin_cii = 1.
# vmax_cii = 10.
vmin_cii = 200.
vmax_cii = 1200.
levels_cii = np.arange(vmin_cii,vmax_cii+200,200)
#cii[22:,:]=np.nan
bmaj_cii = hdr_cii['BMAJ']*3600 #arcsec
bmin_cii = hdr_cii['BMIN']*3600 #arcsec
bpa_cii = hdr_cii['BPA'] #deg
outline_color_cii = cmap_cii(150)

#now open the [CII] and CO data cubes to extract the spectra
#open CII cube
cii_cube = fits.open('../Data/Disk_Map/M82_CII_map_fullysampled.fits')
cii_cube_wcs = WCS(cii_cube[0].header)
cii_cube = cii_cube[0].data
cii_vel_axis = SpectralCube.read('../Data/Disk_Map/M82_CII_map_fullysampled.fits').with_spectral_unit(u.km/u.s).spectral_axis.value

#open CO cube
fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed.fits'
co_tp_cube = fits.open(fname_co_tp)
co_tp_cube_hdr = co_tp_cube[0].header
co_tp_cube = co_tp_cube[0].data
co_tp_wcs = WCS(co_tp_cube_hdr)
co_tp_vel_axis = SpectralCube.read(fname_co_tp).with_spectral_unit(u.km/u.s).spectral_axis.value


#set the RA and Dec from which to extract spectra
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
posb = SkyCoord('9h55m47.2450s','+69d40m36.002s')
posr = SkyCoord('9h55m54.6731s','+69d41m00.805s')
pos = [center,posb,posr]
pos_pix = np.zeros((len(pos),2))
apers_cii = ['']*len(pos)
apers_co = ['']*len(pos)
spec_co = np.zeros((len(pos),co_tp_cube.shape[0]))
spec_cii = np.zeros((len(pos),cii_cube.shape[0]))
for i in range(len(pos)):
	apers_cii[i] = SkyCircularAperture(pos[i],r=bmaj_cii/2*u.arcsec)
	mask_cii = apers_cii[i].to_pixel(cii_cube_wcs.celestial).to_mask(method='center')
	apers_co[i] = SkyCircularAperture(pos[i],r=bmaj_co/2*u.arcsec)
	mask_co = apers_co[i].to_pixel(co_tp_wcs.celestial).to_mask(method='center')
	pos_pix[i,0],pos_pix[i,1] = wcs_co.all_world2pix(pos[i].ra.value,pos[i].dec.value,1,ra_dec_order=True)

	for j in range(co_tp_cube.shape[0]):
		spec_co[i,j] = np.nansum(mask_co.multiply(co_tp_cube[j,:,:],fill_value=np.nan))
	for j in range(cii_cube.shape[0]):
		spec_cii[i,j] = np.nansum(mask_cii.multiply(cii_cube[j,:,:],fill_value=np.nan))

#get plot limits
xlim,ylim = wcs_co.all_world2pix([149.1,148.85],[69.64,69.715],0,ra_dec_order=True)


#get info for spectra subplots
pxoff = [0.42, 0.6, 0.27] #subplot position
pyoff = [0.2, 0.65, 0.65]
paxoff = [0.5, 0.5, 0.75] #arrow end position
payoff = [0.25, 0.75, 0.5]
ax_off = [0.1, 0.04, 0.16]
ay_off = [0.175, 0.05, 0.05]
x_off = np.array([0., 0., -np.sqrt(2)/2])*bmaj_cii/2/arcsec2pix
y_off = np.array([-0.75, 0.75, np.sqrt(2)/2])*bmaj_cii/2/arcsec2pix


#plot
sz=14
cii_color = sns.color_palette('crest',as_cmap=True)(150)
co_color = '0.33'
fig = plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs_co)
ax.contourf(co,levels=levels_co,origin='lower',cmap=cmap_co,alpha=1.,extend='max')
ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=3,transform=ax.get_transform(wcs_cii))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
ax.set_rasterization_zorder(10)

#now plot the extracted spectra
#BOTH ARE IN MAIN BEAM BRIGHTNESS TEMPS
for i in range(len(pos)):
	e1=Ellipse((pos_pix[i,0],pos_pix[i,1]), bmaj_co/2/arcsec2pix, bmaj_co/2/arcsec2pix, 0, ec=co_color, fc='None',lw=2.0,zorder=10)
	s1=Shadow(e1,0,0,color='0.9',fc='None',lw=4)
	ax.add_patch(e1)
	ax.add_patch(s1)
	e2=Ellipse((pos_pix[i,0],pos_pix[i,1]), bmaj_cii/2/arcsec2pix, bmaj_cii/2/arcsec2pix, 0, ec=cii_color, fc='None',lw=2.0,zorder=10)
	s2=Shadow(e2,0,0,color='0.9',fc='None',lw=4)
	ax.add_patch(e2)
	ax.add_patch(s2)
ax.set_aspect('equal')

fsz=18
for i in range(len(pos)):
	f2 = fig.add_axes([pxoff[i],pyoff[i],0.2,0.2])
	f2.step(co_tp_vel_axis,spec_co[i],color=co_color,linewidth=2)
	f2.step(cii_vel_axis,spec_cii[i],color=cii_color,linewidth=2)
	f2.axvline(210,color='k',lw=0.75)
	f2.set_xlim(-150,450)
	f2.minorticks_on()
	f2.tick_params(axis='both',which='both',labelsize=fsz)
	f2.set_xlabel('Velocity (km s$^{-1}$)',fontsize=fsz)
	f2.set_ylabel('T$_{\\mathrm{mb}}$ (K)',fontsize=fsz)

	
	ax.annotate('',xytext=(pos_pix[i,0]+x_off[i],pos_pix[i,1]+y_off[i]),textcoords='data',
				xy=((pxoff[i]+ax_off[i],pyoff[i]+ay_off[i])),xycoords='axes fraction',
				arrowprops=dict(arrowstyle='-|>',fc='k',ec='k',lw=2.0),
				color='k',fontsize=fsz,zorder=8)


plt.savefig('../Plots/Composite_Maps/M82_CO_CII_spectra.pdf',dpi=300,metadata={'Creator':this_script})




