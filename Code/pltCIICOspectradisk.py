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

#load the CO map
fname_co = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits'
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
# co[0:10,:]=np.nan
# co[:,0:10]=np.nan
# co[-10:,:]=np.nan
# co[:,-10:]=np.nan
#set colorbar limits
vmin_co = 0.
vmax_co = 800.
cbreak=0.0998 #default ds9 scaling
#norm=ImageNormalize(co,stretch=LogStretch(),interval=ManualInterval(vmax=vmax,vmin=vmin))
norm_co=ImageNormalize(co,stretch=AsinhStretch(cbreak),interval=ManualInterval(vmax=vmax_co,vmin=vmin_co))
cmap_co = cm.get_cmap('gist_yarg')
levels_co = np.logspace(np.log10(0.12),np.log10(vmax_co),10)
arcsec2pix = hdr_co['CDELT2']*3600


#open the SOFIA upgreat CII map
fname_cii = '../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits'
cii = fits.open(fname_cii)
hdr_cii = cii[0].header
wcs_cii = WCS(hdr_cii)
cii = cii[0].data
#print(np.nanmax(cii))
cmap_cii = sns.color_palette('crest_r',as_cmap=True)
# vmin_cii = 1.5
# vmax_cii = 8.
# levels_cii = np.arange(vmin_cii,vmax_cii+1,1.5)
vmin_cii = 100.
vmax_cii = 800.
levels_cii = np.arange(vmin_cii,vmax_cii+150,150)
#cii[22:,:]=np.nan
bmaj_cii = hdr_cii['BMAJ']*3600 #arcsec
bmin_cii = hdr_cii['BMIN']*3600 #arcsec
bpa_cii = hdr_cii['BPA'] #deg
outline_color_cii = cmap_cii(150)

#now open the [CII] and CO data cubes to extract the spectra
#open CII cube
cii_cube = fits.open('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits')
cii_cube_wcs = WCS(cii_cube[0].header)
cii_cube = cii_cube[0].data
cii_vel_axis = SpectralCube.read('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits').with_spectral_unit(u.km/u.s).spectral_axis.value

#open CO cube
fname_co_tp = '../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_matchedCIICO.fits'
co_tp_cube = fits.open(fname_co_tp)
co_tp_cube_hdr = co_tp_cube[0].header
co_tp_cube = co_tp_cube[0].data
co_tp_wcs = WCS(co_tp_cube_hdr)
co_tp_vel_axis = SpectralCube.read(fname_co_tp).with_spectral_unit(u.km/u.s).spectral_axis.value


#set the RA and Dec from which to extract spectra
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
pospk = SkyCoord('9h55m50.6793s','+69d40m44.0s')
posb = SkyCoord('9h55m47.0198s','+69d40m36.549s')
posr = SkyCoord('9h55m56.1595s','+69d40m52.909s')
pos = [center,pospk,posb,posr]
pos_labels = ['Center','[CII] Peak','Blueshifted','Redshifted']
pos_colors = ['xkcd:dandelion','mediumturquoise','royalblue','tomato']
pos_pix = np.zeros((len(pos),2))
apers_cii = ['']*len(pos)
apers_co = ['']*len(pos)
spec_co = np.zeros((len(pos),co_tp_cube.shape[0]))
spec_cii = np.zeros((len(pos),cii_cube.shape[0]))
for i in range(len(pos)):
	apers_cii[i] = SkyCircularAperture(pos[i],r=bmaj_cii/2*u.arcsec)
	mask_cii = apers_cii[i].to_pixel(cii_cube_wcs.celestial).to_mask(method='center')
	apers_co[i] = SkyCircularAperture(pos[i],r=bmaj_cii/2*u.arcsec)
	mask_co = apers_co[i].to_pixel(co_tp_wcs.celestial).to_mask(method='center')
	pos_pix[i,0],pos_pix[i,1] = wcs_co.all_world2pix(pos[i].ra.value,pos[i].dec.value,1,ra_dec_order=True)

	# for j in range(co_tp_cube.shape[0]):
	# 	spec_co[i,j] = np.nansum(mask_co.multiply(co_tp_cube[j,:,:],fill_value=np.nan))
	# for j in range(cii_cube.shape[0]):
	# 	spec_cii[i,j] = np.nansum(mask_cii.multiply(cii_cube[j,:,:],fill_value=np.nan))

	# for j in range(co_tp_cube.shape[0]):
	# 	spec_co[i,j] = np.nanmedian(mask_co.multiply(co_tp_cube[j,:,:],fill_value=np.nan))
	# for j in range(cii_cube.shape[0]):
	# 	spec_cii[i,j] = np.nanmedian(mask_cii.multiply(cii_cube[j,:,:],fill_value=np.nan))

	for j in range(co_tp_cube.shape[0]):
		spec_co[i,j] = co_tp_cube[j,int(pos_pix[i,1]),int(pos_pix[i,0])]
	pos_pix_cii = cii_cube_wcs.celestial.all_world2pix(pos[i].ra.value,pos[i].dec.value,1,ra_dec_order=True)
	for j in range(cii_cube.shape[0]):
		spec_cii[i,j] = cii_cube[j,int(pos_pix_cii[1]),int(pos_pix_cii[0])]

#get plot limits
#crop to 5'x2'
crop_arcmin = np.array([12.,4.])*u.arcmin/2
xlim_sky = [(center.ra-crop_arcmin[0]).value,(center.ra+crop_arcmin[0]).value]
ylim_sky = [(center.dec-crop_arcmin[1]).value,(center.dec+crop_arcmin[1]).value]
xlim,ylim = wcs_co.all_world2pix(xlim_sky,ylim_sky,1,ra_dec_order=True)

#get info to draw fully-sampled map region
center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
ang_map = -20. #deg, but the cube is already rotated by this much
fullsamp_x = 185.4 #arcsec
fullsamp_y = 65.4 #arcsec
xo,yo = wcs_co.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
xl = fullsamp_x/arcsec2pix
yl = fullsamp_y/arcsec2pix
xor = ((-xl/2)*np.cos(np.radians(ang_map))-(-yl/2)*np.sin(np.radians(ang_map)))+xo+0.5
yor = ((-xl/2)*np.sin(np.radians(ang_map))+(-yl/2)*np.cos(np.radians(ang_map)))+yo-0.5


#get info for spectra subplots
pxoff = [0.225, 0.52, 0.52, 0.225] #subplot position
pyoff = [0.65, 0.65,0.18, 0.18]
paxoff = [0.5, 0.5, 0.5, 0.75] #arrow end position
payoff = [0.25, 0.5, 0.75, 0.5]
ax_off = [0.1, 0.1, 0.04, 0.16]
ay_off = [0.175, 0.05, 0.05, 0.05]
x_off = np.array([0., 0., 0., -np.sqrt(2)/2])*bmaj_cii/2/arcsec2pix
y_off = np.array([-0.75, 0.75, 0.75, np.sqrt(2)/2])*bmaj_cii/2/arcsec2pix


#plot
sz=14
cii_color = sns.color_palette('crest',as_cmap=True)(150)
co_color = '0.33'
fig = plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs_co)
im=ax.imshow(co,origin='lower',cmap=cmap_co,norm=norm_co)
cb=plt.colorbar(im)
cb.set_label('I$_{\mathrm{int}}$ (K km s$^{-1}$)')
#ax.contourf(co,levels=levels_co,origin='lower',cmap=cmap_co,alpha=1.,extend='max')
con=ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=3,transform=ax.get_transform(wcs_cii))
cb.add_lines(con)
r=ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec='k',alpha=0.75,fc='None',linewidth=1.5,zorder=10))
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
ax.set_rasterization_zorder(10)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.invert_xaxis()

#now plot the extracted spectra
#BOTH ARE IN MAIN BEAM BRIGHTNESS TEMPS
for i in range(len(pos)):
	# e1=Ellipse((pos_pix[i,0],pos_pix[i,1]), bmaj_co/2/arcsec2pix, bmaj_co/2/arcsec2pix, 0, ec=co_color, fc='None',lw=2.0,zorder=10)
	# s1=Shadow(e1,0,0,color='0.9',fc='None',lw=4)
	# ax.add_patch(e1)
	# ax.add_patch(s1)
	e2=Ellipse((pos_pix[i,0],pos_pix[i,1]), bmaj_co/2/arcsec2pix, bmaj_co/2/arcsec2pix, 0, ec=pos_colors[i], fc='None',lw=2.0,zorder=10)
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
	#f2.set_ylim(-2,45)
	f2.set_ylim(-0.2,6)
	f2.minorticks_on()
	f2.tick_params(axis='both',which='both',labelsize=fsz)
	f2.set_xlabel('Velocity (km s$^{-1}$)',fontsize=fsz)
	f2.set_ylabel('T$_{\\mathrm{mb}}$ (K)',fontsize=fsz)
	f2.text(f2.get_xlim()[0],f2.get_ylim()[1]*0.95,' '+pos_labels[i],
		color=pos_colors[i],ha='left',va='top',fontsize=fsz-4,
		path_effects=[pe.Stroke(linewidth=1.0, foreground='dimgray'),pe.Normal()])

	
	# a1=ax.annotate('',xytext=(pos_pix[i,0]+x_off[i],pos_pix[i,1]+y_off[i]),textcoords='data',
	# 			xy=((pxoff[i]+ax_off[i],pyoff[i]+ay_off[i])),xycoords='axes fraction',
	# 			arrowprops=dict(arrowstyle='-|>',fc='k',ec='k',lw=2.0),
	# 			color='k',fontsize=fsz,zorder=8)



plt.savefig('../Plots/Composite_Maps/M82_CO_CII_spectra.pdf',dpi=300,metadata={'Creator':this_script})




