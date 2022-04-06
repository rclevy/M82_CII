this_script = __file__

#import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Rectangle)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] =14
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
import seaborn as sns
import copy
from mkGREATfootprint import mk_great_footprint
from regions import Regions,CircleSkyRegion
import aplpy
import matplotlib.image as img

#open Nico's M82 image

fname_co = '../Data/Ancillary_Data/M82.CO.NOEMA+30m.peak.fits'
co = fits.open(fname_co)
hdr_co = co[0].header
wcs_co = WCS(hdr_co)
co = co[0].data
co[co<0]=0.
co = 1.222E6*co/((hdr_co['RESTFRQ']/1E9)**2*hdr_co['BMAJ']*hdr_co['BMIN']*3600**2)
dist = 3.5E3 #kpc
bmaj_co = hdr_co['BMAJ']*3600 #arcsec
bmin_co = hdr_co['BMIN']*3600 #arcsec
bpa_co = hdr_co['BPA'] #deg


#set colorbar limits
vmin_co = 0.
vmax_co = 10.
cbreak=0.0998 #default ds9 scaling
#norm=ImageNormalize(co,stretch=LogStretch(),interval=ManualInterval(vmax=vmax,vmin=vmin))
norm_co=ImageNormalize(co,stretch=AsinhStretch(cbreak),interval=ManualInterval(vmax=vmax_co,vmin=vmin_co))
cmap_co = cm.get_cmap('gist_yarg')
levels_co = np.logspace(np.log10(1.),1,5)
# co[:,0:300]=np.nan
# co[:,1750:]=np.nan
# co[0:250,:]=np.nan
# co[1000:,:]=np.nan

#open the continuum image
fname_cont = '../Data/Ancillary_Data/m82_sma_cont.fits'
cont = fits.open(fname_cont)
hdr_cont = cont[0].header
wcs_cont = WCS(hdr_cont)
cont = cont[0].data
levels_cont = np.linspace(-0.00353959,0.0236673,num=4)

#open the HI
fname_hi = '../Data/Ancillary_Data/m82_hi_24as_mom0.fits'
hi = fits.open(fname_hi)
hdr_hi = hi[0].header
wcs_hi = WCS(hdr_hi)
hi = hi[0].data
vmin_hi = 1E20
vmax_hi = 1.5E21
lev = np.linspace(vmin_hi,vmax_hi,num=20)
arcsec2pix = hdr_hi['CDELT2']*3600
bmaj_hi = hdr_hi['BMAJ']*3600 #arcsec
bmin_hi = hdr_hi['BMIN']*3600 #arcsec
bpa_hi = hdr_hi['BPA'] #deg

hi[950:1040,954:1050]=np.nan
hmap = sns.color_palette('flare',as_cmap=True)
outline_color_hi = hmap(150)

xlim,ylim = wcs_hi.all_world2pix([149.25,148.75],[69.6,69.75],0,ra_dec_order=True)
xlimz,ylimz = wcs_hi.all_world2pix([148.99166666666667,148.93749999999997],[69.675,69.68472222222222],0,ra_dec_order=True)


#open the SOFIA upgreat CII map
fname_cii = '../Data/Disk_Map/M82_CII_map_peak_maskSNR3.fits'
cii = fits.open(fname_cii)
hdr_cii = cii[0].header
wcs_cii = WCS(hdr_cii)
cii = cii[0].data
cmap_cii = sns.color_palette('crest',as_cmap=True)
vmin_cii = 1.
vmax_cii = 10.
levels_cii = np.linspace(vmin_cii,vmax_cii,7)
#cii[22:,:]=np.nan
bmaj_cii = hdr_cii['BMAJ']*3600 #arcsec
bmin_cii = hdr_cii['BMIN']*3600 #arcsec
bpa_cii = hdr_cii['BPA'] #deg
outline_color_cii = cmap_cii(150)

#get positions for beams
x_text = xlim[0]+0.95*(xlim[1]-xlim[0])
y_text = ylim[0]+0.05*(ylim[1]-ylim[0])


center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
array_rot = 70. #deg
# #mk_great_footprint(center_sp,array_rot=array_rot,color='b')
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')

#first make an RGB image using aplpy
# aplpy.make_rgb_cube([fname_hi, fname_co, fname_cii], '../Data/M82_RGB_HI_CO_CII.fits')
# aplpy.make_rgb_image('../Data/M82_RGB_HI_CO_CII.fits','../Plots/Composite_Maps/M82_RGB_HI_CO_CII.png',
# 	vmin_r = vmin_hi, vmax_r = 2*vmax_hi,
# 	vmin_g = vmin_co, vmax_g = vmax_co, stretch_g = 'log',
# 	vmin_b = vmin_cii, vmax_b = vmax_cii)
# #f = aplpy.FITSFigure('../Data/M82_RGB_HI_CO_CII_2d.fits')
# plt.close('all')

fsz=14
fig = plt.figure(1,figsize=(10,10))
plt.clf()
ax = plt.subplot(projection=wcs_hi)
im=ax.imshow(co,origin='lower',cmap=cmap_co,vmin=vmin_co,vmax=vmax_co,transform=ax.get_transform(wcs_co))
im.cmap.set_over('k')
im.cmap.set_under('w')
im.cmap.set_bad('w')
ax.contourf(hi,levels=lev,origin='lower',cmap=hmap,alpha=1.0)
ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,transform=ax.get_transform(wcs_cii))
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
ax.add_patch(Ellipse((x_text,y_text), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
ax.add_patch(Ellipse((x_text,y_text), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='k', fc='None',lw=2.0)) #plot the beam
ax.add_patch(Ellipse((x_text,y_text), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII.pdf',dpi=300,metadata={'Creator':this_script})


#ax.add_patch(Rectangle((xo,yo),xl,yl,angle=ang_map,ec=outline_color_cii,fc='None',linewidth=1.5,zorder=10))
for i in range(len(fp)):
	this_fp = fp[i]
	this_fp.visual['linewidth'] = 1.5
	this_fp.visual['color'] = outline_color_cii
	this_fp.to_pixel(wcs_hi).plot()
plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_outflowfootprints.pdf',dpi=300,metadata={'Creator':this_script})


#now add the spectra in an inset
ax_in=ax.inset_axes([-0.03,0.01,0.5,0.5])
spec = img.imread('../Plots/Outflow_Spectra/M82_CII_Outflow1_sky_10kms.png')
ax_in.imshow(spec)
ax_in.tick_params(axis='both',bottom=False,left=False,labelbottom=False,labelleft=False)
for pos in ['bottom','top','left','right']:
	ax_in.spines[pos].set_color(outline_color_cii)
	ax_in.spines[pos].set_linewidth(2)
plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_Spectra.pdf',dpi=300,metadata={'Creator':this_script})




plt.close()


