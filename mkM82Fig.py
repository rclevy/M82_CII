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

#open Nico's M82 image

co = fits.open('../../data/M82.CO.NOEMA+30m.peak.fits')
hdr = co[0].header
wcs = WCS(hdr)
co = co[0].data
co[co<0]=0.
co = 1.222E6*co/((hdr['RESTFRQ']/1E9)**2*hdr['BMAJ']*hdr['BMIN']*3600**2)
dist = 3.5E3 #kpc


#set colorbar limits
vmin= 0.
vmax = 30.
cbreak=0.0998 #default ds9 scaling
#norm=ImageNormalize(co,stretch=LogStretch(),interval=ManualInterval(vmax=vmax,vmin=vmin))
norm=ImageNormalize(co,stretch=AsinhStretch(cbreak),interval=ManualInterval(vmax=vmax,vmin=vmin))
cmap = cm.get_cmap('gist_yarg')

#open the continuum image
cont = fits.open('../../data/m82_sma_cont.fits')
hdr_cont = cont[0].header
wcs_cont = WCS(hdr_cont)
cont = cont[0].data
levels = np.linspace(-0.00353959,0.0236673,num=4)

#open the HI
hi = fits.open('../../data/m82_hi_24as_mom0.fits')
hdr_hi = hi[0].header
wcs_hi = WCS(hdr_hi).celestial
hi = hi[0].data
lev = np.linspace(1E20,1.5E21,num=20)
arcsec2pix = hdr_hi['CDELT2']*3600

hi[950:1040,954:1050]=np.nan
hmap = sns.color_palette('flare',as_cmap=True)

xlim,ylim = wcs_hi.all_world2pix([149.5,148.5],[69.5,69.8],0,ra_dec_order=True)
xlimz,ylimz = wcs_hi.all_world2pix([148.99166666666667,148.93749999999997],[69.675,69.68472222222222],0,ra_dec_order=True)


#get positions of SOFIA footprints to plot
xo = 869.5
yo = 1005.5
ang_map = -20 #deg
xl = 216./arcsec2pix
yl = 72./arcsec2pix

#get single pointing offsets
x1=973.5
y1=917.5
x2=945.5
y2=835.5
r = 78.1/2/arcsec2pix

center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
array_rot = 70. #deg
#mk_great_footprint(center_sp,array_rot=array_rot,color='b')
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')


fsz=14
fig = plt.figure(1,figsize=(10,10))
plt.clf()
ax = plt.subplot(projection=wcs_hi)
im=ax.imshow(co,origin='lower',cmap=cmap,vmin=vmin,vmax=np.nanmax(co),norm=norm,transform=ax.get_transform(wcs))
im.cmap.set_over('k')
im.cmap.set_under('w')
im.cmap.set_bad('w')
#ax.contour(cont,levels=levels,origin='lower',transform=ax.get_transform(wcs_cont),colors='cornflowerblue')
ax.contourf(hi,levels=lev,origin='lower',cmap=hmap,alpha=1.0)
ax.set_xlim(xlimz)
ax.set_ylim(ylimz)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.set_rasterization_zorder(10)
#plt.savefig('M82_CO_HI_SOFIA_zoom.pdf',rasterized=True,dpi=300)


#zoom out and plot SOFIA footprints
#im.set_vmax(vmax)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.add_patch(Rectangle((xo,yo),xl,yl,angle=ang_map,ec='b',fc='None',linewidth=1.5,zorder=10))
for i in range(len(fp)):
	this_fp = fp[i]
	this_fp.visual['linewidth'] = 1.5
	this_fp.to_pixel(wcs_hi).plot()
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(5)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
ax.set_rasterization_zorder(10)
plt.savefig('../Plots/Composite_Maps/M82_CO_HI_SOFIAfootprints.pdf',rasterized=True,dpi=300,bbox_iches='tight',metadata={'Creator':this_script})
plt.close()


# #plot with inverted background color and no footprints
# # cmap = cm.get_cmap('gist_gray')
# # cmap.set_over('w')
# # cmap.set_under('k')
# # cmap.set_bad('k')
# # black = np.zeros_like(hi)
# fig = plt.figure(1,figsize=(10,10))
# plt.clf()
# ax = plt.subplot(projection=wcs_hi)
# #ax.imshow(black,origin='lower',cmap=cmap,vmin=vmin,vmax=np.nanmax(co),norm=norm)
# im=ax.imshow(co,origin='lower',cmap=cmap,vmin=vmin,vmax=np.nanmax(co),norm=norm,transform=ax.get_transform(wcs))
# #ax.contour(cont,levels=levels,origin='lower',transform=ax.get_transform(wcs_cont),colors='dodgerblue')
# ax.contourf(hi,levels=lev,origin='lower',cmap=hmap,alpha=1.0)
# ax.set_xlim(xlim[0],xlim[1]+10)
# ax.set_ylim(ylim[0]-20,ylim[1]+20)
# ax.coords[0].set_major_formatter('hh:mm:ss.s')
# ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
# ax.coords[0].display_minor_ticks(True)
# ax.coords[1].display_minor_ticks(True)
# ax.coords[0].set_minor_frequency(4)
# ax.coords[1].set_minor_frequency(5)
# plt.xlabel('R.A. (J2000)')
# plt.ylabel('Decl. (J2000)')
# # ax.coords[0].set_ticks_visible(False)
# # ax.coords[0].set_ticklabel_visible(False)
# # ax.coords[1].set_ticks_visible(False)
# # ax.coords[1].set_ticklabel_visible(False)
# ax.set_rasterization_zorder(10)
# plt.savefig('M82_CO_HI.pdf',dpi=300)
# plt.close()



