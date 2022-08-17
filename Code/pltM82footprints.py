this_script = __file__

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from astropy.io import fits
from astropy.wcs import WCS
from astropy import wcs
from astropy.coordinates import SkyCoord,Angle
from astropy.visualization import (ManualInterval, AsinhStretch, ImageNormalize, LogStretch)
import warnings
import astropy.wcs as wcs
from mkGREATfootprint import mk_great_footprint
from regions import Regions,CircleSkyRegion
import astropy.units as u
warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning) #don't throw errors about the fits headers...

plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 16

#import the CO image of M82 from Krieger+2021
co = fits.open('../../data/M82.CO.NOEMA+30m.peak.fits')
hdr_co = co[0].header
wcs_co = WCS(hdr_co)
arcsec2pix = hdr_co['CDELT2']*3600 #arcsec/pix
data_co = co[0].data
data_co[np.isnan(data_co)==True]=0.
data_co[data_co<0]=0.
data_co = 1.222E6*data_co/((hdr_co['RESTFRQ']/1E9)**2*hdr_co['BMAJ']*hdr_co['BMIN']*3600**2)
dist = 3.63E3 #kpc
vmin= 0.
vmax = 30.
cbreak=0.0998 #default ds9 scaling
#norm=ImageNormalize(data_co,stretch=LogStretch(),interval=ManualInterval(vmax=vmax,vmin=vmin))
norm=ImageNormalize(data_co,stretch=AsinhStretch(cbreak),interval=ManualInterval(vmax=vmax,vmin=vmin))


#open the HI
hi = fits.open('../../data/m82_hi_24as_mom0.fits')
hdr_hi = hi[0].header
wcs_hi = WCS(hdr_hi).celestial
hi = hi[0].data
lev = np.linspace(1E20,1.5E21,num=20)
hi[950:1040,954:1050]=np.nan
hmap = sns.color_palette('flare',as_cmap=True)

#make the cycle 8 SOFIA footprint
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset_c8 = (29.152,-80.093) #arcsec
center_sp_c8 = SkyCoord(center.ra+sp_offset_c8[0]*u.arcsec,center.dec+sp_offset_c8[1]*u.arcsec)
array_rot = 70. #deg
#mk_great_footprint(center_sp_c8,array_rot=array_rot,color='b')
fp_c8 = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp_c8.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')
#get the box for the map of the center from C8
xo = 869.5
yo = 1005.5
ang_map = -20 #deg
xo,yo = wcs_hi.all_pix2world(xo,yo,0,ra_dec_order=True)
xo,yo = wcs_co.all_world2pix(xo,yo,0,ra_dec_order=True)
xl = 216./arcsec2pix
yl = 72./arcsec2pix

#make new SOFIA footprints
#sp_offset = (-170,+110) #arcsec
sp_offset = (55,+85) #arcsec
array_rot = 30.0
center_sp= SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
chop_throw = Angle(360.,'arcsec')
chop_angle = np.radians(125.)
dbs1 = center_sp.directional_offset_by(chop_angle,chop_throw)
dbs2 = center_sp.directional_offset_by(chop_angle+np.pi,chop_throw)
# mk_great_footprint(center_sp,array_rot=array_rot,color='k')
# mk_great_footprint(dbs1,array_rot=array_rot,color='teal')
# mk_great_footprint(dbs2,array_rot=array_rot,color='teal')
fp = Regions.read('../Data/upGREAT_Footprint_Regions/../../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')
fp_dbs1 = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+dbs1.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')
fp_dbs2 = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+dbs2.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')


#set up cropping
crop_diameter = 6*60. #arcsec
xcen,ycen = center.to_pixel(wcs_co)
cropx=[xcen-crop_diameter/2/arcsec2pix, xcen+crop_diameter/2/arcsec2pix]
cropy=[ycen-crop_diameter/2/arcsec2pix, ycen+crop_diameter/2/arcsec2pix]
x_text = cropx[0]+0.05*(cropx[1]-cropx[0])
y_text = cropy[0]+0.05*(cropy[1]-cropy[0])

#add a scale bar
sb_kpc = 1. #kpc
sb_arcsec = sb_kpc/dist*206265 #arcsec
sb_pix = sb_arcsec/arcsec2pix
sb_text = str(int(sb_kpc))+' kpc='+str(np.round(sb_arcsec/60,1))+"'"


#plot the composite image
fig=plt.figure(1,figsize=(8,8))
plt.clf()
ax = plt.subplot(projection=wcs_co)
im=ax.imshow(data_co,cmap='RdBu_r',origin='lower',norm=norm)
#im=ax.imshow(data_co,cmap='gist_yarg',origin='lower',norm=norm)
plt.colorbar(im,shrink=0.7)
ax.add_patch(Rectangle((xo,yo),xl,yl,angle=ang_map,ec='b',fc='None',linewidth=1.5,zorder=10))
if crop_diameter > 360.:
	ax.contourf(hi,levels=lev,origin='lower',alpha=0.25,antialiased=True,cmap=hmap,transform=ax.get_transform(wcs_hi))
	ax.set_rasterization_zorder(10)
for i in range(len(fp)):
	this_fp_c8 = fp_c8[i]
	this_fp_c8.visual['linewidth'] = 1.0
	this_fp_c8.to_pixel(wcs_co).plot(label='Cycle 8')

	this_fp = fp[i]
	this_fp.visual['linewidth'] = 2. #make science footprint thicker
	this_fp.to_pixel(wcs_co).plot(label='This proposal')
	fp_dbs1[i].to_pixel(wcs_co).plot()
	fp_dbs2[i].to_pixel(wcs_co).plot()
	# ax.text(this_fp.to_pixel(wcs_co).center.x,this_fp.to_pixel(wcs_co).center.y,str(i),
	# 		ha='center',va='center',color='r',fontsize = plt.rcParams['font.size']-4)
ax.plot([x_text,x_text+sb_pix],[y_text,y_text],'k-',lw=2)
#ax.text((2*x_text+sb_pix)/2,y_text*0.999,sb_text,color='k',ha='center',va='top')
ax.set_xlim(cropx)
ax.set_ylim(cropy)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.coords[0].set_ticklabel(exclude_overlapping=True)
ax.coords[1].set_ticklabel(exclude_overlapping=True)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)')
plt.savefig('../Plots/Composite_Maps/M82_CO_upGREAT_'+str(int(crop_diameter/60))+'arcmin.pdf',bbox_inches='tight',metadata={'Creator':this_script})


#make large figure
#set up cropping
crop_diameter = 17*60. #arcsec
xcen,ycen = center.to_pixel(wcs_co)
cropx=[xcen-crop_diameter/2/arcsec2pix, xcen+crop_diameter/2/arcsec2pix]
cropy=[ycen-crop_diameter/2/arcsec2pix, ycen+crop_diameter/2/arcsec2pix]
x_text = cropx[0]+0.05*(cropx[1]-cropx[0])
y_text = cropy[0]+0.05*(cropy[1]-cropy[0])

data_co[data_co==0]=np.nan

#plot the composite image
fig=plt.figure(1,figsize=(8,8))
plt.clf()
ax = plt.subplot(projection=wcs_co)
#im=ax.imshow(data_co,cmap='RdBu_r',origin='lower',norm=norm)
im=ax.imshow(data_co,cmap='gist_yarg',origin='lower',norm=norm)
ax.add_patch(Rectangle((xo,yo),xl,yl,angle=ang_map,ec='b',fc='None',linewidth=1.5,zorder=10))
ax.contourf(hi,levels=lev,origin='lower',alpha=0.25,antialiased=True,cmap=hmap,transform=ax.get_transform(wcs_hi))
ax.set_rasterization_zorder(10)
for i in range(len(fp)):
	this_fp_c8 = fp_c8[i]
	this_fp_c8.visual['linewidth'] = 1.0
	this_fp_c8.to_pixel(wcs_co).plot(label='Cycle 8')

	this_fp = fp[i]
	this_fp.visual['linewidth'] = 2. #make science footprint thicker
	this_fp.to_pixel(wcs_co).plot(label='This proposal')
	fp_dbs1[i].to_pixel(wcs_co).plot()
	fp_dbs2[i].to_pixel(wcs_co).plot()
	# ax.text(this_fp.to_pixel(wcs_co).center.x,this_fp.to_pixel(wcs_co).center.y,str(i),
	# 		ha='center',va='center',color='r',fontsize = plt.rcParams['font.size']-4)
ax.plot([x_text,x_text+sb_pix],[y_text,y_text],'k-',lw=2)
#ax.text((2*x_text+sb_pix)/2,y_text*0.999,sb_text,color='k',ha='center',va='top')
ax.set_xlim(cropx)
ax.set_ylim(cropy)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.coords[0].set_ticklabel(exclude_overlapping=True)
ax.coords[1].set_ticklabel(exclude_overlapping=True)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)')
plt.savefig('../Plots/Composite_Maps/M82_CO_upGREAT_'+str(int(crop_diameter/60))+'arcmin.pdf',bbox_inches='tight',metadata={'Creator':this_script})


