this_script = __file__

#import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Rectangle)
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
import aplpy
import matplotlib.image as img
from photutils.aperture import SkyEllipticalAperture,aperture_photometry


def padding(array, xx, yy):
	"""
	:param array: numpy array
	:param xx: desired height
	:param yy: desirex width
	:return: padded array
	https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size
	"""

	h = array.shape[0]
	w = array.shape[1]

	a = (xx - h) // 2
	aa = xx - a - h

	b = (yy - w) // 2
	bb = yy - b - w

	return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')



#open Nico's M82 image

#fname_co = '../Data/Ancillary_Data/M82.CO.NOEMA+30m.peak.fits'
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

# #trim the noisy edges of the CO map for plotting
# co[0:268,:]=np.nan
# co[1080:,:]=np.nan
# co[696:,1470:]=np.nan
# co[870:,1260:]=np.nan
# co[:,1715:]=np.nan
# co[0:414,0:776]=np.nan
# co[0:724,0:448]=np.nan
# co[:,0:222]=np.nan
# co[0:391,1628:]=np.nan

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
# co[:,0:300]=np.nan
# co[:,1750:]=np.nan
# co[0:250,:]=np.nan
# co[1000:,:]=np.nan

#open the continuum image
# fname_cont = '../Data/Ancillary_Data/m82_sma_cont.fits'
# cont = fits.open(fname_cont)
# hdr_cont = cont[0].header
# wcs_cont = WCS(hdr_cont)
# cont = cont[0].data
# levels_cont = np.linspace(-0.00353959,0.0236673,num=4)

#open the HI
fname_hi = '../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor_mom0.fits'
hi = fits.open(fname_hi)
hdr_hi = hi[0].header
wcs_hi = WCS(hdr_hi)
hi = hi[0].data
# vmin_hi = 1E20
# vmax_hi = 1.5E21
vmin_hi = 3.84
vmax_hi = 800.
lev = np.linspace(vmin_hi,vmax_hi,num=20)
arcsec2pix = hdr_hi['CDELT2']*3600
bmaj_hi = hdr_hi['BMAJ']*3600 #arcsec
bmin_hi = hdr_hi['BMIN']*3600 #arcsec
bpa_hi = hdr_hi['BPA'] #deg

#mask out the center where there's HI absorption
hi_cen = SkyCoord('9h55m53.0s +69d40m41.0s',frame='fk5')
a = 58*u.arcsec
b = 38*u.arcsec
ang = 55*u.deg
aper = SkyEllipticalAperture(hi_cen,a,b,theta=ang)
aper_mask = aper.to_pixel(wcs_hi).to_mask(method='center')
aper_mask.data = padding(aper_mask.data,hi.shape[1],hi.shape[0])
aper_mask.data[aper_mask.data==1]=np.nan
aper_mask.data[aper_mask.data==0]=1.
hi = hi*aper_mask.data
hi[np.isnan(hi)==True]=1.
aper_mask.data[np.isnan(aper_mask.data)==True]=np.nanmax(hi)
hi = hi*aper_mask.data
hi[hi==1]=np.nan

#save this filled version to a fits file
hdu = fits.PrimaryHDU(data=hi,header=hdr_hi)
hdul = fits.HDUList([hdu])
hdul.writeto('../Data/Ancillary_Data/m82_hi_image_5kms_feathered_pbcor_mom0_fillcen.fits',overwrite=True)

hmap = sns.color_palette('flare',as_cmap=True)
outline_color_hi = hmap(150)

xlimo,ylimo = wcs_hi.all_world2pix([149.5,148.5],[69.5,69.8],0,ra_dec_order=True)
xlim,ylim = wcs_hi.all_world2pix([149.1,148.85],[69.64,69.715],0,ra_dec_order=True)
xlimz,ylimz = wcs_hi.all_world2pix([148.99166666666667,148.93749999999997],[69.675,69.68472222222222],0,ra_dec_order=True)


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

#get positions for beams
x_text = xlim[0]+0.95*(xlim[1]-xlim[0])
y_text = ylim[0]+0.075*(ylim[1]-ylim[0])
x_text_sb = xlim[0]+0.05*(xlim[1]-xlim[0])
x_texto = xlimo[0]+0.95*(xlimo[1]-xlimo[0])
y_texto = ylimo[0]+0.05*(ylimo[1]-ylimo[0])
x_texto_sb = xlimo[0]+0.05*(xlimo[1]-xlimo[0])

#add scale bar
sb_kpc = 2. #kpc
sb_pc = 200. #pc
sb_kpc_pix = sb_kpc/dist*206265/arcsec2pix
sb_pc_pix = sb_pc/1E3/dist*206265/arcsec2pix
sb_kpc_text = str(int(sb_kpc))+' kpc'
sb_pc_text = str(int(sb_pc))+' pc'


center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
center_wcs_hi = wcs_hi.all_world2pix(center.ra.value,center.dec.value,1,ra_dec_order=True)
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
array_rot = 70. #deg
# #mk_great_footprint(center_sp,array_rot=array_rot,color='b')
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')
reg_order = [0,5,6,1,2,3,4]

#get info to draw fully-sampled map region
center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
ang_map = -20. #deg, but the cube is already rotated by this much
fullsamp_x = 185.4 #arcsec
fullsamp_y = 65.4 #arcsec
xo,yo = wcs_hi.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
xl = fullsamp_x/arcsec2pix
yl = fullsamp_y/arcsec2pix
xor = ((-xl/2)*np.cos(np.radians(ang_map))-(-yl/2)*np.sin(np.radians(ang_map)))+xo
yor = ((-xl/2)*np.sin(np.radians(ang_map))+(-yl/2)*np.cos(np.radians(ang_map)))+yo


#first make an RGB image using aplpy
# aplpy.make_rgb_cube([fname_hi, fname_co, fname_cii], '../Data/M82_RGB_HI_CO_CII.fits')
# aplpy.make_rgb_image('../Data/M82_RGB_HI_CO_CII.fits','../Plots/Composite_Maps/M82_RGB_HI_CO_CII.png',
# 	vmin_r = vmin_hi, vmax_r = 2*vmax_hi,
# 	vmin_g = vmin_co, vmax_g = vmax_co, stretch_g = 'log',
# 	vmin_b = vmin_cii, vmax_b = vmax_cii)
# #f = aplpy.FITSFigure('../Data/M82_RGB_HI_CO_CII_2d.fits')
# plt.close('all')


fsz=14
fig = plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs_hi)
im=ax.imshow(hi,origin='lower',cmap=hmap,vmin=vmin_hi,vmax=vmax_hi*1.25)
ax.contourf(co,levels=levels_co,origin='lower',cmap=cmap_co,alpha=0.6,transform=ax.get_transform(wcs_co),extend='max')
#ax.contour(cii,origin='lower',levels=levels_cii,colors='w',linewidths=2.5,transform=ax.get_transform(wcs_cii))
ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=1.5,transform=ax.get_transform(wcs_cii))
ax.set_xlim(xlimo)
ax.set_ylim(ylimo)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
ax.set_rasterization_zorder(10)
ax.add_patch(Ellipse((x_texto,y_texto), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='k', fc='None',lw=2.0)) #plot the beam
ax.add_patch(Ellipse((x_texto,y_texto), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
ax.add_patch(Ellipse((x_texto,y_texto), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
ax.plot([x_texto_sb,x_texto_sb+sb_kpc_pix],[y_texto,y_texto],'-',color='k',lw=3)
ax.text(np.mean([x_texto_sb,x_texto_sb+sb_kpc_pix]),y_texto*1.01,sb_kpc_text,color='k',ha='center',va='bottom')
plt.savefig('../Plots/Composite_Maps/M82_HI_CO_CII_fullHIFOV.pdf',dpi=300,metadata={'Creator':this_script})
ax.set_xlim(xlim)
ax.set_ylim(ylim)
c0=ax.contour(cii,origin='lower',levels=levels_cii,colors='w',linewidths=3.5,transform=ax.get_transform(wcs_cii))
c1=ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=2.5,transform=ax.get_transform(wcs_cii))
b2=ax.add_patch(Ellipse((x_text,y_text), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='k', fc='None',lw=2.0)) #plot the beam
b1=ax.add_patch(Ellipse((x_text,y_text), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
b3=ax.add_patch(Ellipse((x_text,y_text), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
b1.set_path_effects([pe.Stroke(linewidth=4.0, foreground='w'),pe.Normal()])
b3.set_path_effects([pe.Stroke(linewidth=3.0, foreground='w'),pe.Normal()])
s1,=ax.plot([x_text_sb,x_text_sb+sb_pc_pix],[y_text-bmaj_cii/arcsec2pix,y_text-bmaj_cii/arcsec2pix],'-',color='w',lw=3)
s2=ax.text(np.mean([x_text_sb,x_text_sb+sb_pc_pix]),(y_text-bmaj_cii/arcsec2pix)*1.001,sb_pc_text,color='w',ha='center',va='bottom')
c, = ax.plot(center_wcs_hi[0],center_wcs_hi[1],'wx',markersize=10,mew=3.)
plt.savefig('../Plots/Composite_Maps/M82_HI_CO_CII.pdf',dpi=300,metadata={'Creator':this_script})

lw = 2.5
dlw = 1.5
#ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec='w',fc='None',linewidth=lw+dlw,zorder=10))
r=ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec=outline_color_cii,fc='None',linewidth=lw,zorder=10))
r.set_path_effects([pe.Stroke(linewidth=lw+dlw, foreground='w'),pe.Normal()])
rl = []
for i in range(len(fp)):
	this_fp = fp[i]
	this_fp.visual['linewidth'] = lw+dlw
	this_fp.visual['color'] = 'w'
	this_fp.to_pixel(wcs_hi).plot()
	this_fp.visual['linewidth'] = lw
	this_fp.visual['color'] = outline_color_cii
	this_fp.to_pixel(wcs_hi).plot()
	rrll = ax.text(this_fp.to_pixel(wcs_hi).center.x,this_fp.to_pixel(wcs_hi).center.y,str(reg_order[i]),
		color=outline_color_cii,ha='center',va='center',fontsize=fsz)
	rrll.set_path_effects([pe.Stroke(linewidth=2.0, foreground='w'),pe.Normal()])
	rl.append(rrll)
plt.savefig('../Plots/Composite_Maps/M82_HI_CO_CII_footprints.pdf',dpi=300,metadata={'Creator':this_script})
ax.set_xlim(xlimo)
ax.set_ylim(ylimo)
ax.add_patch(Ellipse((x_texto,y_texto), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='k', fc='None',lw=2.0)) #plot the beam
ax.add_patch(Ellipse((x_texto,y_texto), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
ax.add_patch(Ellipse((x_texto,y_texto), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
# ax.plot([x_texto_sb,x_texto_sb+sb_kpc_pix],[y_texto,y_texto],'-',color='k',lw=3)
# ax.text(np.mean([x_texto_sb,x_texto_sb+sb_kpc_pix]),y_texto*1.05,sb_kpc_text,color='k',ha='center',va='bottom')
b1.remove()
b2.remove()
b3.remove()
s1.remove()
s2.remove()
c.remove()
for c in c1.collections:
	c.remove()
for c in c0.collections:
	c.remove()
for rrll in rl:
	rrll.remove()
plt.savefig('../Plots/Composite_Maps/M82_HI_CO_CII_footprints_fullHIFOV.pdf',dpi=300,metadata={'Creator':this_script})



# fsz=14
# fig = plt.figure(1,figsize=(10,10))
# plt.clf()
# ax = plt.subplot(projection=wcs_hi)
# im=ax.imshow(co,origin='lower',cmap=cmap_co,vmin=vmin_co,vmax=vmax_co,transform=ax.get_transform(wcs_co))
# im.cmap.set_over('k')
# im.cmap.set_under('w')
# im.cmap.set_bad('w')
# ax.contourf(hi,levels=lev,origin='lower',cmap=hmap,alpha=1.0)
# #ax.contour(cii,origin='lower',levels=levels_cii,colors='w',linewidths=2.5,transform=ax.get_transform(wcs_cii))
# ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=1.5,transform=ax.get_transform(wcs_cii))
# ax.set_xlim(xlimo)
# ax.set_ylim(ylimo)
# ax.coords[0].set_major_formatter('hh:mm:ss.s')
# ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
# ax.coords[0].display_minor_ticks(True)
# ax.coords[1].display_minor_ticks(True)
# ax.coords[0].set_minor_frequency(4)
# ax.coords[1].set_minor_frequency(4)
# plt.xlabel('R.A. (J2000)')
# plt.ylabel('Decl. (J2000)',labelpad=-1)
# ax.set_rasterization_zorder(10)
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='None', fc='k',lw=2.0)) #plot the beam
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
# ax.plot([x_texto_sb,x_texto_sb+sb_kpc_pix],[y_texto,y_texto],'-',color='k',lw=3)
# ax.text(np.mean([x_texto_sb,x_texto_sb+sb_kpc_pix]),y_texto*1.05,sb_kpc_text,color='k',ha='center',va='bottom')
# plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_fullHIFOV.pdf',dpi=300,metadata={'Creator':this_script})
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# c0=ax.contour(cii,origin='lower',levels=levels_cii,colors='w',linewidths=3.5,transform=ax.get_transform(wcs_cii))
# c1=ax.contour(cii,origin='lower',levels=levels_cii,cmap=cmap_cii,linewidths=2.5,transform=ax.get_transform(wcs_cii))
# b1=ax.add_patch(Ellipse((x_text,y_text), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
# b2=ax.add_patch(Ellipse((x_text,y_text), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='None', fc='k',lw=2.0)) #plot the beam
# b3=ax.add_patch(Ellipse((x_text,y_text), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
# b1.set_path_effects([pe.Stroke(linewidth=4.0, foreground='w'),pe.Normal()])
# b3.set_path_effects([pe.Stroke(linewidth=3.0, foreground='w'),pe.Normal()])
# s1,=ax.plot([x_text_sb,x_text_sb+sb_pc_pix],[y_text,y_text],'-',color='w',lw=3)
# s2=ax.text(np.mean([x_text_sb,x_text_sb+sb_pc_pix]),y_text*1.005,sb_pc_text,color='w',ha='center',va='bottom')
# plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII.pdf',dpi=300,metadata={'Creator':this_script})


# ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec=outline_color_cii,fc='None',linewidth=1.5,zorder=10))
# for i in range(len(fp)):
# 	this_fp = fp[i]
# 	this_fp.visual['linewidth'] = 1.5
# 	this_fp.visual['color'] = outline_color_cii
# 	this_fp.to_pixel(wcs_hi).plot()
# plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_footprints.pdf',dpi=300,metadata={'Creator':this_script})
# ax.set_xlim(xlimo)
# ax.set_ylim(ylimo)
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_hi/arcsec2pix, bmaj_hi/arcsec2pix, bpa_hi, ec=outline_color_hi, fc='None',lw=3.0)) #plot the beam
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_co/arcsec2pix, bmaj_co/arcsec2pix, bpa_co, ec='None', fc='k',lw=2.0)) #plot the beam
# ax.add_patch(Ellipse((x_texto,y_texto), bmin_cii/arcsec2pix, bmaj_cii/arcsec2pix, bpa_cii, ec=outline_color_cii, fc='None',lw=2.0)) #plot the beam
# # ax.plot([x_texto_sb,x_texto_sb+sb_kpc_pix],[y_texto,y_texto],'-',color='k',lw=3)
# # ax.text(np.mean([x_texto_sb,x_texto_sb+sb_kpc_pix]),y_texto*1.05,sb_kpc_text,color='k',ha='center',va='bottom')
# b1.remove()
# b2.remove()
# b3.remove()
# s1.remove()
# s2.remove()
# for c in c1.collections:
# 	c.remove()
# plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_footprints_fullHIFOV.pdf',dpi=300,metadata={'Creator':this_script})


#now add the spectra in an inset
# ax_in=ax.inset_axes([-0.03,0.01,0.5,0.5])
# spec = img.imread('../Plots/Outflow_Spectra/M82_CII_Outflow1_sky_10kms.png')
# ax_in.imshow(spec)
# ax_in.tick_params(axis='both',bottom=False,left=False,labelbottom=False,labelleft=False)
# for pos in ['bottom','top','left','right']:
# 	ax_in.spines[pos].set_color(outline_color_cii)
# 	ax_in.spines[pos].set_linewidth(2)
# plt.savefig('../Plots/Composite_Maps/M82_CO_HI_CII_Spectra_fullHIFOV.pdf',dpi=300,metadata={'Creator':this_script})




plt.close()


