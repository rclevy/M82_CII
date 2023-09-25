#calculate some properties of the [CII] in the disk of M82

this_script = __file__

import numpy as np
from astropy.io import fits

#########################################
# first measure the C+ mass in the disk #
#########################################

from calcColumnDensity import B_T,SigRuln,MCp_from_NCp

def NCp_from_ICII(I_CII,B):
	return I_CII/(B*3.3E-16) #cm-2

#define some fixed values
d = 3.63E6 #pc
T = 100. #K, T_CNM
n_CNM = 100. #cm^-3
dv = 10. #km/s
f_CNM = 0.5

#open the masked CII integrated intensity map
I_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_maskSNR2.fits')
hdr_I_CII = I_CII[0].header
pix_size_arcsec = I_CII[0].header['CDELT2']*3600
A_pix_pc2 = (pix_size_arcsec/206265*d)**2 #pc^2
A_pix_cm2 = A_pix_pc2*(3.086E18)**2 #cm^2
I_CII = I_CII[0].data
I_CII_SNR2 = I_CII.copy()
eI_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_emom0_maskSNR2.fits')[0].data

#now calculate the C+ column density from the integrated intensity map
N_Cp = NCp_from_ICII(I_CII,B_T(T,SigRuln(T,n_CNM))) #cm^-2
eN_Cp = NCp_from_ICII(eI_CII,B_T(T,SigRuln(T,n_CNM))) #cm^-2

#now calculate the mass by multiplying by the pixel area and mass of a carbon atom
M_Cp = MCp_from_NCp(N_Cp,A_pix_cm2) #Msun
eM_Cp = MCp_from_NCp(eN_Cp,A_pix_cm2) #Msun

#find the total mass in the mapped region (that isn't masked out)
M_Cp_tot = np.nansum(M_Cp) #Msun
eM_Cp_tot = np.sqrt(np.nansum(eM_Cp**2))
print('The total C+ mass in the disk is %.2e +/- %.2e Msun' %(M_Cp_tot,eM_Cp_tot))

Iint_tot = np.nansum(I_CII)


#######################################
# measure the [CII] rotation velocity #
#######################################

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
import seaborn as sns
from scipy.ndimage import rotate
from spectral_cube import SpectralCube
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def MarkerSizePixelScale(fig,arcsec2pix,dist):
	#scale the size of the points in a scatter plot to the pixel size
	pperpix = fig.dpi/72. #X points/pixel in the figure
	impix_pc = arcsec2pix/206265*dist #size of a pixel in the data in pc
	s = pperpix*impix_pc
	return s

def rot_map_horiz(arr):
	PA = -20.#deg
	arr[np.isnan(arr)==True]=-999
	arr_rot = rotate(arr,PA,reshape=False)	
	arr_rot[arr_rot<=0]=np.nan
	return arr_rot

def rot_cube_horiz(arr):
	PA = -20.#deg
	arr[np.isnan(arr)==True]=0
	arr_rot = rotate(arr,PA,axes=(2,1),reshape=False)	
	arr_rot[arr_rot<=-800]=np.nan
	arr_rot[arr_rot==0]=np.nan
	return arr_rot

cii_color = sns.color_palette('crest',as_cmap=True)(150)
co_color = '0.66'
cii_cmap = ListedColormap(sns.cubehelix_palette(start=0.6,rot=-0.65,light=1.0,dark=0.2,n_colors=256,reverse=False))
#co_cmap = ListedColormap(cm.gist_yarg(range(256)))
#co_cmap.set_over('k')
co_cmap = ListedColormap(sns.cubehelix_palette(start=0.6,rot=-0.65,light=0.9,dark=0.0,hue=0,n_colors=256,reverse=False))


#open the masked CII integrated intensity map
I_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits')[0].data
I_CII_mask = I_CII_SNR2.copy()
I_CII_mask[np.isnan(I_CII_SNR2)==False]=1.
fits.writeto('../Data/Disk_Map/Moments/M82_CII_map_maskSNR2.fits',data=I_CII_mask,header=hdr_I_CII,overwrite=True)
pix_size_arcsec = hdr_I_CII['CDELT2']*3600
A_pix_pc2 = (pix_size_arcsec/206265*d)**2 #pc^2
A_pix_cm2 = A_pix_pc2*(3.086E18)**2 #cm^2
I_CII[I_CII<=0]=np.nan
I_CII[I_CII > 800]=800.
I_CII_rot = rot_map_horiz(I_CII)
I_CII_rot = I_CII_rot#*I_CII_mask
V_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom1_matchedCIICO.fits')
hdr = V_CII[0].header
arcsec2pix = hdr['CDELT2']*3600
wcs = WCS(hdr)
V_CII = V_CII[0].data
V_CII_rot = rot_map_horiz(V_CII)#*I_CII_mask
Vsys_CO = 218.
Vsys_CII = 212.
Vmax_CII = 85. #EYEBALLED
#this cube is already oriented with the major axis || to x-axis
cen = SkyCoord('09h55m52.72s','69d40m45.8s')
RA_pix,Dec_pix = wcs.all_world2pix(cen.ra,cen.dec,1,ra_dec_order=True)
xax_off_pix = np.arange(0,V_CII.shape[1],1)-RA_pix
xax_off_arcsec = xax_off_pix*arcsec2pix
xax_off_pc = xax_off_arcsec/206265*d #pc
xax_off_kpc = xax_off_pc/1E3 #kpc

#now open the CO to compare
V_CO = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom1_matchedCIICO.fits')
hdr_CO = V_CO[0].header
wcs_CO = WCS(hdr_CO)
arcsec2pix_CO = hdr_CO['CDELT2']*3600
V_CO = V_CO[0].data
RA_pix_CO,Dec_pix_CO = wcs_CO.all_world2pix(cen.ra,cen.dec,1,ra_dec_order=True)
#rotate the CO so that the major axis is horizontal
V_CO_rot = rot_map_horiz(V_CO)
#clean up
V_CO_rot[V_CO_rot>Vsys_CO+150]=np.nan
V_CO_rot[V_CO_rot<Vsys_CO-150]=np.nan
#crop to fully sampled region of CII (3'x1')
V_CO_rot[0:int(Dec_pix_CO-30/arcsec2pix_CO),:]=np.nan
V_CO_rot[int(Dec_pix_CO+30/arcsec2pix_CO):,:]=np.nan
V_CO_rot[:,0:int(RA_pix_CO-90/arcsec2pix_CO)]=np.nan
V_CO_rot[:,int(RA_pix_CO+90/arcsec2pix_CO):]=np.nan
xax_off_pix_CO = np.arange(0,V_CO_rot.shape[1],1)-RA_pix_CO
xax_off_arcsec_CO = xax_off_pix_CO*arcsec2pix_CO
xax_off_pc_CO = xax_off_arcsec_CO/206265*d #pc
xax_off_kpc_CO = xax_off_pc_CO/1E3 #kpc
I_CO = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits')[0].data
I_CO_rot = rot_map_horiz(I_CO)
I_CO_rot[I_CO_rot>800]=800
Vmax_CO = 85.


# #make a position-velocity diagram
# fig=plt.figure(1)
# plt.clf()
# for i in range(V_CO_rot.shape[0]):
# 	plt.scatter(xax_off_kpc_CO,V_CO_rot[i,:]-Vsys_CO,s=20,marker='o',
# 		alpha=0.75,c=I_CO_rot[i,:],cmap=co_cmap,zorder=3)
# plt.axhline(Vmax_CO,color=co_color,lw=1.5)
# plt.axhline(-Vmax_CO,color=co_color,lw=1.5)
# for i in range(V_CII.shape[0]):
# 	plt.scatter(xax_off_kpc,V_CII_rot[i,:]-Vsys_CII,s=MarkerSizePixelScale(fig,arcsec2pix*1.2,d),marker='_',
# 		alpha=0.75,lw=8,c=I_CII_rot[i,:],cmap=cii_cmap,zorder=2)
# plt.axhline(Vmax_CII,color=cii_color,lw=1.5,linestyle='--')
# plt.axhline(-Vmax_CII,color=cii_color,lw=1.5,linestyle='--')
# # plt.axhline(0,color='gray',zorder=1,alpha=0.5,lw=0.75)
# # plt.axvline(0,color='gray',zorder=1,alpha=0.5,lw=0.75)
# plt.grid(which='major',axis='both',alpha=0.2)
# plt.xlabel('Offset from Center along Major Axis (kpc)')
# plt.ylabel('Velocity (km s$^{-1})$')
# plt.minorticks_on()
# plt.xlim(-1.5,1.5)
# plt.ylim(-120,120)
# #make colorbars for the CII and CO data
# I_CII_rot[0,0]=0.
# I_CO_rot[0,0]=0.
# ax_cii = fig.add_axes([0.92,0.125,0.025,0.75])
# norm_cii = Normalize(vmin=0,vmax=800)
# cb_cii = ColorbarBase(ax_cii,cmap=cii_cmap,norm=norm_cii,
# 	orientation='vertical')
# cb_cii.ax.tick_params(labelsize=10)
# cb_cii.set_label('Integrated Intensity (K km s$^{-1}$)',fontsize=10)
# levels = np.arange(0,850,100)
# for i in range(len(levels)):
# 	this_col = co_cmap(int(256/len(levels)*i))
# 	cb_cii.ax.hlines(levels[i],0,levels[-1],colors=this_col,linewidth=2)
# plt.savefig('../Plots/Center_Maps/M82_CII_CO_PVdiagram_RCLversion.pdf',bbox_inches='tight',metadata={'Creator':this_script})


#make a more traditional PV diagram
cii_cube = fits.open('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits')
hdr = cii_cube[0].header
cii_cube = cii_cube[0].data
levels_cii = np.arange(0.25,3.5,0.25)
cii_cmap = ListedColormap(cm.Spectral_r(range(256)))
cii_cmap.set_under('w')
co_cube = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_matchedCIICO.fits')[0].data
#rotate the cube so the major axis is horizontal
cii_cube_rot = rot_cube_horiz(cii_cube.copy())
co_cube_rot = rot_cube_horiz(co_cube.copy())
#mask out noisy pixels
cii_rms = 0.09 #K
co_rms = 0.03 #K
cii_cube_rot[cii_cube_rot<cii_rms]=np.nan
co_cube_rot[co_cube_rot<co_rms]=np.nan
#collapse the cubes along the y-axis weighting by the intensity
cii_cube_rot[np.isnan(cii_cube_rot)==True]=1E-9
cii_pv = np.average(cii_cube_rot,axis=1,weights=cii_cube_rot)
co_cube_rot[np.isnan(co_cube_rot)==True]=1E-9
co_pv = np.average(co_cube_rot,axis=1,weights=co_cube_rot)

# cii_pv = np.roll(cii_pv,1,axis=1)
# co_pv = np.roll(co_pv,1,axis=1)

co_cmap = 'Greys_r'
levels_co = np.arange(0.25,3.25,0.5)
#now plot
plt.close('all')
fig=plt.figure(1)
plt.clf()
#im = plt.imshow(cii_pv,cmap=cii_cmap,vmin=0,vmax=3.5,aspect='auto')
im = plt.contourf(cii_pv,cmap=cii_cmap,levels=levels_cii,extend='both')
ax = plt.gca()
con = ax.contour(co_pv,levels=levels_co,cmap=co_cmap,extend='both',linewidths=2)
#relabel axes
x = xax_off_kpc.copy()
xl_kpc = 1.5
xmax = np.argmin(np.abs(x-xl_kpc))
xmin = np.argmin(np.abs(-xl_kpc-x))
xt = np.linspace(xmin,xmax,7)
xtl = np.linspace(-xl_kpc,xl_kpc,7)
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(xmin,xmax)
ax.set_xlabel('Offset from Center along Major Axis (kpc)')
vax = SpectralCube.read('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits').spectral_axis.value
Vsys = 210.
y = vax-Vsys
yl_kms = 200.
ymax = np.argmin(np.abs(y-yl_kms))
ymin = np.argmin(np.abs(-yl_kms-y))
yt = np.linspace(ymin,ymax,9)
ytl = np.linspace(-yl_kms,yl_kms,9).astype(int)
ax.set_yticks(yt)
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.set_yticklabels(ytl)
plt.ylim(ymin,ymax)
ax.set_ylabel('Velocity (km s$^{-1}$)')
plt.grid(which='major',axis='both',alpha=0.2)
# plt.axhline((ymin+ymax)/2,color='dimgrey',lw=0.75,alpha=0.5)
# plt.axvline((xmin+xmax)/2,color='dimgrey',lw=0.75,alpha=0.5)
#add colorbar
cb = plt.colorbar(im)
#cb.ax.tick_params(labelsize=10)
cb.set_label('Intensity (K)')#,fontsize=10)
cb.add_lines(con)
plt.savefig('../Plots/Center_Maps/M82_CII_CO_PVdiagram.pdf',bbox_inches='tight',metadata={'Creator':this_script})













# #don't think pvextractor is going to work because it works in gal lat and long!
# from pvextractor import PathFromCenter, extract_pv_slice
# import astropy.units as u
# from spectral_cube import SpectralCube
# cii_cube = SpectralCube.read('../Data/Disk_Map/M82_CII_map_fullysampled_matchedCIICO.fits')
# co_cube = SpectralCube.read('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_matchedCIICO.fits')
# pl = 3.1*u.arcmin
# pw = 1.1*u.arcmin
# pa = 200.*u.deg
# Nsamp = int(np.round((pl/(cii_cube.header['CDELT2']*u.deg).to(u.arcmin)).value))
# p = PathFromCenter(center=cen,length=pl,width=pw,angle=pa,sample=Nsamp)
# cii_pv = extract_pv_slice(cii_cube,p)
# co_pv = extract_pv_slice(co_cube,p)
# #cii_pv.header['CRPIX1']=int(cii_pv.header['NAXIS1']-1)/2

# fig=plt.figure(1)
# plt.clf()
# ax = plt.subplot(111,projection=WCS(cii_pv.header))
# im = plt.imshow(cii_pv.data,cmap=cii_cmap)
# #im = plt.imshow(co_pv.data,cmap=co_cmap)
# ax.set_aspect(0.25)
# ax.invert_yaxis()
# ax.coords[0].set_format_unit(u.arcsec)
# ax.coords[0].set_major_formatter('x')
# ax.coords[0].display_minor_ticks(True)
# ax.coords[1].set_format_unit(u.km/u.s)
# ax.coords[1].display_minor_ticks(True)
# #get the center pixel num
# crpix1 = int((cii_pv.header['NAXIS1']-1)/2)
# ax.set_xlabel('Offset from Center (")')
# ax.set_ylabel('Velocity (km s$^{-1}$)')
# ax.axvline(cii_pv.header['CRPIX1'],color='gray')
# ax.axhline(int(cii_pv.header['NAXIS2'])/2,color='gray')








# I_CII[0,0]=0.
# I_CO_rot[0,0]=0.
# ax_cii = fig.add_axes([0.92,0.125,0.025,0.75])
# norm_cii = Normalize(vmin=0,vmax=800)
# cb_cii = ColorbarBase(ax_cii,cmap=cii_cmap,norm=norm_cii,
# 	orientation='vertical')
# cb_cii.ax.tick_params(labelsize=8,tickdir='in',color='w')
# plt.setp(plt.getp(cb_cii.ax.axes, 'yticklabels'), color='w',
# 	rotation=90,verticalalignment='center',x=-0.25)
# cb_cii.set_label('[CII] Integrated Intensity (K km s$^{-1}$)',fontsize=10,color=cii_color)

# ax_co = fig.add_axes([0.99,0.125,0.025,0.75])
# norm_co = Normalize(vmin=0,vmax=800)
# cb_co = ColorbarBase(ax_co,cmap=co_cmap,norm=norm_co,
# 	orientation='vertical')
# cb_co.ax.tick_params(labelsize=8,tickdir='in',color='w')
# plt.setp(plt.getp(cb_co.ax.axes, 'yticklabels'), color='w',
# 	rotation=90,verticalalignment='center',x=-0.25)
# cb_co.set_label('CO Integrated Intensity (K km s$^{-1}$)',fontsize=10)



# #make a position-velocity diagram
# fig=plt.figure(1)
# plt.clf()
# for i in range(V_CO_rot.shape[0]):
# 	plt.scatter(xax_off_pc_CO,V_CO_rot[i,:]-Vsys,s=MarkerSizePixelScale(fig,arcsec2pix,d),marker='_',alpha=0.5,
# 		lw=8,c=I_CO_rot[i,:],cmap=co_cmap,zorder=2)
# plt.axhline(Vmax_CO,color=co_color,lw=1.5)
# plt.axhline(-Vmax_CO,color=co_color,lw=1.5)
# for i in range(V_CII.shape[0]):
# 	plt.scatter(xax_off_pc,V_CII[i,:]-Vsys,s=MarkerSizePixelScale(fig,arcsec2pix,d),marker='_',alpha=0.75,
# 		lw=9,ec='w',fc=None,cmap=cii_cmap,zorder=3)
# 	plt.scatter(xax_off_pc,V_CII[i,:]-Vsys,s=MarkerSizePixelScale(fig,arcsec2pix,d),marker='_',alpha=0.75,
# 		lw=8,c=I_CII[i,:],cmap=cii_cmap,zorder=3)
# plt.axhline(Vmax_CII,color=cii_color,lw=1.5)
# plt.axhline(-Vmax_CII,color=cii_color,lw=1.5)
# # plt.axhline(0,color='gray',zorder=1,alpha=0.5,lw=0.75)
# # plt.axvline(0,color='gray',zorder=1,alpha=0.5,lw=0.75)
# plt.grid(which='major',axis='both',alpha=0.2)
# plt.xlabel('Offset from Center along Major Axis (pc)')
# plt.ylabel('Velocity (km s$^{-1})$')
# plt.minorticks_on()
# plt.xlim(-1500,1500)
# plt.ylim(-150,150)

# #make colorbars for the CII and CO data
# I_CII[0,0]=0.
# I_CO_rot[0,0]=0.
# ax_cii = fig.add_axes([0.92,0.125,0.025,0.75])
# norm_cii = Normalize(vmin=np.nanmin(I_CII),vmax=np.nanmax(I_CII))
# cb_cii = ColorbarBase(ax_cii,cmap=cii_cmap,norm=norm_cii,
# 	orientation='vertical')
# cb_cii.ax.tick_params(labelsize=8,tickdir='in',color='w')
# plt.setp(plt.getp(cb_cii.ax.axes, 'yticklabels'), color='w',
# 	rotation=90,verticalalignment='center',x=-0.25)
# cb_cii.set_label('[CII] Integrated Intensity (K km s$^{-1}$)',fontsize=10,color=cii_color)

# ax_co = fig.add_axes([0.99,0.125,0.025,0.75])
# norm_co = Normalize(vmin=np.nanmin(I_CO_rot),vmax=np.nanmax(I_CO_rot))
# cb_co = ColorbarBase(ax_co,cmap=co_cmap,norm=norm_co,
# 	orientation='vertical')
# cb_co.ax.tick_params(labelsize=8,tickdir='in',color='w')
# plt.setp(plt.getp(cb_co.ax.axes, 'yticklabels'), color='w',
# 	rotation=90,verticalalignment='center',x=-0.25)
# cb_co.set_label('CO Integrated Intensity (K km s$^{-1}$)',fontsize=10)

# plt.savefig('../Plots/Center_Maps/M82_CII_CO_PVdiagram.pdf',bbox_inches='tight',metadata={'Creator':this_script})


# import sys
# sys.path.insert(0, '../../../ALMA_Dwarfs_CO/Code')
# from FitTiltedRings import fit_tilted_rings
# from astropy.coordinates import SkyCoord
# from astropy.wcs import WCS
# from matplotlib.patches import Ellipse

# V_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom1_maskSNR2.fits')
# hdr = V_CII[0].header
# wcs = WCS(hdr)
# V_CII = V_CII[0].data
# eV_CII = fits.open('../Data/Disk_Map/Moments/M82_CII_map_emom1_maskSNR2.fits')[0].data
# cen = SkyCoord('09h55m52.72s','69d40m45.8s')
# RA = cen.ra.value
# Dec = cen.dec.value
# PA = 90. #deg
# inc = 80. #deg
# Vsys = 210. #km/s
# save_dir = '../Plots/Center_Maps/RotationCurve/'
# R,eR,Vrot,eVrot,Vrad,eVrad,dVsys,edVsys,chisq,chisqr,rms=fit_tilted_rings('M82',hdr,V_CII,eV_CII,RA,Dec,PA,inc,Vsys,save_dir=save_dir,min_pixels_per_ring=False)

# # #open the CO data to use the wcs
# # hdr_co = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_peak.fits')[0].header
# # wcs_co = WCS(hdr_co)
# RA_pix,Dec_pix = wcs.all_world2pix(RA,Dec,1,ra_dec_order=True)


# #plot the velocity field
# plt.figure(1)
# plt.clf()
# ax=plt.subplot(projection=wcs_co)
# im=ax.imshow(V_CII-Vsys,origin='lower',cmap='RdYlBu_r',vmin=-100,vmax=100)
# ax.plot(RA_pix,Dec_pix,'kx')
# cb=plt.colorbar(im)
# cb.set_label('V$_{\mathrm{[CII]}}$-V$_{\mathrm{sys}}$ (km s$^{-1}$)')
# ax.coords[0].set_major_formatter('hh:mm:ss.s')
# ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
# ax.coords[0].display_minor_ticks(True)
# ax.coords[1].display_minor_ticks(True)
# ax.coords[0].set_minor_frequency(4)
# ax.coords[1].set_minor_frequency(4)
# ax.coords[0].set_ticklabel(exclude_overlapping=True)
# ax.coords[1].set_ticklabel(exclude_overlapping=True)
# ax.set_xlabel('R.A. (J2000)')
# ax.set_ylabel('Decl. (J2000)')
# plt.text(0.03,0.97,gal_name,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,fontsize=plt.rcParams['font.size']+2)
# #plot the rings over the velcity field
# R_pix = R[1:]/3600/hdr['CDELT2'] #ring center
# R_shift = np.concatenate([R_pix[1:],np.array([np.inf])])
# R_outer = np.nanmean([R_pix,R_shift],axis=0)[0:-1]
# [ax.add_patch(Ellipse((RA_pix,Dec_pix),2*ro,2*ro*np.cos(np.radians(inc)),PA+90, ec='k',fc='None',zorder=5)) for ro in R_outer]

