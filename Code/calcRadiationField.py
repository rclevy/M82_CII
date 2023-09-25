#try to back out the radiation field in the outflow
#using Figure 9 of Kaufman+1999

this_script = __file__

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Circle, Rectangle)
from astropy.wcs import WCS
import upGREATUtils
from regions import Regions
from scipy.optimize import curve_fit
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 24

#first get the ratios of the CII/CO integrated intensities
# [CII] in units of erg/s/cm^2
# CO in units of erg/s/cm^2/sr


#these conversions are specific to [CII]!
# def Kkms_to_ergscm2_cii(Iint,Apix_sr):
# #	return Iint/(1.43E5)*Apix_sr #Goldsmith+2012

# # def ergscm2_to_Kkms_cii(Iint,Apix_sr):
# # 	return Iint*1.43E5/Apix_sr #Goldsmith+2012

# #these conversion for specific to CO(1-0)
# def Kkms_to_ergscm2sr_co(Iint,Apix_sr):
# 	return 2.17E-15*Iint/Apix_sr

def Kkms_to_ergscm2sr(Iint,nu_Hz):
	k = 1.38E-16 #cgs
	c = 2.9979E10 #cgs
	return 2*np.pi*k*nu_Hz**3/(4*np.log(2)*c**2*(c/1E5))*Iint

def ratio_ergscm2sr_to_Kkms(x):
	return (115E9)/(1.9E12)*x

def ratio_Kkms_to_ergscm2sr(x):
	return (1.9E12)/(115E9)*x


#open the disk maps
#use the ones that are from the matched CO and CII data
#Apix_sr = (7./206265)**2
Iint_cii_disk_Kkms = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits')
Iint_co_disk_Kkms = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits')
Iint_cii_disk = Kkms_to_ergscm2sr(Iint_cii_disk_Kkms[0].data,1.9E12)#erg/s/cm^2/sr
Iint_co_disk = Kkms_to_ergscm2sr(Iint_co_disk_Kkms[0].data,115E9)#erg/s/cm^2/sr
hdr_matched = Iint_co_disk_Kkms[0].header
wcs_matched = WCS(hdr_matched)
wcs_matched.wcs.crpix[0] = wcs_matched.wcs.crpix[0]-0.5
wcs_matched.wcs.crpix[1] = wcs_matched.wcs.crpix[1]-0.5
Iint_cii_disk = np.roll(np.roll(Iint_cii_disk,0,axis=0),-1,axis=-1)

ratio_cii_co_disk = Iint_cii_disk/Iint_co_disk
#mask to SNR>2
mask_SNR2 = fits.open('../Data/Disk_Map/Moments/M82_CII_map_maskSNR2.fits')[0].data
ratio_cii_co_disk*=mask_SNR2


#open the outflow values
dpix_arcsec = 14.1
#Apix_sr = np.pi*(dpix_arcsec/2)**2/(206265**2)
moms_cii = pd.read_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams.csv')
pix_num = moms_cii['Pixel Number'].values
Iints_cii_Kkms = moms_cii['Mom0'].values #K km/s
Iints_cii = Kkms_to_ergscm2sr(Iints_cii_Kkms,1.9E12) #erg/s/cm^2/sr

ratios_cii_co = np.zeros_like(Iints_cii)

for i in range(len(pix_num)):
	Iint_cii = Iints_cii[pix_num[i]]

	#open the CO int inten
	moms_co = pd.read_csv('../Data/Outflow_Pointings/CO_TP_Moments/outflow1_pix'+str(pix_num[i])+'_momentparams.csv')
	Iint_co_Kkms = moms_co['Mom0'].values[0] #K km/s
	#convert CO from K km/s to erg/s/cm^2/sr
	Iint_co = Kkms_to_ergscm2sr(Iint_co_Kkms,115E9)#erg/s/cm^2/sr

	ratios_cii_co[i] = Iint_cii/Iint_co


#now plot the disk map and outflows like for the moment maps
#based on pltCIImap.py
def MarkerSizeBeamScale(fig,bmaj,arcsec2pix):
	#scale the size of the points in a scatter plot to the beam size
	pperpix = fig.dpi/72. #X points/pixel
	bmaj_pix = bmaj/arcsec2pix
	bmaj_points = bmaj_pix*pperpix
	s = bmaj_points**2
	return s

#open the original CO data for plotting only
hdr = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_peak.fits')[0].header
wcs = WCS(hdr)
arcsec2pix = hdr['CDELT2']*3600

#get info to draw fully-sampled map region
center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
ang_map = -20. #deg, but the cube is already rotated by this much
fullsamp_x = 185.4 #arcsec
fullsamp_y = 65.4 #arcsec
xo,yo = wcs.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
xl = fullsamp_x/arcsec2pix
yl = fullsamp_y/arcsec2pix
xor = ((-xl/2)*np.cos(np.radians(ang_map))-(-yl/2)*np.sin(np.radians(ang_map)))+xo
yor = ((-xl/2)*np.sin(np.radians(ang_map))+(-yl/2)*np.cos(np.radians(ang_map)))+yo
xx = np.array([np.ceil(-xl/2),np.floor(xl/2)]).astype(int)
yy = np.array([np.ceil(-yl/2),np.floor(yl/2)]).astype(int)
xlim = (xx*np.cos(np.radians(ang_map))-yy*np.sin(np.radians(ang_map)))+xo
ylim = np.flip((xx*np.sin(np.radians(ang_map))-yy*np.cos(np.radians(ang_map)))+yo)


cmap = 'Spectral_r'
# vmin = 2.5
# vmax = 4.0
vmin = 0.
vmax = 7500.

cb_lim_Kkms = [ratio_ergscm2sr_to_Kkms(vmin),ratio_ergscm2sr_to_Kkms(vmax)]
cb_tl_Kkms = np.arange(cb_lim_Kkms[0],np.round(cb_lim_Kkms[1]),50).astype(int)
cb_t_Kkms = ratio_Kkms_to_ergscm2sr(cb_tl_Kkms)


fig=plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs)
#im = ax.imshow(np.log10(ratio_cii_co_disk),cmap=cmap,origin='lower',vmin=vmin,vmax=vmax,transform=ax.get_transform(wcs_matched))
im = ax.imshow(ratio_cii_co_disk,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax,transform=ax.get_transform(wcs_matched))
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.set_xlabel('R.A. (J2000)')
ax.set_ylabel('Decl. (J2000)')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cb = plt.colorbar(im,pad=0.15)
cb.ax.minorticks_on()
#cb.set_label('log$_{10}$(I$_\mathrm{int}^\mathrm{[CII]}$/I$_\mathrm{int}^\mathrm{CO}$)')
cb.set_label('I$_\mathrm{int,erg~s^{-1}~cm^{-2}~sr^{-1}}^\mathrm{[CII]}$/I$_\mathrm{int,erg~s^{-1}~cm^{-2}~sr^{-1}}^\mathrm{CO}$')
cb2 = cb.ax.secondary_yaxis('left',functions=(ratio_ergscm2sr_to_Kkms,ratio_Kkms_to_ergscm2sr))
cb2.minorticks_on()
cb2.set_ylabel('I$_\mathrm{int,K~km~s^{-1}}^\mathrm{[CII]}$/I$_\mathrm{int,K~km~s^{-1}}^\mathrm{CO}$')
ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec='k',fc='None',linewidth=1.5,zorder=10))
c, = ax.plot(xo,yo,'kx',markersize=10,mew=3.)
#now add the corresponding outflow data
n_pixels = 7
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
center_str = center_sp.to_string('hmsdms',precision=4).replace(' ','')[0:-2]+'s'
array_rot = 70. #deg
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_str+'_Rot'+str(array_rot)+'.reg')
reg_order = [0,5,6,1,2,3,4]

for i in range(n_pixels):
	this_fp = fp[i]
	RA = this_fp.center.ra.value
	Dec = this_fp.center.dec.value
	RA_pix,Dec_pix = wcs.all_world2pix(RA,Dec,1,ra_dec_order=True)
	#rr = np.log10(ratios_cii_co[reg_order[i]])
	rr = ratios_cii_co[reg_order[i]]

	if np.isnan(rr)==False:
		ax.scatter(RA_pix,Dec_pix,s=MarkerSizeBeamScale(fig,dpix_arcsec/2,1/arcsec2pix),c=rr,cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='k',linewidths=1.)
		print('Pixel %i has ICII/ICO=%1.f' %(reg_order[i],rr))	
	else:
		ax.scatter(RA_pix,Dec_pix,s=MarkerSizeBeamScale(fig,dpix_arcsec/2,1/arcsec2pix),c='w',edgecolors='k',linewidths=1.)

bmaj_pix = dpix_arcsec/arcsec2pix
new_ymin = ylim[0]-5*bmaj_pix
ax.set_ylim(new_ymin)
plt.savefig('../Plots/Center_Maps/M82_ratio_CII_CO_outflow.pdf',bbox_inches='tight',metadata={'Creator':this_script})


#now open and plot the PDR plots from Kaufman+1999 Fig 9
K99 = pd.read_csv('../Data/Ancillary_Data/Kaufman1999_Fig9.csv')
logn = K99['log10_n_cm3'].values
logG = K99['log10_G_G0'].values
Iratio = K99['ICII_ICO'].values
logI = np.log10(Iratio)

#so let's assume some fixed density
this_logn = 4. #cm^-3

#pick out the values close to this
idx = np.where((np.round(logn,1)==this_logn-0.1) | (np.round(logn,1)==this_logn) | (np.round(logn,1)==this_logn+0.1))[0]

#fit with a powerlaw to interpolate
def ex(x,a,b):
	return a*np.exp(b*x)
p,_ = curve_fit(ex,logI[idx],logG[idx])


G_from_ratio_disk = ex(np.log10(ratio_cii_co_disk),*p)
hdr_G = hdr_matched.copy()
hdr_G['BUNIT'] = 'log10G0'
del(hdr_G['COMMENT'])
del(hdr_G['LINE'])
hdr_G['COMMENT']='Radiation field strength derived from CII and CO lines and Kaufman+1999 PDR model (see '+this_script+')'
fits.writeto('../Data/Disk_Map/M82_RadnField_map_matchedCIICO.fits',G_from_ratio_disk,header=hdr_G,overwrite=True)


# get uncertainties
dn = 0.1
this_elogn = 0.5
this_elogn_outflow = 2.0

eln = np.arange(this_logn-this_elogn,this_logn+this_elogn+dn,dn)
elno = np.arange(this_logn-this_elogn_outflow,this_logn+dn,dn)

eG_from_ratio_disk = np.zeros((G_from_ratio_disk.shape[0],G_from_ratio_disk.shape[1],len(eln)))

for i in range(len(eln)):
	eidx = np.where((np.round(logn,1)==eln[i]-0.1) | (np.round(logn,1)==eln[i]) | (np.round(logn,1)==eln[i]+0.1))[0]
	if len(eidx)>0:
		ep,_=curve_fit(ex,logI[eidx],logG[eidx])
		eG = ex(np.log10(ratio_cii_co_disk),*ep)
		eG_from_ratio_disk[:,:,i] = eG
	else:
		eG_from_ratio_disk[:,:,i] = np.nan

elG_from_ratio_disk = G_from_ratio_disk-np.nanmin(eG_from_ratio_disk,axis=2)
euG_from_ratio_disk = np.nanmax(eG_from_ratio_disk,axis=2)-G_from_ratio_disk

# vmin=0.25
# vmax=2.75
vmin=0.25
vmax=5.0
fig=plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs)
im = ax.imshow(G_from_ratio_disk,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax,transform=ax.get_transform(wcs_matched))
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.set_xlabel('R.A. (J2000)')
ax.set_ylabel('Decl. (J2000)')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cb = plt.colorbar(im)
cb.ax.minorticks_on()
cb.set_label('log$_{10}(G/G_0)$ for $n=10^%1.f$ cm$^{-3}$' %this_logn)
ax.add_patch(Rectangle((xor,yor),xl,yl,angle=ang_map,ec='k',fc='None',linewidth=1.5,zorder=10))
c, = ax.plot(xo,yo,'kx',markersize=10,mew=3.)
#now add the corresponding outflow data
n_pixels = 7
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
center_str = center_sp.to_string('hmsdms',precision=4).replace(' ','')[0:-2]+'s'
array_rot = 70. #deg
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_str+'_Rot'+str(array_rot)+'.reg')
reg_order = [0,5,6,1,2,3,4]
dist_pix = np.array([1.44,1.33,0.95,1.08,1.56,1.89,1.86])*1E3 #pc


G_from_ratio_outflow = np.zeros((n_pixels,))


for i in range(n_pixels):
	this_fp = fp[i]
	RA = this_fp.center.ra.value
	Dec = this_fp.center.dec.value
	RA_pix,Dec_pix = wcs.all_world2pix(RA,Dec,1,ra_dec_order=True)
	gg = ex(np.log10(ratios_cii_co[reg_order[i]]),*p)
	G_from_ratio_outflow[i] = gg

	if np.isnan(gg)==False:
		ax.scatter(RA_pix,Dec_pix,s=MarkerSizeBeamScale(fig,dpix_arcsec/2,1/arcsec2pix),c=gg,cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='k',linewidths=1.)
		print('Pixel %i has logG/G0=%.1f' %(reg_order[i],gg))
	else:
		ax.scatter(RA_pix,Dec_pix,s=MarkerSizeBeamScale(fig,dpix_arcsec/2,1/arcsec2pix),c='w',edgecolors='k',linewidths=1.)

bmaj_pix = dpix_arcsec/arcsec2pix
new_ymin = ylim[0]-5*bmaj_pix
ax.set_ylim(new_ymin)
plt.savefig('../Plots/Center_Maps/M82_RadnField_n'+str(int(this_logn))+'_CII_CO_outflow.pdf',bbox_inches='tight',metadata={'Creator':this_script})


eG_from_ratio_outflow = np.zeros((G_from_ratio_outflow.shape[0],len(elno)))
for i in range(len(elno)):
	eidx = np.where((np.round(logn,1)==elno[i]-0.1) | (np.round(logn,1)==elno[i]) | (np.round(logn,1)==elno[i]+0.1))[0]
	if len(eidx)>0:
		ep,_=curve_fit(ex,logI[eidx],logG[eidx])
		eG = ex(np.log10(ratios_cii_co[reg_order]),*ep)
		eG_from_ratio_outflow[:,i] = eG
	else:
		eG_from_ratio_outflow[:,i] = np.nan
elG_from_ratio_outflow = np.abs(G_from_ratio_outflow-np.nanmin(eG_from_ratio_outflow,axis=1))
euG_from_ratio_outflow = np.abs(np.nanmax(eG_from_ratio_outflow,axis=1)-G_from_ratio_outflow)

tab = {'log10G_G0':G_from_ratio_outflow,'el_log10G_G0':elG_from_ratio_outflow,'eu_log10G_G0':euG_from_ratio_outflow}
df = pd.DataFrame.from_dict(tab)
df.insert(0,'Pixel Number',reg_order,True)
df.sort_values('Pixel Number',inplace=True)
df.to_csv('../Data/Outflow_Pointings/outflow1_RadnField.csv',index=False)


#plot radiation field as a function of distance from the center
#also with z^-2 geometric scaling
gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
distance = 3.63*u.Mpc

#calculate the offset of every pixel in the map from the galaxy center
naxis1 = Iint_co_disk_Kkms[0].header['NAXIS1']
naxis2 = Iint_co_disk_Kkms[0].header['NAXIS2']
pix_sz_pc = Iint_co_disk_Kkms[0].header['CDELT2']*3600/206265*distance.value*1E6
x = np.arange(naxis1)
y = np.arange(naxis2)
xx,yy = np.meshgrid(x,y)
ras,decs = wcs_matched.all_pix2world(xx,yy,0)
pix_coords = SkyCoord(ras*u.deg,decs*u.deg)
pix_seps_arcsec = gal_center.separation(pix_coords).to(u.arcsec)
pix_seps_pc = (pix_seps_arcsec.value/206265*distance).to(u.pc)
xx = np.ravel(pix_seps_pc.value)
yy = np.ravel(G_from_ratio_disk)
elyy = np.ravel(elG_from_ratio_disk)
euyy = np.ravel(euG_from_ratio_disk)

#get values for representative error bars
ave_ex = np.nanmedian(pix_sz_pc/2)
ave_ely = np.nanmedian(elyy)
ave_euy = np.nanmedian(euyy)

#get data from Contursi+2013 Fig 20
#use No [CII] correction from South
C13_dist_arcsec = np.array([16.95986934, 25.4777397, 33.97485762, 42.54369414, 51.06154083, 59.66683473])
C13_dist_pc = C13_dist_arcsec/206265*distance.value*1E6
C13_G = np.array([548.0244717, 554.0489029, 313.1954479, 309.2200554, 175.9137772, 97.68763545])
C13_G_eu = np.array([827.461364, 826.4806729, 830.8035621, 461.2668631, 264.0087376, 163.5016179])-C13_G
C13_G_el = C13_G - np.array([278.0112972, 276.0042927, 157.9225287, 121.6202353, 88.70145854, 49.2570327])

x_geom = np.linspace(1,2100,500)
y_geom2 = 350*(x_geom/500.)**(-2)
y_geom1 = 3500*(x_geom/500.)**(-1)

#plot
fig=plt.figure(1,figsize=(14,10))
plt.clf()
l00=plt.scatter(xx,yy,s=200,c=yy,marker='s',cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='k',linewidths=1.,zorder=3,label='Disk (this work)')

l1 = plt.errorbar(dist_pix,G_from_ratio_outflow,xerr=dpix_arcsec/2/206265*distance.value*1E6,yerr=(elG_from_ratio_outflow,np.zeros_like(elG_from_ratio_outflow)),
	color='k',fmt=',',capsize=3,label='Outflow (this work)',zorder=2)
l11 = plt.scatter(dist_pix,G_from_ratio_outflow,s=300,c=G_from_ratio_outflow,marker='o',cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='k',linewidths=1.,zorder=3)

l2=plt.errorbar(C13_dist_pc,np.log10(C13_G),xerr=24./2/206265*distance.value*1E6,yerr=(C13_G_el/(C13_G*np.log(10)),C13_G_eu/(C13_G*np.log(10))),
	markersize=20,fmt='kd',capsize=3,label='Contursi et al. (2013)')
l3,=plt.plot(x_geom,np.log10(y_geom2),'-',lw=2.0,color='gray',label='r$^{-2}$ Geometric Dilution')
#l4,=plt.plot(x_geom,np.log10(y_geom1),'-',lw=1.25,color='gray',label='r$^{-3}$ Geometric Dilution')

plt.errorbar(np.array([[100.]]), np.array([[0.8]]), markersize=15,xerr=ave_ex,yerr=np.array([[ave_ely,ave_euy]]).T,fmt='s',color='k',mfc='w',lw=1,capsize=3,zorder=2)

plt.minorticks_on()
plt.xlabel('Distance from Galaxy Center (pc)')
plt.ylabel('log$_{10}(G/G_0)$ for $n=10^%1.f$ cm$^{-3}$' %this_logn)
plt.xlim(0.,2100.)
plt.ylim(0,5)
plt.axvline(500,color='gray',linestyle='--',zorder=1)
plt.legend([(l00),(l1,l11),l2,l3],[l00.get_label(),l1.get_label(),l2.get_label(),l3.get_label()],fontsize=plt.rcParams['font.size']-4)
plt.savefig('../Plots/Center_Maps/M82_RadnField_n'+str(int(this_logn))+'_with_distance.pdf',bbox_inches='tight',metadata={'Creator':this_script})

plt.close('all')


