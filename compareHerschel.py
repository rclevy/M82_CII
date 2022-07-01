#compare the SOFIA and Herschel [CII] data

this_script = __file__

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.gridspec as gridspec
import upGREATUtils
import pandas as pd
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 24

# def fmt(x,pos):
# 	a, b = '{:.0e}'.format(x).split('e')
# 	b = int(b)
# 	if (b==0):
# 		out = r'0'
# 	else:
# 		out = r'${} \times 10^{{{}}}$'.format(a, b)
# 	return out


def pacs_Jy2K(im_Jypix,hdr):
	#convert from Jy/pixel to Jy/beam
	bmaj = 14. #arcsec
	bmin = 14. #arcsec
	nu = 1900.537 #GHz
	beam_area = np.pi*bmaj*bmin/(4*np.log(2)) #arcsec**2
	pix_area = np.abs(hdr['CDELT1']*hdr['CDELT2']*3600**2) #arcsec**2
	im_Jybeam = im_Jypix*beam_area/pix_area
	Tr_K = 1.222E6*im_Jybeam/(nu**2*bmaj*bmin) #Tr
	eta_mb = 0.76 #Fig 6 http://research.iac.es/proyecto/herschel/pacs/PACS_SPIE_6265-09.pdf
	im_K = Tr_K/eta_mb #convert to main beam temperature
	return im_Jybeam,Tr_K,im_K

def sofia_K2Jy(Tmb,hdr):
	eta_mb = 0.67
	eta_fss = 0.97
	bmaj = hdr['BMAJ']*3600.
	bmin = hdr['BMIN']*3600.
	nu = 1900.537 #GHz
	beam_area = np.pi*bmaj*bmin/(4*np.log(2)) #arcsec**2
	pix_area = np.abs(hdr['CDELT1']*hdr['CDELT2']*3600**2) #arcsec**2
	Tr = Tmb*eta_mb
	# Ta = Tr*eta_fss
	# S_Jypix = 971.*Ta #Jy/pixel
	S_Jybeam = Tr*nu**2*bmaj*bmin/1.222E6
	#S_Jybeam = S_Jypix*beam_area/pix_area
	return S_Jybeam,Tr

#open the PACS peak intensity map
pacs_peak = fits.open('../Data/Ancillary_Data/Herschel_PACS_158um_peak_14asGaussPSF_contsub.fits')
pacs_peak_hdr = pacs_peak[0].header
pacs_peak_wcs = WCS(pacs_peak_hdr)
pacs_peak_Jypix = pacs_peak[0].data #Jy/pixel
pacs_peak_Jybeam,_,_ = pacs_Jy2K(pacs_peak_Jypix,pacs_peak_hdr) #Jy/beam
# pacs_peak_levels = np.linspace(1.,5.,5) #K
pacs_peak_levels = np.linspace(500.,2000.,5) #Jy/beam
pacs_peak_cmap = sns.color_palette('light:salmon',as_cmap=True)


#open the SOFIA upgreat CII peak map
sofia_peak = fits.open('../Data/Disk_Map/Moments/M82_CII_map_peak_maskSNR5.fits')
sofia_peak_hdr = sofia_peak[0].header
sofia_peak_wcs = WCS(sofia_peak_hdr)
sofia_peak_Jybeam,_ = sofia_K2Jy(sofia_peak[0].data,sofia_peak_hdr)
sofia_peak_cmap = ListedColormap(sns.cubehelix_palette(start=0.6,rot=-0.65,light=0.9,dark=0.2,n_colors=256))
sofia_peak_vmin = 0.


#open the PACS mom0 map
pacs_mom0 = fits.open('../Data/Ancillary_Data/Herschel_PACS_158um_intinten_14asGaussPSF_contsub.fits')
pacs_mom0_hdr = pacs_mom0[0].header
pacs_mom0_wcs = WCS(pacs_mom0_hdr)
pacs_mom0_Jypix = pacs_mom0[0].data #Jy/pixel km/s
pacs_mom0_Jybeam,_,_ = pacs_Jy2K(pacs_mom0_Jypix,pacs_mom0_hdr) #Jy/beam km/s
pacs_mom0_levels = np.logspace(4.5,np.log10(300000),5)
pacs_mom0_cmap = sns.color_palette('light:salmon',as_cmap=True)


#open the SOFIA upgreat CII mom0 map
sofia_mom0 = fits.open('../Data/Disk_Map/Moments/M82_CII_map_intinten_maskSNR5.fits')
sofia_mom0_hdr = sofia_mom0[0].header
sofia_mom0_wcs = WCS(sofia_mom0_hdr)
sofia_mom0_Jybeam,_ = sofia_K2Jy(sofia_mom0[0].data,sofia_mom0_hdr) #Jy/beam
sofia_mom0_cmap = ListedColormap(sns.cubehelix_palette(start=1.1,rot=-0.65,light=0.9,dark=0.2,n_colors=256))
sofia_mom0_vmin = 0.

#crop to fully sampled map region
center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
ang_map = -20. #deg
fullsamp_x = 185.4 #arcsec
fullsamp_y = 65.4 #arcsec
arcsec2pix = pacs_peak_hdr['CDELT2']*3600
xo,yo = pacs_peak_wcs.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
xl = fullsamp_x/arcsec2pix
yl = fullsamp_y/arcsec2pix
xx = np.array([np.ceil(-xl/2),np.floor(xl/2)]).astype(int)
yy = np.array([np.ceil(-yl/2),np.floor(yl/2)]).astype(int)
xlim = (xx*np.cos(np.radians(ang_map))-yy*np.sin(np.radians(ang_map)))+xo
ylim = np.flip((xx*np.sin(np.radians(ang_map))-yy*np.cos(np.radians(ang_map)))+yo)

#add beam and scale bar
x_text = xlim[0]+0.05*(xlim[1]-xlim[0])
y_text = ylim[0]+0.075*(ylim[1]-ylim[0])

fsz=14
fig = plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=pacs_peak_wcs)
im=ax.imshow(sofia_peak_Jybeam/1E3,origin='lower',cmap=sofia_peak_cmap,vmin=sofia_peak_vmin,transform=ax.get_transform(sofia_peak_wcs))
con=ax.contour(pacs_peak_Jybeam/1E3,origin='lower',levels=pacs_peak_levels/1E3,cmap=pacs_peak_cmap,linewidths=2.5)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cb = plt.colorbar(im,shrink=0.7)
cb.add_lines(con)
cb.set_label('[CII] I$_{\mathrm{peak}}$ ($\\times 10^3$ Jy beam$^{-1}$)')
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
# b1=ax.add_patch(Ellipse((x_text,y_text), sofia_peak_bmaj/arcsec2pix, sofia_peak_bmin/arcsec2pix, sofia_peak_bpa, ec=sofia_peak_cmap(150), fc='None', lw=2.))
# b2=ax.add_patch(Ellipse((x_text,y_text), pacs_peak_bmaj/arcsec2pix, pacs_peak_bmin/arcsec2pix, pacs_peak_bpa, ec='k', fc='None', lw=2.))
plt.savefig('../Plots/Center_Maps/SOFIA_PACS_peak.pdf',bbox_inches='tight',metadata={'Creator':this_script})
plt.close()

fig = plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=pacs_mom0_wcs)
im=ax.imshow(sofia_mom0_Jybeam/1E5,origin='lower',cmap=sofia_mom0_cmap,vmin=sofia_mom0_vmin,transform=ax.get_transform(sofia_mom0_wcs))
con=ax.contour(pacs_mom0_Jybeam/1E5,origin='lower',levels=pacs_mom0_levels/1E5,cmap=pacs_mom0_cmap,linewidths=2.5)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cb = plt.colorbar(im,shrink=0.7)#,format=ticker.FuncFormatter(fmt))
cb.add_lines(con)
cb.set_label('[CII] I$_{\mathrm{int}}$ ($\\times10^5$ Jy beam$^{-1}$ km s$^{-1}$)')
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
plt.xlabel('R.A. (J2000)')
plt.ylabel('Decl. (J2000)',labelpad=-1)
# b1=ax.add_patch(Ellipse((x_text,y_text), sofia_peak_bmaj/arcsec2pix, sofia_peak_bmin/arcsec2pix, sofia_peak_bpa, ec=sofia_peak_cmap(150), fc='None', lw=2.))
# b2=ax.add_patch(Ellipse((x_text,y_text), pacs_peak_bmaj/arcsec2pix, pacs_peak_bmin/arcsec2pix, pacs_peak_bpa, ec='k', fc='None', lw=2.))
plt.savefig('../Plots/Center_Maps/SOFIA_PACS_intinten.pdf',bbox_inches='tight',metadata={'Creator':this_script})
plt.close()



#now compare the spectra in the outflow
#based on pltCIICOHIspectra.py
plt.rcParams['font.size'] = 14

#load some properties of M82
gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
distance = 3.63*u.Mpc

n_pixels = 2 #pixels per pointing
n_rows = 1
n_cols = int(np.floor(n_pixels/n_rows)*n_rows)

#set up plotting colors, same as in pltCompositeImage.py
#sofia_color = sns.color_palette('crest',as_cmap=True)(150)
sofia_color = sns.color_palette('crest',as_cmap=True)(150)
pacs_color = 'tomato'

pix_order_labels =  np.array([1,6])

fig = plt.figure(1,figsize=(8*2/3,8/3))
plt.clf()
gs = gridspec.GridSpec(n_rows,n_cols)

start_cols = [0,1]
i=0

for j in range(n_pixels):
	this_row = 0

	#load the CII data from the fits file
	cii_spec,cii_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'.fits')
	#clip bad data at the edge of the band
	cii_spec[cii_vel>425.]=np.nan

	cii_vel = np.flip(cii_vel)

	cii_spec_Jybeam,_ = sofia_K2Jy(np.flip(cii_spec),sofia_peak_hdr)
	cii_dv = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s



	#load the PACS data
	pacs_dat = pd.read_csv('../Data/Outflow_Pointings/PACS/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'_smo.csv')
	pacs_vel = pacs_dat['Velocity_kms'].values
	pacs_spec_Jypix = pacs_dat['Intensity_Jy'].values
	pacs_spec_Jybeam,_,_ = pacs_Jy2K(pacs_spec_Jypix,pacs_peak_hdr)

	# pacs_sf = np.nanmax(cii_spec)/np.nanmax(pacs_spec)
	# pacs_spec_norm = pacs_spec*pacs_sf

	#get distance of this pixel from the GC
	_,dist_pc,_ = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)

	#plot the data
	ax = plt.subplot(gs[this_row,start_cols[j]])
	c0=ax.fill_between(cii_vel,cii_spec_Jybeam,color=sofia_color,alpha=0.5,step='pre')
	c1,=ax.step(cii_vel,cii_spec_Jybeam,color=sofia_color,lw=1.5,label='SOFIA')
	c2,=ax.step(pacs_vel,pacs_spec_Jybeam,color=pacs_color,lw=1.75,label='PACS')
	ax.text(0.05,0.95,pix_order_labels[j],ha='center',va='top',transform=ax.transAxes)
	ax.text(0.975,0.95,'%.2f kpc' %np.round(dist_pc.value/1E3,2),ha='right',va='top',transform=ax.transAxes)


	# if j == n_pixels-1:
	if j == 1:
		leg = plt.legend([(c0,c1),c2],[c1.get_label(),c2.get_label()],
			handlelength=1.25,fontsize=plt.rcParams['font.size']-2,loc='right',bbox_to_anchor=(1.55,0.5,0.2,0.2))


	ax.set_xlim(-150,450)
	#ax.set_ylim(-0.05,0.25)
	ax.minorticks_on()

	if j !=0:
		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])
	else:
		ax.set_xlabel('V$_{\mathrm{LSRK}}$ (km s$^{-1}$)')
		ax.set_ylabel('I$_{\mathrm{[CII]}}$ (Jy beam$^{-1}$)')
plt.savefig('../Plots/Outflow_Spectra/M82_CII_SOFIA_PACS_Outflow'+str(i+1)+'.pdf',bbox_inches='tight',metadata={'Creator':this_script})





