this_script = __file__

#plot CII, CO, and HI spectra for each of the single pointings
#based on pltCIIspectra.py

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import upGREATUtils
import seaborn as sns
from astropy.convolution import Box1DKernel,convolve

plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

#load some properties of M82
gal_center = SkyCoord('09h55m52.72s 69d40m45.8s')
distance = 3.5*u.Mpc

n_pointings = 2
n_pixels = 7 #pixels per pointing
n_rows = int(np.ceil(np.sqrt(n_pixels)))
n_cols = int(np.floor(n_pixels/n_rows)*n_rows)

#set up plotting colors, same as in pltCompositeImage.py
cii_color = sns.color_palette('crest',as_cmap=True)(150)
co_color = 'k'
hi_color = sns.color_palette('flare',as_cmap=True)(150)

#get CO and HI velocity resolutions for smoothing
co_dv = fits.open('../Data/Ancillary_Data/M82-NOEMA-30m-mask-merged.clean.fits')[0].header['CDELT3']/1E3 #km/s
hi_dv = -1*fits.open('../Data/Ancillary_Data/m82_hi_24as_feathered_nhi.fits')[0].header['CDELT3']/1E3 #km/s

for i in range(n_pointings):

	#get "sky" pixel order labels
	if i==0:
		pix_order_labels =  np.array([1,6,2,0,5,3,4])
	else:
		pix_order_labels =  np.array([1,2,6,0,3,5,4])

	fig = plt.figure(1,figsize=(8,8))
	plt.clf()
	gs = gridspec.GridSpec(n_rows,n_cols)

	start_cols = [1,3,0,2,4,1,3]

	for j in range(n_pixels):
		this_row = int(np.floor((j+1)/n_rows))

		#load the CII data from the fits file
		cii_spec,cii_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'.fits')
		cii_vel = np.flip(cii_vel)
		cii_spec = np.flip(cii_spec)
		#cii_spec_norm = cii_spec/np.nanmax(cii_spec)
		cii_dv = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s

		#load the CO and HI data
		co_dat = pd.read_csv('../Data/Outflow_Pointings/CO/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'.csv')
		co_vel = co_dat['Velocity_kms'].values
		co_spec = co_dat['Intensity_K'].values
		co_sf = np.nanmax(cii_spec)/np.nanmax(co_spec)
		co_spec_norm = co_spec*co_sf

		hi_dat = pd.read_csv('../Data/Outflow_Pointings/HI/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'.csv')
		hi_vel = np.flip(hi_dat['Velocity_kms'].values)
		hi_spec_cm2 = np.flip(hi_dat['Intensity_cm-2'].values)
		hi_spec = hi_spec_cm2/(1.823E18) #Martini+2018, K
		hi_sf = np.nanmax(cii_spec)/np.nanmax(hi_spec)
		hi_spec_norm = hi_spec*hi_sf

		#smooth the CO and HI data to the CII vel res
		co_kernel = Box1DKernel(cii_dv/co_dv)
		hi_kernel = Box1DKernel(cii_dv/hi_dv)
		co_vel_smo = convolve(co_vel,co_kernel)
		co_spec_smo = convolve(co_spec_norm,co_kernel)
		hi_vel_smo = convolve(hi_vel,hi_kernel)
		hi_spec_smo = convolve(hi_spec_norm,hi_kernel)


		#get distance of this pixel from the GC
		_,dist_pc,_ = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)

		#plot the data
		ax = plt.subplot(gs[this_row,start_cols[j]:start_cols[j]+2],)
		c0=ax.fill_between(cii_vel,cii_spec,color=cii_color,alpha=0.5,step='pre')
		c1,=ax.step(cii_vel,cii_spec,color=cii_color,lw=1.5,label='[CII]')
		c2,=ax.step(co_vel_smo,co_spec_smo,color=co_color,lw=1.5,label='CO')#\\%.1e' %(1/co_sf))
		c3,=ax.step(hi_vel_smo,hi_spec_smo,color=hi_color,lw=1.5,label='HI')#\\%.1e' %(1/hi_sf))
		leg = plt.legend([(c0,c1),c2,c3],[c1.get_label(),c2.get_label(),c3.get_label()],
			fontsize=plt.rcParams['font.size']-2,loc='upper left')
		ax.text(0.5,0.95,pix_order_labels[j],ha='center',va='top',transform=ax.transAxes)
		ax.text(0.975,0.95,'%.2f kpc' %np.round(dist_pc.value/1E3,2),ha='right',va='top',transform=ax.transAxes)

		ax.set_xlim(-100,400)
		if i==0:
			ax.set_ylim(-0.05,0.25)
		else:
			ax.set_ylim(-0.25,0.5)
		ax.minorticks_on()

		if j !=5:
			ax.xaxis.set_ticklabels([])
			ax.yaxis.set_ticklabels([])
		else:
			ax.set_xlabel('Velocity (km s$^{-1}$)')
			ax.set_ylabel('T$_{\mathrm{mb}}$ (K)')
	plt.savefig('../Plots/Outflow_Spectra/M82_CII_CO_HI_Outflow'+str(i+1)+'.pdf',bbox_inches='tight',metadata={'Creator':this_script})
