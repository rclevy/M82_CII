this_script = __file__

#plot CII, CO, and HI spectra for each of the single pointings
#based on pltCIIspectra.py

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Box1DKernel,convolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import upGREATUtils
import seaborn as sns


plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14

#load some properties of M82
gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
distance = 3.63*u.Mpc

n_pointings = 2
n_pixels = 7 #pixels per pointing
n_rows = int(np.ceil(np.sqrt(n_pixels)))
n_cols = int(np.floor(n_pixels/n_rows)*n_rows)

#set up plotting colors, same as in pltCompositeImage.py
cii_color = sns.color_palette('crest',as_cmap=True)(150)
co_color = '0.33'
hi_color = sns.color_palette('flare',as_cmap=True)(150)

co_norms = np.zeros((n_pixels,))
hi_norms = np.zeros((n_pixels,))

#get CO and HI velocity resolutions for smoothing
# co_dv = fits.open('../Data/Ancillary_Data/M82-NOEMA-30m-mask-merged.clean.fits')[0].header['CDELT3']/1E3 #km/s
# hi_dv = -1*fits.open('../Data/Ancillary_Data/m82_hi_24as_feathered_nhi.fits')[0].header['CDELT3']/1E3 #km/s

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

	if i==0:
		cii_sf_N = pd.read_csv('../Data/Outflow_Pointings/CII/outflow1_Mass_ColumnDensities.csv')['peak_NHI_NHICII'].values
	else:
		cii_sf_N = np.ones((n_pixels,))

	for j in range(n_pixels):
		this_row = int(np.floor((j+1)/n_rows))

		#load the CII data from the fits file
		cii_spec,cii_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'.fits')
		Vclip_red,Vclip_blue = upGREATUtils.get_bad_edge_chans(i,pix_order_labels[j])
		cii_vel,cii_spec = upGREATUtils.rm_bad_edge_chans(cii_vel,cii_spec,which_end='both',Vred=Vclip_red,Vblue=Vclip_blue)
		

		this_sf_N = cii_sf_N[j] #additional scaling based on the column density ratio

		cii_vel = np.flip(cii_vel)
		cii_spec = np.flip(cii_spec)
		#cii_spec_norm = cii_spec/np.nanmax(cii_spec)
		cii_dv = header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s
		cii_vix = np.where((cii_vel>=-100) & (cii_vel <= 400))
		cii_max = np.nanmax(cii_spec[cii_vix])

		n_edge = 6
		cii_vel = cii_vel[0:-n_edge]
		cii_spec = cii_spec[0:-n_edge]
		cii_vmax = np.nanmax(cii_vel)
		cii_vmin = np.nanmin(cii_vel)

		#load the HI data
		hi_dat = pd.read_csv('../Data/Outflow_Pointings/HI/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'_smo.csv')
		hi_vel = np.flip(hi_dat['Velocity_kms'].values)
		hi_spec = np.flip(hi_dat['Intensity_K'].values)
		#hi_spec = hi_spec_cm2/(1.823E18) #Martini+2018, K
		idx = np.where((hi_vel <= cii_vmax) & (hi_vel >= cii_vmin))
		hi_vel = hi_vel[idx]
		hi_spec = hi_spec[idx]
		hi_vix = np.where((hi_vel>-100) & (hi_vel < 400))
		hi_sf = cii_max/np.nanmax(hi_spec[hi_vix])*this_sf_N
		hi_norms[j] = 1/hi_sf
		hi_spec_norm = hi_spec*hi_sf

		#load the CO data
		co_dat = pd.read_csv('../Data/Outflow_Pointings/CO_TP/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'_smo.csv')
		co_vel = co_dat['Velocity_kms'].values
		co_spec = co_dat['Intensity_K'].values
		idx = np.where((co_vel <= cii_vmax) & (co_vel >= cii_vmin))
		co_vel = co_vel[idx]
		co_spec = co_spec[idx]
		co_vix = np.where((co_vel>-100) & (co_vel < 400))
		co_sf = cii_max/np.nanmax(co_spec[co_vix])*this_sf_N
		co_norms[j] = 1/co_sf
		co_spec_norm = co_spec*co_sf


		# #smooth the CO and HI data to the CII vel res
		# co_kernel = Box1DKernel(cii_dv/co_dv)
		# hi_kernel = Box1DKernel(cii_dv/hi_dv)
		# co_vel_smo = convolve(co_vel,co_kernel)
		# co_spec_smo = convolve(co_spec_norm,co_kernel)
		# hi_vel_smo = convolve(hi_vel,hi_kernel)
		# hi_spec_smo = convolve(hi_spec_norm,hi_kernel)

		# #linearly interpolate the CO and HI spectra onto the same velocity grid
		# cii_int = interp1d(cii_vel,cii_spec)
		# cii_spec_smo = cii_int(cii_vel_smo)
		# co_int = interp1d(co_vel,co_spec_norm)
		# co_spec_smo = co_int(cii_vel_smo)
		# hi_int = interp1d(hi_vel,hi_spec_norm)
		# hi_spec_smo = hi_int(cii_vel_smo)


		#get distance of this pixel from the GC
		_,dist_pc,_ = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)

		#plot the data
		ax = plt.subplot(gs[this_row,start_cols[j]:start_cols[j]+2],)
		c0=ax.fill_between(cii_vel[cii_vix],cii_spec[cii_vix],color=cii_color,alpha=0.5,step='pre')
		c1,=ax.step(cii_vel,cii_spec,color=cii_color,lw=1.5,label='[CII]')
		c2,=ax.step(co_vel,co_spec_norm,color=co_color,lw=1.75,label='CO')#\\%.1e' %(1/co_sf))
		c3,=ax.step(hi_vel,hi_spec_norm,color=hi_color,lw=1.75,label='HI')#\\%.1e' %(1/hi_sf))
		ax.text(0.05,0.95,pix_order_labels[j],ha='center',va='top',transform=ax.transAxes)
		ax.text(0.975,0.95,'%.2f kpc' %np.round(dist_pc.value/1E3,2),ha='right',va='top',transform=ax.transAxes)

		ax.text(0.025,0.8,'CO/%.1f' %(1/co_sf),color=co_color,fontsize=plt.rcParams['font.size']-3,ha='left',va='top',transform=ax.transAxes)
		ax.text(0.025,0.7,'HI/%.1f' %(1/hi_sf),color=hi_color,fontsize=plt.rcParams['font.size']-3,ha='left',va='top',transform=ax.transAxes)

		# if j == n_pixels-1:
		if j == 1:
			leg = plt.legend([(c0,c1),c2,c3],[c1.get_label(),c2.get_label(),c3.get_label()],
				handlelength=1.25,fontsize=plt.rcParams['font.size']-2,loc='center left',bbox_to_anchor=(1.025,0.65,0.2,0.2))


		ax.set_xlim(-150,450)
		if i==0:
			ax.set_ylim(-0.05,0.3)
		else:
			ax.set_ylim(-0.25,0.5)
		ax.minorticks_on()

		if j==5:
			ax.set_xlabel('V$_{\mathrm{LSRK}}$ (km s$^{-1}$)')
		elif j==0:
			ax.set_ylabel('T$_{\mathrm{mb}}$ (K)')
		elif (j==1) | (j==3) | (j==4) | (j==6):
			ax.yaxis.set_ticklabels([])
			
			
	plt.savefig('../Plots/Outflow_Spectra/M82_CII_CO_HI_Outflow'+str(i+1)+'.pdf',bbox_inches='tight',metadata={'Creator':this_script})

	#save CO and HI scale factors
	tab = {'Pixel_Number': pix_order_labels, 'ICO_ICII': co_norms, 'IHI_ICII': hi_norms}
	df = pd.DataFrame(tab)
	df.to_csv('../Data/Outflow_Pointings/CO_HI_normalizations_Outflow'+str(i+1)+'.csv',index=False)
