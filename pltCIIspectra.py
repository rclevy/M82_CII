this_script = __file__

#load and plot the outflow spectra in the 7-point footprint config

import argparse

parser=argparse.ArgumentParser(
    description='''Plot [CII] spectra in outflow from SOFIA in multipanels arranged by the detector and sky orientations. Can specify a velocity resolution.''',
    epilog='''Author: R. C. Levy - Last updated: 2022-04-04''')
parser.add_argument('--vel_res', type=float, help='Desired velocity resolution in km/s')
parser.add_argument('--plot_fit',type=str, help='Overplot Gaussian fit (True) or not (False; default)')

args=parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import upGREATUtils
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd

plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14


def gauss(x,a,b,c):
	return a*np.exp((-(x-b)**2/(2*c**2)))

def gauss2(x,a1,b1,c1,a2,b2,c2):
	return a1*np.exp((-(x-b1)**2/(2*c1**2)))+a2*np.exp((-(x-b2)**2/(2*c2**2)))

if args.vel_res:
	new_dv = -args.vel_res
else:
	new_dv = np.nan

if args.plot_fit:
	if args.plot_fit=='True':
		plot_fit = True
		fitsuff = '_gaussianfits'
	else:
		plot_fit = False
		fitsuff = ''
else:
	plot_fit = False
	fitsuff = ''

#load some properties of M82
gal_center = SkyCoord('09h55m52.72s 69d40m45.7s')
distance = 3.63*u.Mpc

n_pointings = 2
n_pixels = 7 #pixels per pointing
n_rows = int(np.ceil(np.sqrt(n_pixels)))
n_cols = int(np.floor(n_pixels/n_rows)*n_rows)

cii_color = sns.color_palette('crest',as_cmap=True)(150)
fit_color = sns.color_palette('crest',as_cmap=True)(220)

for i in range(n_pointings):

	start_cols = [1,3,0,2,4,1,3]

	for orient in ['detector','sky']:


		#set up the subplots for this pointing
		fig = plt.figure(1,figsize=(8,8))
		plt.clf()
		gs = gridspec.GridSpec(n_rows,n_cols)

		if orient == 'sky':
			#as they are on the sky
			if i==0:

				pix_order_labels =  np.array([1,6,2,0,5,3,4])
			else:
				pix_order_labels =  np.array([1,2,6,0,3,5,4])
		else:
			#as they are on the detector
			pix_order_labels = np.array([4,3,5,0,2,6,1])


		#get the correct pixel order for plotting
		pix_order_detector,pix_order_sky = upGREATUtils.pixel_order_from_class(pix_order_labels)


		if orient == 'sky':
			#as they are on the sky
			pix_order = pix_order_sky
		else:
			#as they are on the detector
			pix_order = pix_order_detector

		for j in range(n_pixels):
			this_pix = pix_order[j]
			this_pix_label = str(pix_order_labels[j])
			this_row = int(np.floor((j+1)/n_rows))
			ax = plt.subplot(gs[this_row,start_cols[j]:start_cols[j]+2],)

			#load the data from the fits file
			this_spec,this_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow'+str(i+1)+'_'+str(this_pix)+'.fits')
			dv = -header['CDELT1']/header['RESTFREQ']*2.9979E5 #km/s

			#resample the velocity axis
			if np.isnan(new_dv) == False:
				if np.abs(new_dv) < np.abs(dv):
					print('Error: New velocity resolution must be coarser than native (%i km/s). Proceeding with native velocity resolution.' %np.abs(dv))
					new_spec = this_spec.copy()
					new_vel = this_vel.copy()
					new_dv = dv
				else:
					n_chan_sum = int(np.round(new_dv/dv))
					n_chan_clip = np.mod(len(this_spec),n_chan_sum)
					new_spec = np.nansum(this_spec[n_chan_clip:].reshape(-1,n_chan_sum),axis=1)
					new_vel = np.arange(this_vel[n_chan_clip],this_vel[-1],new_dv)				
			else:
				new_spec = this_spec.copy()
				new_vel = this_vel.copy()
				new_dv = dv

			#get distance of this pixel from the GC
			_,dist_pc,_ = upGREATUtils.calc_pix_distance_from_gal_center(gal_center, header, distance)

			#plot the data
			ax.fill_between(new_vel,new_spec,color=cii_color,alpha=0.5,step='pre')
			ax.step(new_vel,new_spec,color=cii_color)
			ax.text(0.05,0.95,this_pix_label,ha='center',va='top',transform=ax.transAxes)
			ax.text(0.95,0.95,'%.2f kpc' %np.round(dist_pc.value/1E3,2),ha='right',va='top',transform=ax.transAxes)


			#overplot gaussian fit
			if plot_fit == True:
				fitinfo = pd.read_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow'+str(i+1)+'_pix'+str(pix_order_labels[j])+'_gaussianfitparams.csv')
				Ipk = fitinfo['Peak'].values[0]
				eIpk = fitinfo['ePeak'].values[0]
				Iint = fitinfo['IntInten'].values[0]
				eIint = fitinfo['eIntInten'].values[0]
				Vc = fitinfo['Vcen'].values[0]
				eVc = fitinfo['eVcen'].values[0]
				sig = fitinfo['FWHM'].values[0]/2.355
				esig = fitinfo['eFWHM'].values[0]/2.355
				chi2r = fitinfo['Chi2r'].values[0]
				gauss_fit = gauss(this_vel,Ipk,Vc,sig)
				ax.plot(this_vel,gauss_fit,color=fit_color,lw=2.)
				#fit_str = '$\mathrm{I}_{\mathrm{peak}} = %.2f\pm%.2f \mathrm{~K}$\n$\mathrm{V}_0 = %1.f\pm%1.f \mathrm{~km~s}^{-1}$\n$\sigma_{\mathrm{V}} = %1.f\pm%1.f \mathrm{~km~s}^{-1}$\n$\mathrm{I}_{\mathrm{int}} = %1.f\pm%1.f \mathrm{~K~km~s}^{-1}$' %(Ipk,eIpk,Vc,eVc,sig,esig,Iint,eIint)		
				#ax.text(0.025,0.85,fit_str,ha='left',va='top',transform=ax.transAxes,color=fit_color,fontsize=plt.rcParams['font.size']-2)



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
		plt.savefig('../Plots/Outflow_Spectra/M82_CII_Outflow'+str(i+1)+'_'+orient+'_'+str(int(np.abs(new_dv)))+'kms'+fitsuff+'.pdf',bbox_inches='tight',metadata={'Creator':this_script})

