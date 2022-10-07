this_script = __file__

#calculate the [CII] column density, channel-by-channel
import numpy as np
import pandas as pd
import upGREATUtils
from scipy.interpolate import interp1d
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14

#define some key functions

def B_T(T,SigRuln):
	#Crawford+1985, Goldsmith+2012, Tarantino+2021
	Aul = 2.3E-6 #s^-1
	return 2*np.exp(-91.2/T)/(1+2*np.exp(-91.2/T)+Aul/SigRuln)

def SigRuln(T,n_CNM):
	Rul_H0 = 4.0E-11*(16+0.35*T**0.5+48*T**-1) #Goldsmith+2012
	return 1.038*Rul_H0*n_CNM #Draine+2011

def NCp_from_TpCII(Tp_CII,dv,B):
	#Crawford+1985, Goldsmith+2012, Tarantino+2021
	return dv*Tp_CII/B*3.0E15

def NHI_CII(NCp_from_TpCII,f_CNM):
	C_H = 1.5E-4 #C/H, Gerin+2015
	#assumes all C singly-ionized and all H atomic
	return NCp_from_TpCII/C_H/f_CNM

def MCp_from_NCp(N_Cp,A_pix_cm2):
	m_Cp = 1.99E-23 #g
	Msun = 1.99E33 #g
	return N_Cp*A_pix_cm2*m_Cp/Msun #Msun


#define some fixed parameters
D_pix = 14.1 #arcsec
d = 3.63E6 #pc
A_pix = np.pi*(D_pix/2/206265*d)**2 #pc^2
A_pix_cm2 = A_pix*(3.086E18)**2 #cm^2
T = 100. #K, T_CNM
n_CNM = 100. #cm^-3
dv = 10. #km/s
f_CNM = 0.5
n_pixels = 7
n_rows = int(np.ceil(np.sqrt(n_pixels)))
n_cols = int(np.floor(n_pixels/n_rows)*n_rows)
pix_order_labels =  np.array([1,6,2,0,5,3,4])

#set up plotting colors, same as in pltCompositeImage.py
cii_color = sns.color_palette('crest',as_cmap=True)(150)
cii_color_dark = sns.color_palette('crest',as_cmap=True)(225)
co_color = '0.33'
hi_color = sns.color_palette('flare',as_cmap=True)(150)

fig = plt.figure(1,figsize=(8,8))
plt.clf()
gs = gridspec.GridSpec(n_rows,n_cols)

start_cols = [1,3,0,2,4,1,3]

#EVENTUALLY DONT HARD CODE THIS
rms = np.array([8.3, 9.4, 9.6, 6.9, 6.8, 16.8, 11.5])/1E3 #K

Nn_cii = []
Mm_cii = []
eu_Mm_cii = []
el_Mm_cii = []
med_ratios = []
peak_ratios = []

for i in range(n_pixels):
	this_row = int(np.floor((i+1)/n_rows))


	#load the CII data from the fits file
	cii_spec,cii_vel,header,_ = upGREATUtils.load_upgreat_data('../Data/Outflow_Pointings/CII/outflow1_pix'+str(pix_order_labels[i])+'.fits')
	Vclip_red,Vclip_blue = upGREATUtils.get_bad_edge_chans(1,pix_order_labels[i])
	cii_vel,cii_spec = upGREATUtils.rm_bad_edge_chans(cii_vel,cii_spec,which_end='both',Vred=Vclip_red,Vblue=Vclip_blue)

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
	hi_dat = pd.read_csv('../Data/Outflow_Pointings/HI/outflow1_pix'+str(pix_order_labels[i])+'_smo_summed.csv')
	hi_vel = np.flip(hi_dat['Velocity_kms'].values)
	hi_spec = np.flip(hi_dat['Intensity_K'].values)
	idx = np.where((hi_vel <= cii_vmax) & (hi_vel >= cii_vmin))
	hi_vel = hi_vel[idx]
	hi_spec = hi_spec[idx]
	#interpolate hi onto cii velocity grid
	hi_int = interp1d(hi_vel,hi_spec,fill_value='extrapolate')
	hi_spec = hi_int(cii_vel)

	#now calculate the CII column density at each channel
	N_cii = NCp_from_TpCII(cii_spec,dv,B_T(T,SigRuln(T,n_CNM))) #cm^-2
	N_hi_cii = NHI_CII(N_cii,f_CNM)

	#get uncertinaties
	N_cii_max = NCp_from_TpCII(cii_spec+rms[pix_order_labels[i]],dv,B_T(T,SigRuln(T,n_CNM))) #cm^-2
	N_hi_cii_max = NHI_CII(N_cii_max,f_CNM)
	N_cii_min = NCp_from_TpCII(cii_spec-rms[pix_order_labels[i]],dv,B_T(T,SigRuln(T,n_CNM))) #cm^-2
	N_hi_cii_min = NHI_CII(N_cii_min,f_CNM)

	#sum the CII column
	N_cii_sum = np.nansum(N_cii)
	N_cii_max_sum = np.nansum(N_cii_max)
	N_cii_min_sum = np.nansum(N_cii_min)
	Nn_cii.append(N_cii_sum)
	this_M_cii = MCp_from_NCp(N_cii_sum,A_pix_cm2) #Msun
	this_M_cii_max = MCp_from_NCp(N_cii_max_sum,A_pix_cm2) #Msun
	this_M_cii_min = MCp_from_NCp(N_cii_min_sum,A_pix_cm2) #Msun
	Mm_cii.append(this_M_cii) #Msun
	eu_Mm_cii.append(this_M_cii_max-this_M_cii)
	el_Mm_cii.append(this_M_cii-this_M_cii_min)

	#convert the HI intensity back to a column density
	N_hi = hi_spec*dv*1.823E18 #cm^-2

	#take the ratio
	ratio_N_hi_cii = N_hi/N_hi_cii
	#only find the ratio where there's signal in the HI
	vix = np.where((cii_vel > 25.) & (cii_vel < 300.))
	med_ratio_N_hi_cii = np.nanmedian(ratio_N_hi_cii[vix])
	med_ratios.append(med_ratio_N_hi_cii)

	#find the ratio at the CII peak
	tt = N_hi_cii.copy()
	tt[cii_vel < 25] = np.nan
	tt[cii_vel > 300] = np.nan
	ixp = np.nanargmax(tt)
	peak_ratio = np.nanmax(N_hi)/N_hi_cii[ixp]
	peak_ratios.append(peak_ratio)


	#
	norm = 1E20

	#also compare for different values of f_CNM
	N_hi_cii_70 = NHI_CII(N_cii,0.7)/dv
	N_hi_cii_30 = NHI_CII(N_cii,0.3)/dv
	d_fCNM = np.nanmedian(N_hi_cii_30[vix]/norm-N_hi_cii_70[vix]/norm)
	d_fCNM_mid = np.nanmedian(N_hi_cii_30[vix]/norm-N_hi_cii[vix]/dv/norm)


	#plot the data
	ax = plt.subplot(gs[this_row,start_cols[i]:start_cols[i]+2],)
	c2=ax.fill_between(cii_vel[vix],N_hi[vix]/dv/norm,color=hi_color,alpha=0.1,step='pre')
	c0=ax.fill_between(cii_vel[vix],N_hi_cii[vix]/dv/norm,color=cii_color,alpha=0.5,step='pre')
	c3,=ax.step(cii_vel,N_hi/dv/norm,color=hi_color,lw=1.75,label='$N_{\mathrm{HI}}$')#\\%.1e' %(1/hi_sf))
	c1,=ax.step(cii_vel,N_hi_cii/dv/norm,color=cii_color,lw=1.5,label='$N_{\mathrm{HI}}^{\mathrm{[CII]}}$')
	
	#add errorbars for different f_cnm values
	xx = -120; yy = 3.25;
	ax.plot([xx,xx],[yy,yy-d_fCNM],color=cii_color,lw=1.5)
	ax.plot([xx-10,xx+10],[yy-d_fCNM_mid,yy-d_fCNM_mid],color=cii_color,lw=1.5)
	ax.plot([xx-5,xx+5],[yy,yy],color=cii_color,lw=1.5)
	ax.plot([xx-5,xx+5],[yy-d_fCNM,yy-d_fCNM],color=cii_color,lw=1.5)
	if i==0:
		ax.text(xx+10,yy,' $f_\mathrm{CNM}$=0.3',color=cii_color,
			ha='left',va='center',fontsize=plt.rcParams['font.size']-8)
		ax.text(xx+10,yy-d_fCNM_mid,' $f_\mathrm{CNM}$=0.5',color=cii_color,
			ha='left',va='center',fontsize=plt.rcParams['font.size']-8)
		ax.text(xx+10,yy-d_fCNM,' $f_\mathrm{CNM}$=0.7',color=cii_color,
			ha='left',va='center',fontsize=plt.rcParams['font.size']-8)

	ax.text(0.05,0.95,pix_order_labels[i],ha='center',va='top',transform=ax.transAxes)
	ax.text(0.975,0.975,'$\mathrm{\\frac{max(N_{HI})}{max(N_{HI}^{[CII]})}=%.1f}$' %(peak_ratio),
		ha='right',va='top',fontsize=plt.rcParams['font.size']-4,transform=ax.transAxes)

	if i == 1:
		leg = plt.legend([(c0,c1),(c2,c3)],[c1.get_label(),c3.get_label()],
			handlelength=1.25,fontsize=plt.rcParams['font.size']-2,loc='center left',bbox_to_anchor=(1.025,0.65,0.2,0.2))


	ax.set_xlim(-150,450)
	ax.set_ylim(-0.5,4.)
	ax.minorticks_on()
	ax.set_yticks([0,1,2,3,4])
	ax.grid(which='major',axis='y',alpha=0.5)


	if i==5:
		ax.set_xlabel('V$_{\mathrm{LSRK}}$ (km s$^{-1}$)')
		#ax.set_ylabel('N$_{\mathrm{HI}}$ ($\\times10^{21}$ cm$^{-2}$)')
	elif (i==0):
		ax.set_ylabel('$N\\times10^{%i}$\n(cm$^{-2}$ (km s$^{-1}$)$^{-1}$)' %np.log10(norm))
	elif (i==1) | (i==3) | (i==4) | (i==6):
		#ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])
plt.savefig('../Plots/Outflow_Spectra/M82_ColumnDensity_CII_HI_Outflow1.pdf',bbox_inches='tight',metadata={'Creator':this_script})


Mmm_cii = Mm_cii.copy()
eu_Mmm_cii = eu_Mm_cii.copy()
el_Mmm_cii = el_Mm_cii.copy()
Mmm_cii[4]=0.
Mmm_cii[5]=0.
eu_Mmm_cii[4]=0.
el_Mmm_cii[5]=0.
eu_Mmm_cii[4]=0.
el_Mmm_cii[5]=0.
print('The total C+ mass in the outflow is %.2e Msun' %np.nansum(Mmm_cii))

tab = {'Pixel Number':pix_order_labels, 'ColumnDensity_Cp_cm2': Nn_cii,
	'Mass_Cp_Msun':Mm_cii,'e_Mass_Cp_Msun_upper':eu_Mm_cii,'e_Mass_Cp_Msun_lower':el_Mm_cii,
	'med_NHI_NHICII': med_ratios, 'peak_NHI_NHICII': peak_ratios}
df = pd.DataFrame.from_dict(tab)
df.to_csv('../Data/Outflow_Pointings/CII/outflow1_Mass_ColumnDensities.csv',index=False)


#now add the CO to the column density plot
del(leg)
for i in range(n_pixels):
	this_row = int(np.floor((i+1)/n_rows))

	#open the CO data
	co_dat = pd.read_csv('../Data/Outflow_Pointings/CO_TP/outflow1_pix'+str(pix_order_labels[i])+'_smo.csv')
	co_vel = co_dat['Velocity_kms'].values
	co_spec = co_dat['Intensity_K'].values
	idx = np.where((co_vel <= cii_vmax) & (co_vel >= cii_vmin))
	co_vel = co_vel[idx]
	co_spec = co_spec[idx]
	#interpolate hi onto cii velocity grid
	co_int = interp1d(co_vel,co_spec,fill_value='extrapolate')
	co_spec = co_int(cii_vel)

	#now convert the CO intensity to a H2 column density
	XCO = 0.5E20 #cm-2/(K km/s), starburst value from Bolatto+2013
	#XCO = 2E20 #cm-2/(K km/s), MW value from Bolatto+2013
	N_h2_co = XCO*co_spec*dv #cm^-2

	ax = plt.subplot(gs[this_row,start_cols[i]:start_cols[i]+2],)
	c4=ax.fill_between(cii_vel[vix],N_h2_co[vix]/dv/norm,color=co_color,alpha=0.1,step='pre')
	c5,=ax.step(cii_vel,N_h2_co/dv/norm,color=co_color,lw=1.75,label='$N_{\mathrm{H_{2}}}^{\mathrm{CO}}$')#\\%.1e' %(1/hi_sf))


	if i == 1:
		leg = plt.legend([(c0,c1),(c2,c3),(c4,c5)],[c1.get_label(),c3.get_label(),c5.get_label()],
			handlelength=1.25,fontsize=plt.rcParams['font.size']-2,loc='center left',bbox_to_anchor=(1.025,0.625,0.2,0.2))
plt.savefig('../Plots/Outflow_Spectra/M82_ColumnDensity_CII_HI_CO_Outflow1.pdf',bbox_inches='tight',metadata={'Creator':this_script})

