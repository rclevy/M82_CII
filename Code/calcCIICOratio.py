this_script = __file__

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 24

#interpolate onto a common grid to get ratio
step = 0.01
x = np.arange(0,5+step,step)
co = pd.read_csv('../Data/Ancillary_Data/WCO_Av_B13.csv')
cii = pd.read_csv('../Data/Ancillary_Data/WCII_Av_B13.csv')
ico = interp1d(co['Av'].values,co['W'].values,fill_value='extrapolate')
icii = interp1d(cii['Av'].values,cii['W'].values,fill_value='extrapolate')
tab = {'Av': x, 'W_CO': ico(x), 'W_CII': icii(x)}
model = pd.DataFrame.from_dict(tab)
model.to_csv('../Data/Ancillary_Data/Av_WCO_WCII_B13.csv',index=False)
Av_model = model['Av'].values
Irat_model = model['W_CII'].values/model['W_CO'].values

#open the disk data
cii_disk = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits')[0].data
co_disk = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits')[0].data
Irat_disk = cii_disk/co_disk
ave_Irat_disk = np.nanmean(Irat_disk)
std_Irat_disk = np.nanstd(Irat_disk)


#open the outflow pointing data
cii_outflow = pd.read_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams.csv')
pix_num = cii_outflow['Pixel Number'].values
I_CII_out = cii_outflow['Mom0'].values
eI_CII_out = cii_outflow['eMom0'].values
I_CO_out = np.zeros_like(I_CII_out)
eI_CO_out = np.zeros_like(I_CII_out)
for i in range(7):
	co_outflow = pd.read_csv('../Data/Outflow_Pointings/CO_TP_Moments/outflow1_pix'+str(i)+'_momentparams.csv')
	I_CO_out[i] = co_outflow['Mom0'].values[0]
	eI_CO_out[i] = co_outflow['eMom0'].values[0]
Irat = I_CII_out/I_CO_out
eIrat = np.sqrt((eI_CII_out/I_CII_out)**2+(eI_CO_out/I_CO_out)**2)
#sort in order of increasing distance from midplane
idx = np.array([6,1,0,2,3])
cols = ['xkcd:melon','xkcd:golden rod','xkcd:boring green','xkcd:cerulean','xkcd:lavender']
hatch = [None,'||','//','\\\\',None]
ls = ['--',(0,(5,10)),(7.5,(5,10)),'-.',(0,(1,2))]

plt.figure(1,figsize=(14,8))
plt.clf()
lm, = plt.plot(Av_model,Irat_model,'k-',lw=3,label='Bolatto et al. 2013')
plt.yscale('log')
plt.xlim(0,5)
plt.ylim(1E-2)
plt.minorticks_on()
plt.xlabel('$A_V$')
plt.ylabel('I$_\mathrm{int,K~km~s^{-1}}^\mathrm{[CII]}$/I$_\mathrm{int,K~km~s^{-1}}^\mathrm{CO}$')
plt.grid(which='major',axis='both')


for i in range(len(np.ravel(Irat_disk))):
	plt.axhline(np.ravel(Irat_disk)[i],
	color='xkcd:purply pink',lw=0.75,alpha=0.1)
ld1 = plt.axhline(ave_Irat_disk,
	color='xkcd:purply pink',lw=2,label='Disk')

l0 = ['']*len(idx)
l1 = ['']*len(idx)
for i in range(len(idx)):
	l0[i] = plt.axhspan(Irat[idx[i]]-eIrat[idx[i]],Irat[idx[i]]+eIrat[idx[i]],
		color=cols[i],alpha=0.2,hatch=hatch[i],zorder=2)
	l1[i] = plt.axhline(Irat[idx[i]],
		color=cols[i],zorder=4,lw=2,linestyle=ls[i],label='Pixel '+str(idx[i]))
l = [(ll0,ll1) for ll0,ll1 in zip(l0,l1)]
l.insert(0,lm)
l.insert(1,ld1)
ll = [(ll1.get_label()) for ll1 in l1]
ll.insert(0,lm.get_label())
ll.insert(1,ld1.get_label())
plt.legend(l,ll,fontsize=plt.rcParams['font.size']-6,loc='upper right')
plt.arrow(0.875,0.85,0.,-0.25,transform=plt.gca().transAxes,
	color='k',zorder=10,linewidth=3,
	head_width=0.01)
plt.text(0.915,0.725,'Increasing distance\nfrom midplane',rotation=90,
	ha='center',va='center',transform=plt.gca().transAxes,fontsize=plt.rcParams['font.size']-10,zorder=10)
plt.savefig('../Plots/ICII_ICO_Av.pdf',bbox_inches='tight',metadata={'Creator':this_script})

