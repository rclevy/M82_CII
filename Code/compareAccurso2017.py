this_script = __file__

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.io import fits
import seaborn as sns
import matplotlib.patches as mpatches


def z_from_dist(dist):
	#this only works for nearby sources!
	H0 = 70. #km/s/Mpc
	v = H0*dist #km/s
	c = 2.9979E5 #km/s
	z = v/c
	return z

def z_from_dist_vsys(dist,vsys):
	#this only works for nearby sources!
	c = 2.9979E5 #km/s
	z = vsys/c
	return z

def DL_from_z(z):
	cosmo = FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)
	return cosmo.luminosity_distance(z).value


#set up some constants
nu_CO_GHz = 115.
nu_CII_GHz = 1.9005E3
a17_beam_maj = 21.4 #arcsec, CO and [CII] are matched resolution
m82_CII_beam_maj = 14.1 #arcsec
m82_CII_pixel_size = 7. #arcsec
m82_CO_beam_maj = 22. #arcsec
m82_CO_pixel_size = 7. #arcsec
m82_dist = 3.63 #Mpc
m82_vsys = 210. #km/s
m82_z = z_from_dist_vsys(m82_dist,m82_vsys)
nbeam_out = 1 #for the upGREAT CII measurements
nbeam_disk = m82_CII_pixel_size**2/(np.pi*(m82_CII_beam_maj/2)**2)
nbeam_a17 = 1
nbeam_pacs = m82_CO_pixel_size**2/(np.pi*(m82_CO_beam_maj/2)**2)

def K_to_Jybeam(T,nu,bmaj,bmin):
	#T in K
	#nu on GHz
	#bmaj,bmin in arcsec
	return T*nu**2*bmaj*bmin/(1.222E6) #Jy/beam


def Lsun_from_Iint(Iint,nu,z,DL):
	#Iint in Jy km/s
	#nu in GHz
	#DL in Mpc
	#Solomon+1997 Eq 1 (this is general, not just for CO)
	return 1.04E-3*Iint*nu*(1+z)**-1*DL**2 #Lsun

def Iint_from_Lsun(Lsun,nu,z,DL):
	return Lsun*nu**-1*(1+z)*DL**-2/(1.04E-3)

def Jybeam_to_K(Iint,nu,bmaj,bmin):
	return 1.222E6*Iint/(nu**2*bmaj*bmin)

def Lprime_to_L(Lprime,nu):
	#combining equations 1 and 3 of Solomon+1997
	#nu_rest in GHz
	return 3.17E-12*nu**3*Lprime


#open the Accurso+2017 data
a17 = pd.read_csv('../Data/Ancillary_Data/Accurso2017_Tables1-2.csv')
a17_L_CO_Lsun = 10**a17['log_LCO1-0_Lsun'].values
a17_L_CII_Lsun = 10**a17['log_LCII_Lsun'].values
a17_z = a17['z_SDSS'].values
a17_dist = DL_from_z(a17_z)
a17_Iint_CO_Kkms = Jybeam_to_K(Iint_from_Lsun(a17_L_CO_Lsun,nu_CO_GHz,a17_z,a17_dist)/nbeam_a17,nu_CO_GHz,a17_beam_maj,a17_beam_maj)
a17_Iint_CII_Kkms = Jybeam_to_K(Iint_from_Lsun(a17_L_CII_Lsun,nu_CII_GHz,a17_z,a17_dist)/nbeam_a17,nu_CII_GHz,a17_beam_maj,a17_beam_maj)


dgs = np.where(a17['DGS'].values==1)[0]
ndgs = np.where(a17['DGS'].values==0)[0]


#open the M82 data
#outflow data
m82_out = pd.read_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams.csv')
m82_out_Iint_CO_Kkms = m82_out['Mom0_CO'].values
m82_out_eIint_CO_Kkms = m82_out['eMom0_CO'].values
m82_out_Iint_CII_Kkms = m82_out['Mom0'].values
m82_out_eIint_CII_Kkms = m82_out['eMom0'].values
out_pix_text = m82_out['Pixel Number'].values

# #disk data
m82_disk_Iint_CII_Kkms = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_matchedCIICO.fits')[0].data.ravel()
m82_disk_Iint_CO_Kkms = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedCIICO.fits')[0].data.ravel()
m82_pacs_Iint_CO_Kkms = fits.open('../Data/Ancillary_Data/M82.CO.30m.IRAM_reprocessed_mom0_matchedPACSCIICO.fits')[0].data.ravel()

#convert from K km/s to Jy/beam km/s
m82_out_Idv_CO_Jybeamkms = K_to_Jybeam(m82_out_Iint_CO_Kkms,nu_CO_GHz,m82_CO_beam_maj,m82_CO_beam_maj)#*nbeam_out
m82_out_eIdv_CO_Jybeamkms = K_to_Jybeam(m82_out_eIint_CO_Kkms,nu_CO_GHz,m82_CO_beam_maj,m82_CO_beam_maj)#*nbeam_out
m82_out_Idv_CII_Jybeamkms = K_to_Jybeam(m82_out_Iint_CII_Kkms,nu_CII_GHz,m82_CII_beam_maj,m82_CII_beam_maj)#*nbeam_out
m82_out_eIdv_CII_Jybeamkms = K_to_Jybeam(m82_out_eIint_CII_Kkms,nu_CII_GHz,m82_CII_beam_maj,m82_CII_beam_maj)#*nbeam_out
m82_disk_Idv_CO_Jybeamkms = K_to_Jybeam(m82_disk_Iint_CO_Kkms,nu_CO_GHz,m82_CO_beam_maj,m82_CO_beam_maj)#*nbeam_disk
m82_disk_Idv_CII_Jybeamkms = K_to_Jybeam(m82_disk_Iint_CII_Kkms,nu_CII_GHz,m82_CII_beam_maj,m82_CII_beam_maj)#*nbeam_disk
m82_pacs_Idv_CO_Jybeamkms = K_to_Jybeam(m82_pacs_Iint_CO_Kkms,nu_CO_GHz,m82_CO_beam_maj,m82_CO_beam_maj)#*nbeam_disk
m82_pacs_Idv_CII_Jybeamkms = fits.open('../Data/Ancillary_Data/Herschel_PACS_158um_mom0_22asGaussPSF_matchedPACSCIICO.fits')[0].data.ravel()

#convert from Jy/beam to Jy/pixel
m82_out_Idv_CO_Jykms = m82_out_Idv_CO_Jybeamkms/nbeam_out
m82_out_eIdv_CO_Jykms = m82_out_eIdv_CO_Jybeamkms/nbeam_out
m82_out_Idv_CII_Jykms = m82_out_Idv_CII_Jybeamkms/nbeam_out
m82_out_eIdv_CII_Jykms = m82_out_eIdv_CII_Jybeamkms/nbeam_out
m82_disk_Idv_CO_Jykms = m82_disk_Idv_CO_Jybeamkms/nbeam_disk
m82_disk_Idv_CII_Jykms = m82_disk_Idv_CII_Jybeamkms/nbeam_disk
m82_pacs_Idv_CO_Jykms = m82_pacs_Idv_CO_Jybeamkms/nbeam_disk
m82_pacs_Idv_CII_Jykms = m82_pacs_Idv_CII_Jybeamkms/nbeam_pacs*2


#convert to luminosity in Lsun
m82o_L_CO_Lsun = Lsun_from_Iint(m82_out_Idv_CO_Jykms,nu_CO_GHz,m82_z,m82_dist)
m82o_eL_CO_Lsun = Lsun_from_Iint(m82_out_eIdv_CO_Jykms,nu_CO_GHz,m82_z,m82_dist)
m82o_L_CII_Lsun = Lsun_from_Iint(m82_out_Idv_CII_Jykms,nu_CII_GHz,m82_z,m82_dist)
m82o_eL_CII_Lsun = Lsun_from_Iint(m82_out_eIdv_CII_Jykms,nu_CII_GHz,m82_z,m82_dist)
m82d_L_CO_Lsun = Lsun_from_Iint(m82_disk_Idv_CO_Jykms,nu_CO_GHz,m82_z,m82_dist)
m82d_L_CII_Lsun = Lsun_from_Iint(m82_disk_Idv_CII_Jykms,nu_CII_GHz,m82_z,m82_dist)
m82p_L_CO_Lsun = Lsun_from_Iint(m82_pacs_Idv_CO_Jykms,nu_CO_GHz,m82_z,m82_dist)
m82p_L_CII_Lsun = Lsun_from_Iint(m82_pacs_Idv_CII_Jykms,nu_CII_GHz,m82_z,m82_dist)

#print the outflow ratios to a file
m82o_LCII_LCO = m82o_L_CII_Lsun/m82o_L_CO_Lsun
m82o_e_LCII_LCO = m82o_LCII_LCO*np.sqrt((m82o_eL_CII_Lsun/m82o_L_CII_Lsun)**2+(m82o_eL_CO_Lsun/m82o_L_CO_Lsun)**2)
tab = {'LCII_LCO':m82o_LCII_LCO,
	   'e_LCII_LCO':m82o_e_LCII_LCO}
df = pd.DataFrame.from_dict(tab)
df.insert(0,'Pixel Number',range(7),True)
df.set_index('Pixel Number',drop=False,inplace=True)
df.to_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams_luminosity_ratios.csv',index=False)


# report ratios
a17_LCII_LCO_sf_med = np.nanmedian(a17_L_CII_Lsun[ndgs]/a17_L_CO_Lsun[ndgs])
a17_LCII_LCO_dg_med = np.nanmedian(a17_L_CII_Lsun[dgs]/a17_L_CO_Lsun[dgs])
m82_LCII_LCO_out_med = np.nanmedian(m82o_L_CII_Lsun/m82o_L_CO_Lsun)
m82_LCII_LCO_disk_med = np.nanmedian(m82d_L_CII_Lsun/m82d_L_CO_Lsun)
m82_LCII_LCO_pacs_med = np.nanmedian(m82p_L_CII_Lsun/m82p_L_CO_Lsun)

print('Median LCII/LCO ratios (in units of L_sun)')
print('Accurso+2017 (star-forming)\t%.1f' %a17_LCII_LCO_sf_med)
#print('Accurso+2017 (dwarfs)\t\t%.1f' %a17_LCII_LCO_dg_med)
print('M82 (disk)\t\t\t%.1f' %m82_LCII_LCO_disk_med)
print('M82 (outflow)\t\t\t%.1f' %m82_LCII_LCO_out_med)


##also comapre to SMG from Ivison+2021
I10_L_CO_Lsun = 8.0E5
I10_eL_CO_Lsun = 0.4E5
I10_L_CII_Lsun = 5.5E9
I10_eL_CII_Lsun = 1.3E9

#compare to ALESS 73.1
DB14_L_CO_Lsun = Lprime_to_L(2.0E10,nu_CO_GHz) #Coppin+2010
DB14_L_CII_Lsun = 5.15E9

cii_color = sns.color_palette('crest',as_cmap=True)(150)
gray_cmap = sns.cubehelix_palette(n_colors=256,light=0.9,dark=0.0,hue=0.01,reverse=True,as_cmap=True)

plt.figure(1)
plt.clf()
sns.kdeplot(x=m82d_L_CO_Lsun,y=m82d_L_CII_Lsun,log_scale=True,
	cmap='crest',fill=True,levels=10,alpha=1.0)
l1=plt.errorbar(m82o_L_CO_Lsun,m82o_L_CII_Lsun,
	xerr=m82o_eL_CO_Lsun,yerr=m82o_eL_CII_Lsun,
	fmt='o',color=cii_color,capsize=3,label='M82 outflow (upGREAT; this work)')
for i in range(len(out_pix_text)):
	plt.text(m82o_L_CO_Lsun[i]*1.05,m82o_L_CII_Lsun[i]*0.95,out_pix_text[i],
		color=cii_color,va='top',ha='left',fontsize=plt.rcParams['font.size']-2)
# l3,=plt.plot(m82p_L_CO_Lsun,m82p_L_CII_Lsun,'rd',alpha=0.2,label='M82 (PACS; Contursi et al. 2013)')
l2,=plt.plot(a17_L_CO_Lsun[ndgs],a17_L_CII_Lsun[ndgs],'*',color='tomato',alpha=0.5,zorder=5,label='Star-forming galaxies (Accurso et al. 2017)')# (Accurso et al. 2017b)')
l3=sns.kdeplot(x=m82p_L_CO_Lsun,y=m82p_L_CII_Lsun,log_scale=True,
	cmap=gray_cmap,fill=False,levels=10,alpha=0.85,linewidths=1.25,label='M82 (PACS; Contursi et al. 2013)')

plt.grid(which='major',axis='both',alpha=0.5)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('L$_{\mathrm{CO}}$ (L$_\odot$)')
plt.ylabel('L$_{\mathrm{[CII]}}$ (L$_\odot$)')
plt.xlim(left=1E2)
plt.ylim(bottom=5E4)
# plt.legend([l0,l1,l2],
# 	[l0.get_label(),l1.get_label(),l2.get_label()],
# 	loc='lower right',fontsize=plt.rcParams['font.size']-5)
#plot some lines of constant LCII/LCO
# xlim = plt.xlim()
# xl = np.array(xlim)
# ylim = plt.ylim()
# yl = np.array(ylim)
# Lrat = np.array([100.,1000.,10000.])
# # for lr in Lrat:
# # 	plt.plot(xl,lr*xl,'k--',alpha=0.5,lw=0.75,zorder=2)
# # plt.plot(xl,a17_LCII_LCO_sf_med*xl,'b-')
# # # plt.plot(xl,m82_LCII_LCO_pacs_med*xl,'r-')
# # plt.plot(xl,m82_LCII_LCO_disk_med*xl,'-',color='gray')
# # plt.plot(xl,m82_LCII_LCO_out_med*xl,'k-')
# # plt.text(xl[1],m82_LCII_LCO_out_med*xl[1]*0.78,'$\mathrm{log_{10}\\left[med\\left(\\frac{L_{[CII]}}{L_{CO}}\\right)\\right]}$=%.1f' %np.log10(m82_LCII_LCO_out_med),color='k',rotation=26,va='top',ha='right')
# # plt.text(xl[1],m82_LCII_LCO_disk_med*xl[1]*0.8,'%.1f' %np.log10(m82_LCII_LCO_disk_med),color='gray',rotation=26,va='top',ha='right')
# # # plt.text(xl[1],m82_LCII_LCO_pacs_med*xl[1]*0.8,'%.1f' %np.log10(m82_LCII_LCO_pacs_med),color='r',rotation=26,va='top',ha='right')
# # plt.text(xl[1],a17_LCII_LCO_sf_med*xl[1]*0.8,'%.1f' %np.log10(a17_LCII_LCO_sf_med),color='b',rotation=26,va='top',ha='right')
# # #plt.text(xl[1],a17_LCII_LCO_dg_med*xl[1]*0.8,'%.1f' %np.log10(a17_LCII_LCO_dg_med),color='r',rotation=22,va='top',ha='right')
# plt.xlim(xlim)
# # plt.ylim(ylim)
plt.savefig('../Plots/Composite_Maps/M82_CII_CO_luminosities_Accurso2017.pdf',bbox_inches='tight',metadata={'Creator':this_script})

# plt.figure(2)
# plt.clf()
# l0,=plt.plot(m82_disk_Iint_CO_Kkms,m82_disk_Iint_CII_Kkms,'ks',alpha=0.3,label='M82 disk')
# l1=plt.errorbar(m82_out_Iint_CO_Kkms,m82_out_Iint_CII_Kkms,
# 	xerr=m82_out_eIint_CO_Kkms,yerr=m82_out_eIint_CII_Kkms,
# 	fmt='ko',color='k',capsize=3,label='M82 outflow')
# l2,=plt.plot(a17_Iint_CO_Kkms[ndgs],a17_Iint_CII_Kkms[ndgs],'D',color='b',alpha=0.3,label='Star-forming galaxies (Accurso+2017)')
# l3,=plt.plot(a17_Iint_CO_Kkms[dgs],a17_Iint_CII_Kkms[dgs],'d',color='r',alpha=0.3,label='Dwarf galaxies (Accurso+2017)')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('I$_{\mathrm{int,CO}}$ (K km s$^{-1}$)')
# plt.ylabel('I$_{\mathrm{int,[CII]}}$ (K km s$^{-1}$)')
# plt.legend([l0,l1,l2,l3],[l0.get_label(),l1.get_label(),l2.get_label(),l3.get_label()],
# 	loc='lower right',fontsize=plt.rcParams['font.size']-2)
# plt.savefig('../Plots/Composite_Maps/M82_CII_CO_intinten_Accurso2017.pdf',bbox_inches='tight',metadata={'Creator':this_script})



#now open the radiation field strength info
m82d_LCII_LCO = m82d_L_CII_Lsun/m82d_L_CO_Lsun
G_disk = 10**fits.open('../Data/Disk_Map/M82_RadnField_map_matchedCIICO.fits')[0].data.ravel()
df_G_out = pd.read_csv('../Data/Outflow_Pointings/outflow1_RadnField.csv')
G_out = 10**df_G_out['log10G_G0'].values
el_G_out = df_G_out['el_log10G_G0'].values*G_out*np.log(10)
eu_G_out = df_G_out['eu_log10G_G0'].values*G_out*np.log(10)

plt.figure(2)
plt.clf()
l0,=plt.plot(G_disk,m82d_LCII_LCO,'ks',alpha=0.25,label='M82 disk')
l1=plt.errorbar(G_out,m82o_LCII_LCO,
	xerr=(eu_G_out,el_G_out),yerr=m82o_e_LCII_LCO,
	fmt='o',color='k',capsize=3,label='M82 outflow')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$G~(G_0)$')
plt.ylabel('$L_{\mathrm{[CII]}}/L_{\mathrm{CO}}$')
plt.xlim(1E0,1E5)
plt.ylim(1E2,5E3)
plt.grid(which='major',axis='both',alpha=0.5,zorder=1)
plt.grid(which='minor',axis='y',alpha=0.25,zorder=1)
plt.legend([l0,l1],[l0.get_label(),l1.get_label()],
	loc='upper left',fontsize=plt.rcParams['font.size']-4)
plt.savefig('../Plots/Composite_Maps/M82_CII_CO_luminosities_RadnField.pdf',bbox_inches='tight',metadata={'Creator':this_script})



#plt.close('all')


