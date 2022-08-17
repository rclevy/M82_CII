#calculate some properties of the [CII] in the disk of M82

import numpy as np
from astropy.io import fits
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
pix_size_arcsec = I_CII[0].header['CDELT2']*3600
A_pix_pc2 = (pix_size_arcsec/206265*d)**2 #pc^2
A_pix_cm2 = A_pix_pc2*(3.086E18)**2 #cm^2
I_CII = I_CII[0].data
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