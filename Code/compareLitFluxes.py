#compare [CII] flux measurements from the Literature to what I measure here from PACS and upGREAT


import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import aperture_photometry,SkyRectangularAperture, SkyCircularAperture


#define some constants
c = 2.9979E5*u.km/u.s
nu = 1.900537*u.THz

#define some unit conversions
def Jypixkms_to_Wm2pix(I):
	a = 6.3E-20*u.W/u.m**2/u.pix/(u.Jy/u.pix*u.km/u.s)
	return a*I

def Wm2pix_to_Jypixkms(I):
	a = 6.3E-20*u.W/u.m**2/u.pix/(u.Jy/u.pix*u.km/u.s)
	return I/a

def ergscm2pix_to_Wm2pix(I):
	a = 1E-3*u.W/u.m**2/u.pix/(u.erg/u.s/u.cm**2/u.pix)
	return a*I

def Wm2pix_to_ergscm2pix_to_(I):
	a = 1E-3*u.W/u.m**2/u.pix/(u.erg/u.s/u.cm**2/u.pix)
	return I/a

def ergscm2sr_to_Kkms(I):
	a = 1.43E5*u.K*u.km/u.s/(u.erg/u.s/u.cm**2/u.sr)
	return a*I

def Kkms_to_ergscm2sr(I):
	a = 1.43E5*u.K*u.km/u.s/(u.erg/u.s/u.cm**2/u.sr)
	return I/a

#start with the PACS data published in Herrera-Camus+2018
I_PACS_HC18 = 1.43E-13*u.W/u.m**2/u.pixel
Apix_PACS_HC18 = (47*u.arcsec)**2/u.pixel

#KAO (Stacey+1991)
I_KAO_S91 = 69.E-5*u.erg/u.s/u.cm**2/u.sr
Apix_KAO_S91 = np.pi*(120*u.arcsec/2)**2/u.pixel
I_KAO_S91 = (I_KAO_S91*Apix_KAO_S91).to(u.W/u.m**2/u.pixel)

#open the PACS mom0 (continuum subtracted)
I_PACS = fits.open('../Data/Ancillary_Data/Herschel_PACS_158um_intinten_contsub.fits')
hdr_PACS = I_PACS[0].header
I_PACS = I_PACS[0].data*u.Jy/u.pixel*u.km/u.s
I_PACS = Jypixkms_to_Wm2pix(I_PACS)

#open the PACS mom0 (continuum subtracted and convolved to the upGREAT beam and pixel scale)
I_PACS_14as = fits.open('../Data/Ancillary_Data/Herschel_PACS_158um_intinten_14asGaussPSF_contsub.fits')
hdr_PACS_14as = I_PACS_14as[0].header
I_PACS_14as = I_PACS_14as[0].data*u.Jy/u.pixel*u.km/u.s
I_PACS_14as = Jypixkms_to_Wm2pix(I_PACS_14as)

#open the upGREAT mom0 (already continuum subtracted)
I_upGREAT = fits.open('../Data/Disk_Map/Moments/M82_CII_map_mom0_maskSNR2.fits')
hdr_upGREAT = I_upGREAT[0].header
I_upGREAT = I_upGREAT[0].data*u.K*u.km/u.s
Apix_upGREAT = (hdr_upGREAT['CDELT2']*3600*u.arcsec)**2/u.pix
I_upGREAT = Kkms_to_ergscm2sr(I_upGREAT)
I_upGREAT = I_upGREAT*(Apix_upGREAT.to(u.sr/u.pix))
I_upGREAT = ergscm2pix_to_Wm2pix(I_upGREAT)

# print('Though not over the same area, S91 KAO is %.1fx higher than HC18 PACS' 
# 	%((I_KAO_S91/I_PACS_HC18).value))
# print('\t(the area of S91 is %.1fx larger than HC18)' 
# 	%((Apix_KAO_S91/Apix_PACS_HC18).value))


#first compare to the PACS map from HC18, extract over the same area
center = SkyCoord('9h55m52.22s','+69d40m46.9s')
aper_HC18 = SkyRectangularAperture(center,47*u.arcsec,47*u.arcsec,theta=96.25238954968728*u.deg)
I_PACS_HC18aper = aper_HC18.to_pixel(WCS(hdr_PACS)).to_mask(method='center').multiply(I_PACS.value)*I_PACS.unit
I_PACS_HC18aper_sum = np.nansum(I_PACS_HC18aper)
I_PACS_14as_HC18aper = aper_HC18.to_pixel(WCS(hdr_PACS_14as)).to_mask(method='center').multiply(I_PACS_14as.value)*I_PACS_14as.unit
I_PACS_14as_HC18aper_sum = np.nansum(I_PACS_14as_HC18aper)
I_upGREAT_HC18aper = aper_HC18.to_pixel(WCS(hdr_upGREAT)).to_mask(method='center').multiply(I_upGREAT.value)*I_upGREAT.unit
I_upGREAT_HC18aper_sum = np.nansum(I_upGREAT_HC18aper)

print('In a 47"x47" region (region taken by Herrera-Camus+2018):')
print('\tHC18 report %.2e W/m^2 from PACS' %I_PACS_HC18.value)
print('\tI find %.2e W/m^2 from native PACS' %I_PACS_HC18aper_sum.value)
print('\tI find %.2e W/m^2 from 14" PACS' %I_PACS_14as_HC18aper_sum.value)
print('\tI find %.2e W/m^2 from upGREAT' %I_upGREAT_HC18aper_sum.value)
print('\tSo upGREAT is %.1fx higher than 14" PACS and %.1fx higher than HC18'
	%((I_upGREAT_HC18aper_sum/I_PACS_14as_HC18aper_sum).value,(I_upGREAT_HC18aper_sum/I_PACS_HC18).value))
print('\tHC18 is %.1fx higher than native PACS' 
	%((I_PACS_HC18/I_PACS_HC18aper_sum).value))

#now extract in a 120" aperture from S91
center = SkyCoord('9h51m43.9s','+69d55m01s',equinox='J1950', frame='fk4').transform_to('icrs')
aper_S91 = SkyCircularAperture(center,60*u.arcsec)
I_PACS_S91aper = aper_S91.to_pixel(WCS(hdr_PACS)).to_mask(method='center').multiply(I_PACS.value)*I_PACS.unit
I_PACS_S91aper_sum = np.nansum(I_PACS_S91aper)
I_PACS_14as_S91aper = aper_S91.to_pixel(WCS(hdr_PACS_14as)).to_mask(method='center').multiply(I_PACS_14as.value)*I_PACS_14as.unit
I_PACS_14as_S91aper_sum = np.nansum(I_PACS_14as_S91aper)
I_upGREAT_S91aper = aper_S91.to_pixel(WCS(hdr_upGREAT)).to_mask(method='center').multiply(I_upGREAT.value)*I_upGREAT.unit
I_upGREAT_S91aper_sum = np.nansum(I_upGREAT_S91aper)

print('In a 120" circular region (region taken by Stacey+1991):')
print('\tS91 report %.2e W/m^2 from KAO' %I_KAO_S91.value)
print('\tI find %.2e W/m^2 from native PACS' %I_PACS_S91aper_sum.value)
print('\tI find %.2e W/m^2 from 14" PACS' %I_PACS_14as_S91aper_sum.value)
print('\tI find %.2e W/m^2 from upGREAT' %I_upGREAT_S91aper_sum.value)
print('\tSo upGREAT is %.1fx higher than 14" PACS and %.1fx higher than KAO'
	%((I_upGREAT_S91aper_sum/I_PACS_14as_S91aper_sum).value,(I_upGREAT_S91aper_sum/I_KAO_S91).value))
print('\tKAO is %.1fx higher than PACS' 
	%((I_KAO_S91/I_PACS_S91aper_sum).value))







