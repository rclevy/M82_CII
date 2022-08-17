import numpy as np

def load_upgreat_data(filepath):
	#from Elizabeth Tarantino
	#load upgreat data from fits file and fix the header so the wcs works
	import numpy as np
	from astropy.io import fits
	from astropy.wcs import WCS
	import astropy.units as u

	hdulist = fits.open(filepath)
	data = hdulist[0].data
	header = hdulist[0].header
	hdulist.close
	# grab header values
	n = header['NAXIS1']
	vel_ref = header['VELO-LSR']
	rest_freq = header['RESTFREQ'] 
	# useful values
	c = 3e8
	spec = data[0,0,0,:]
	index = np.arange(n)
	# remove the STOKES axis so WCS works
	w = WCS(header, naxis = 1)
	# create the frequency axis 
	freq = w.wcs_pix2world(index, 1)
	freq = np.array(freq).transpose()
	
	# transform the frequency axis to velocity
	freq_shift = rest_freq - ((vel_ref*rest_freq)/c)
	freq = freq + freq_shift
	freq_to_vel = u.doppler_radio(rest_freq * u.Hz)
	vel = (freq * u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)  
	
	return spec, vel[:,0].value, header, w

def get_bad_edge_chans(pointing_num,pix_num):
	#velocity clipping for bad edge channels
	#pixel order 0,1,2,3,4,5,6
	Vclip_red = np.array([[480.,440.,460.,np.inf,470.,np.inf,440.,],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]])
	Vclip_blue = np.array([[-110.,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,],[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]])
	return Vclip_red[pointing_num,pix_num], Vclip_blue[pointing_num,pix_num]

def rm_bad_edge_chans(vel,spec,which_end='both',Vred=np.inf,Vblue=-np.inf):
	#remove bad channels from edges the bandpass
	#spectrum is ordered so red is first
	if (which_end=='both') or (which_end=='red'):
		idx = np.where(vel >= Vred)
		spec[idx] = np.nan
	if (which_end=='both') or (which_end=='blue'):
		idx = np.where(vel <= Vblue)
		spec[idx] = np.nan
	return vel,spec



def calc_pix_distance_from_gal_center(gal_center, header, distance):
	#calculate the distance of the given pixel from a specifed RA/Dec
	#gal_center should be passed as a SkyCoord object
	#distance should have the unit attached
	import numpy as np
	from astropy.coordinates import SkyCoord
	import astropy.units as u

	ra_pix = header['CRVAL2']*u.deg #deg
	dec_pix = header['CRVAL3']*u.deg #deg
	ra_off = header['CDELT2']*u.deg #deg
	dec_off = header['CDELT3']*u.deg #deg
	pix_center = SkyCoord(ra_pix+ra_off,dec_pix+dec_off)

	sep_arcsec = gal_center.separation(pix_center).to(u.arcsec)
	sep_pc = (sep_arcsec.value/206265*distance).to(u.pc)
	return sep_arcsec,sep_pc,pix_center

def pixel_order_from_class(plot_order_sky):
	import numpy as np
	#class's output pixel order is
	output_spectrum_number = np.arange(0,7,1) #from get 1, get 2, ...
	# this corresponds to a upGREAT pixel order
	upGREAT_pixel_number = np.array([0,4,3,2,1,6,5])
	#for plotting in the upGREAT frame
	plot_order_upGREAT = np.array([4,3,5,0,2,6,1])
	#now sort to get the right order based on the upGREAT plotting order
	#https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
	sort_upn = np.argsort(upGREAT_pixel_number)
	find_pou = np.searchsorted(upGREAT_pixel_number[sort_upn],plot_order_upGREAT)
	idx_plot_order_upGREAT = sort_upn[find_pou]
	spectrum_number_plot_order_upGREAT = output_spectrum_number[idx_plot_order_upGREAT]
	#repeat to get the right order based on the upGREAT sky order
	find_pos = np.searchsorted(upGREAT_pixel_number[sort_upn],plot_order_sky)
	idx_plot_order_sky = sort_upn[find_pos]
	spectrum_number_plot_order_sky = output_spectrum_number[idx_plot_order_sky]
	return spectrum_number_plot_order_upGREAT, spectrum_number_plot_order_sky


def Tmb2Jy(Tmb):
	#convert T_mb (K) to S (Jy)
	#https://www-sofia.atlassian.net/wiki/spaces/OHFC8/pages/1703955/6.+GREAT#6.1.2.1---Imaging-Sensitivities
	eta_mb = 0.67 #average
	Tr = Tmb*eta_mb #below Eq 6-5
	eta_fss = 0.97
	Ta = Tr*eta_fss #Eq 6-5
	S = 971*Ta #Jy, Eq 6-7
	return S

def Jy2Tmb(S):
	#convert S (Jy) to T_mb (K)
	#https://www-sofia.atlassian.net/wiki/spaces/OHFC8/pages/1703955/6.+GREAT#6.1.2.1---Imaging-Sensitivities
	Ta = S/971 #K, Eq 6-7
	eta_fss = 0.97
	Tr = Ta/eta_fss #Eq 6-5
	eta_mb = 0.67 #average
	Tmb = Tr/eta_mb #K, below Eq 6-5
	return Tmb








