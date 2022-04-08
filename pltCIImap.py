this_script = __file__

import argparse

parser=argparse.ArgumentParser(
	description='''Plot [CII] map in disk from SOFIA.''',
	epilog='''Author: R. C. Levy - Last updated: 2022-04-05''')
parser.add_argument('map_type', type=str, help='Which map type to plot. Options are: mom0, mom1, mom2, peak, vcen, vcen_cgrad, fwhm, epeak, evcen, efwhm')
parser.add_argument('--mask_level', type=float, help='SNR mask level to use (default: no masking)')
parser.add_argument('--addtl_mask_level', type=float, help='Additional masking to use, based on associated error map. This is only valid for map_type = [peak, vcen, vcen_cgrad, fwhm]')
parser.add_argument('--cmap', type=str, help='Colormap to use for plotting (default: grayscale)')
parser.add_argument('--norm', type=str, help='Normalization type, linear, log, asinh (default:linear)')
parser.add_argument('--clim', type=float, nargs = 2, help='Colorbar limits: cmin cmax')
args=parser.parse_args()

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (ManualInterval, ImageNormalize, LogStretch, AsinhStretch, LinearStretch)
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Circle)
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import sys
import seaborn as sns
import upGREATUtils
from regions import Regions
import pandas as pd


plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 24

map_type = args.map_type
map_types = ['mom0', 'mom1', 'mom2', 'intinten', 'peak', 'vcen', 'vcen_cgrad', 'fwhm', 'eintinten', 'epeak', 'evcen', 'efwhm']

if map_type not in map_types:
	print('Error: map type \''+map_type+'\' not valid! Valid options are %s' %map_types)
	sys.exit(1)

if (map_type == 'mom0') or (map_type == 'intinten'):
	map_type_str = 'Integrated Intensity (K km s$^{-1}$)'
elif (map_type == 'peak') or (map_type == 'epeak'):
	map_type_str = 'Peak Intensity (K)'
elif (map_type == 'fwhm') or (map_type == 'efwhm') or (map_type == 'mom2'):
	map_type_str = 'FWHM Velocity Dispersion (km s$^{-1})$'
else:
	map_type_str = 'Velocity (km s$^{-1})$'

#open the file with the specified pre-masking
if args.mask_level:
	level = args.mask_level
	masksuff = 'maskSNR'+str(int(level))
else:
	level = np.nan
	masksuff = 'nomask'

#mask data more based on the SNR of its error map
if args.addtl_mask_level:
	if map_type in ['intinten', 'peak', 'vcen', 'vcen_cgrad', 'fwhm']:
		snr_mask_level = args.addtl_mask_level
	else:
		snr_mask_level = np.nan
else:
	snr_mask_level = np.nan

#specify the colormap for plotting
if args.cmap:
	cmap = cm.get_cmap(args.cmap)
else:
	#cmap = 'gray_r'
	cmap = ListedColormap(sns.color_palette('mako_r',256, desat=0.8))

#open the fits file
fname = 'M82_CII_map_'+map_type+'_'+masksuff
data = fits.open('../Data/Disk_Map/Moments/'+fname+'.fits')
hdr = data[0].header
wcs = WCS(hdr)
data = data[0].data
bmaj = hdr['BMAJ']*3600 #arcsec
bmaj = hdr['BMIN']*3600 #arcsec
bpa = hdr['BPA'] #deg

#do additional masking
if np.isnan(snr_mask_level)==False:
	if map_type == 'vcen_cgrad':
		efname = '../Data/Disk_Map/M82_CII_map_evcen_'+masksuff
	else:
		efname = '../Data/Disk_Map/M82_CII_map_e'+map_type+'_'+masksuff
	edata = fits.open(efname+'.fits')[0].data
	snr = np.abs(data)/np.abs(edata)
	data[snr<snr_mask_level]=np.nan
	fname = 'M82_CII_map_'+map_type+'_'+masksuff+'_'+'maskmapSNR'+str(int(snr_mask_level))

#set up plot normalization
if args.norm:
	norm_type = args.norm
else:
	norm_type = 'linear'
if args.clim:
	clim = args.clim
	if (clim[0] > np.nanmin(data)) & (clim[1] < np.nanmax(data)):
		cb_ext = 'both'
	elif clim[0] > np.nanmin(data):
		cb_ext = 'min'
	elif clim[1] < np.nanmax(data):
		cb_ext = 'max'
	else:
		cb_ext = 'neither'
else:
	clim = (np.nanmin(data),np.nanmax(data))
	cb_ext = 'neither'

if norm_type.casefold() == 'asinh'.casefold():	
	norm=ImageNormalize(data,stretch=AsinhStretch(0.0998),interval=ManualInterval(vmin=clim[0],vmax=clim[1]))
elif norm_type.casefold() == 'log'.casefold():
	norm=ImageNormalize(data,stretch=LogStretch(),vmin=clim[0],vmax=clim[1])
else:
	norm=ImageNormalize(data,stretch=LinearStretch(),interval=ManualInterval(vmin=clim[0],vmax=clim[1]))


#open the CO data to use its wcs
hdr_co = fits.open('../Data/Ancillary_Data/M82.CO.NOEMA+30m.peak.fits')[0].header
wcs_co = WCS(hdr_co)
arcsec2pix = hdr_co['CDELT2']*3600


#crop to fully sampled map region
center_AOR = SkyCoord('9h55m52.7250s +69d40m45.780s')
ang_map = -20. #deg
fullsamp_x = 185.4 #arcsec
fullsamp_y = 65.4 #arcsec
xo,yo = wcs_co.all_world2pix(center_AOR.ra.value,center_AOR.dec.value,1,ra_dec_order=True)
xl = fullsamp_x/arcsec2pix
yl = fullsamp_y/arcsec2pix
xx = np.array([np.ceil(-xl/2),np.floor(xl/2)]).astype(int)
yy = np.array([np.ceil(-yl/2),np.floor(yl/2)]).astype(int)
xlim = (xx*np.cos(np.radians(ang_map))-yy*np.sin(np.radians(ang_map)))+xo
ylim = np.flip((xx*np.sin(np.radians(ang_map))-yy*np.cos(np.radians(ang_map)))+yo)

#add beam and scale bar
x_text = xlim[0]+0.05*(xlim[1]-xlim[0])
y_text = ylim[0]+0.075*(ylim[1]-ylim[0])
x_text_sb = xlim[0]+0.95*(xlim[1]-xlim[0])
y_text_sb = ylim[0]+0.95*(ylim[1]-ylim[0])
bmaj_pix = hdr['BMAJ']*3600/arcsec2pix
bmin_pix = hdr['BMIN']*3600/arcsec2pix
bpa = hdr['BPA']
dist = 3.5E6 #pc
sb_pc = 200 #pc
sb_pc_pix = sb_pc/dist*206265/arcsec2pix
sb_pc_text = str(int(sb_pc))+' pc'

#plot the map
fig=plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs_co)
im = ax.imshow(data,cmap=cmap,origin='lower',norm=norm,transform = ax.get_transform(wcs))
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.set_xlabel('R.A. (J2000)')
ax.set_ylabel('Decl. (J2000)')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
cb = plt.colorbar(im,extend=cb_ext,shrink=0.7)
cb.set_label(map_type_str)
b1=ax.add_patch(Ellipse((x_text,y_text), bmaj_pix, bmin_pix, bpa, ec='None', fc='k'))
ax.plot([x_text_sb,x_text_sb-sb_pc_pix],[y_text_sb,y_text_sb],'-',color='k',lw=3)
ax.text(np.mean([x_text_sb,x_text_sb-sb_pc_pix]),y_text_sb*0.985,sb_pc_text,
	color='k',ha='center',va='top',fontsize=plt.rcParams['font.size']-2)
plt.savefig('../Plots/Center_Maps/'+fname+'.pdf',bbox_inches='tight',metadata={'Creator':this_script})


#now add the corresponding outflow data
n_pixels = 7
center = SkyCoord('9h55m52.7250s','+69d40m45.780s')
sp_offset = (29.152,-80.093) #arcsec
center_sp = SkyCoord(center.ra+sp_offset[0]*u.arcsec,center.dec+sp_offset[1]*u.arcsec)
array_rot = 70. #deg
fp = Regions.read('../Data/upGREAT_Footprint_Regions/SOFIA_upGREAT_LFA_regions_Cen'+center_sp.to_string('hmsdms').replace(' ','')+'_Rot'+str(array_rot)+'.reg')
reg_order = [0,5,6,1,2,3,4]

for i in range(n_pixels):
	this_fp = fp[i]
	RA = this_fp.center.ra.value
	Dec = this_fp.center.dec.value
	RA_pix,Dec_pix = wcs_co.all_world2pix(RA,Dec,1,ra_dec_order=True)
	#open the fit params
	p = pd.read_csv('../Data/Outflow_Pointings/CII_GaussianFits/outflow1_pix'+str(reg_order[i])+'_gaussianfitparams.csv')
	if map_type == 'peak':
		spec_val = p['Peak'].values[0]
	elif map_type == 'epeak':
		spec_val = p['ePeak'].values[0]
	elif (map_type == 'vcen') or (map_type=='vcen_cgrad') or (map_type=='mom1'):
		spec_val = p['Vcen'].values[0]
	elif (map_type == 'evcen') or (map_type == 'emom1'):
		spec_val = p['eVcen'].values[0]
	elif (map_type == 'fwhm') or (map_type == 'mom2'):
		spec_val = p['FWHM'].values[0]
	elif (map_type == 'efwhm') or (map_type == 'emom2'):
		spec_val = p['eFWHM'].values[0]
	elif (map_type == 'mom0') or (map_type=='intinten'):
		spec_val = p['IntInten'].values[0]
	elif map_type=='eintinten':
		spec_val = p['eIntInten'].values[0]
	else:
		spec_val = np.inf
	#find the color this corresponds to in the current colormap
	cmap_frac = (spec_val-clim[0])/(clim[1]-clim[0])*256
	spec_col = cmap(int(np.round(cmap_frac)))[0:3]
	ax.add_patch(Circle((RA_pix,Dec_pix), bmaj_pix/2, ec='k', fc=spec_col, lw=1.))


new_ymin = ylim[0]-5*bmaj_pix
new_y_text = new_ymin+0.05*(ylim[1]-new_ymin)
ax.set_ylim(new_ymin)
b1.remove()
cb.remove()
ax.add_patch(Ellipse((x_text,new_y_text), bmaj_pix, bmin_pix, bpa, ec='None', fc='k'))
cb = plt.colorbar(im,extend=cb_ext)
cb.set_label(map_type_str)
plt.savefig('../Plots/Center_Maps/'+fname+'_outflow.pdf',bbox_inches='tight',metadata={'Creator':this_script})



plt.close()


