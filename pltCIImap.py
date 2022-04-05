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
import matplotlib.pyplot as plt
from astropy.visualization import (ManualInterval, ImageNormalize, LogStretch, AsinhStretch, LinearStretch)
import sys
import upGREATUtils


plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

map_type = args.map_type
map_types = ['mom0', 'mom1', 'mom2', 'peak', 'vcen', 'vcen_cgrad', 'fwhm', 'epeak', 'evcen', 'efwhm']

if map_type not in map_types:
	print('Error: map type \''+map_type+'\' not valid! Valid options are %s' %map_types)
	sys.exit(1)

if map_type == 'mom0':
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
	if map_type in ['peak', 'vcen', 'vcen_cgrad', 'fwhm']:
		snr_mask_level = args.addtl_mask_level
	else:
		snr_mask_level = np.nan
else:
	snr_mask_level = np.nan

#specify the colormap for plotting
if args.cmap:
	cmap = args.cmap
else:
	cmap = 'gray_r'

#open the fits file
fname = 'M82_CII_map_'+map_type+'_'+masksuff
data = fits.open('../Data/Disk_Map/'+fname+'.fits')
hdr = data[0].header
wcs = WCS(hdr)
data = data[0].data
bmaj,bmin,bpa = upGREATUtils.beam_from_header(hdr)

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


#plot the map
fig=plt.figure(1,figsize=(14,10))
plt.clf()
ax = plt.subplot(projection=wcs)
im = ax.imshow(data,cmap=cmap,origin='lower',norm=norm)
ax.coords[0].set_major_formatter('hh:mm:ss.s')
ax.coords[0].set_separator(('$^{\mathrm{h}}$','$^{\mathrm{m}}$','$^{\mathrm{s}}$'))
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)
ax.coords[0].set_minor_frequency(4)
ax.coords[1].set_minor_frequency(4)
ax.set_xlabel('R.A. (J2000)')
ax.set_ylabel('Decl. (J2000)')
cb = plt.colorbar(im,extend=cb_ext,shrink=0.75)
cb.set_label(map_type_str)
plt.savefig('../Plots/Center_Maps/'+fname+'.pdf',bbox_inches='tight')
plt.close()


