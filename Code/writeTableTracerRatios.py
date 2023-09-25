#write out a .tex file of the CII, CO, and HI integrated intensity ratios and the CII/CO luminosity ratios
import numpy as np
import pandas as pd

n_pixels = 7

#open the integrated intensity file
dfi = pd.read_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams.csv')

#open the luminosity ratio file
dfl = pd.read_csv('../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams_luminosity_ratios.csv').drop(['Pixel Number'],axis=1)

#concat dfs
df = pd.concat([dfi,dfl],axis=1)

idx = np.where(np.isnan(df['ratio_Mom0_CO_CII'].values)==True)[0]

#write this table to a tex file
tex_tab_file = '../Data/Outflow_Pointings/CII_Moments/outflow1_momentparams_COHI.tex'
with open(tex_tab_file,'w') as fp:
	fp.write('\\begin{deluxetable}{cccc}\n')
	fp.write('\\tablecaption{Ratios of the integrated intensities and luminosities in the outflow.\\label{tab:outflowfits_HICO}}\n')
	# fp.write('\\tablehead{Pixel Number & ${\\rm log_{10}(I_{int,HI}/I_{int,[CII]})}$ & ${\\rm log_{10}(I_{int,CO}/I_{int,[CII]})}$}')
	fp.write('\\tablehead{Pixel Number & ${\\rm I_{int,HI}/I_{int,[CII]}}$ & ${\\rm I_{int,CO}/I_{int,[CII]}}$ & ${\\rm L_{[CII]}/L_{CO}}$}')
	fp.write('\n\\startdata\n')

	for j in range(n_pixels):
		#convert to strings
		if j in idx:
			ratio_hi_str = '---'
			ratio_co_str = '---'
			ratio_lum_str = '---'
		else:
			# ratio_hi_str = '%.2f $\\pm %.2f$' %(np.log10(df['ratio_Mom0_HI_CII'].values[j]),df['eratio_Mom0_HI_CII'].values[j]/(df['ratio_Mom0_HI_CII'].values[j]*np.log(10)))
			# ratio_co_str = '%.2f $\\pm %.2f$' %(np.log10(df['ratio_Mom0_CO_CII'].values[j]),df['eratio_Mom0_CO_CII'].values[j]/(df['ratio_Mom0_CO_CII'].values[j]*np.log(10)))
			ratio_hi_str = '%.1f $\\pm$ %.1f' %(df['ratio_Mom0_HI_CII'].values[j],df['eratio_Mom0_HI_CII'].values[j])
			ratio_co_str = '%.1f $\\pm$ %.1f' %(df['ratio_Mom0_CO_CII'].values[j],df['eratio_Mom0_CO_CII'].values[j])
			ratio_lum_str = '%1.f $\\pm$ %1.f' %(df['LCII_LCO'].values[j],df['e_LCII_LCO'].values[j])

		fp.write('%i & %s & %s & %s\\\\\n' 
			%(df['Pixel Number'].values[j],ratio_hi_str,ratio_co_str,ratio_lum_str))


	fp.write('\\enddata\n')
	fp.write('\\tablecomments{\\change{To form the ratios, all integrated intensities are in K~km~s$^{-1}$ units and all luminosities are in $L_\odot$ units.}} \n\n')
	fp.write('\\end{deluxetable}\n')