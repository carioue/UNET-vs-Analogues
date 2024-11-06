import numpy as np
import xarray as xr
import pandas as pd
import os

""" This code reconstructs daily temperatures from the analogues  
As SLP data, raw temperature data are available thanks to CMIP6 working groups. Here temperatures are pre-processed to calculate temperature anomalies: routines from https://gitlab.com/ribesaurelien/france_study 
are used to compute non stationary normals.
"""

jours=[]
dates =xr.open_dataset('psl_day_MIROC6_unet_r5i1p1f1_noleapday.nc").sel(time=slice(f'{1880}-01-01',f'{2100}-12-31')).time
for i in dates:
	jours.append(str(i.values)[0:10])

anomalies=xr.open_dataset('anomalies_40membres.nc').tasano #load netcdf file containing temperature anomalies of the 40 members
name_files=["r10i1p1f1","r11i1p1f1","r12i1p1f1","r13i1p1f1","r14i1p1f1","r15i1p1f1","r16i1p1f1","r17i1p1f1","r18i1p1f1","r19i1p1f1","r20i1p1f1","r21i1p1f1","r22i1p1f1","r23i1p1f1","r24i1p1f1","r25i1p1f1","r26i1p1f1","r27i1p1f1","r28i1p1f1","r29i1p1f1","r30i1p1f1","r31i1p1f1","r32i1p1f1","r33i1p1f1","r34i1p1f1","r35i1p1f1","r36i1p1f1","r37i1p1f1","r38i1p1f1","r39i1p1f1","r40i1p1f1","r41i1p1f1","r43i1p1f1","r44i1p1f1","r45i1p1f1","r46i1p1f1","r47i1p1f1","r48i1p1f1","r49i1p1f1","r50i1p1f1"]

member_target = 'r5i1p1f1'
data_dates= pd.read_csv('TARGETr5i1p1f1_analogues_40membres_1950_2022.csv') #load file constructed with ANALOGUES2_compare40members.py 
data_dates = data_dates.drop('Unnamed: 0',axis=1)
	
reconstruction=[]

#for one target day, find the 20 analogue temperature anomaly maps and take the mean
for d in range(0,len(data_dates),20):
	cc=[anomalies[jours.index(data_dates['analogue'][i])+name_files.index(data_dates['membre'][i])*80665] for i in range(d,d+20)]
	conc_moy = xr.concat(cc, dim='time').mean(dim='time')
	reconstruction.append(conc_moy)

np.save('Trec_'+str(member_target)+'.npy', reconstruction)
