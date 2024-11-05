import numpy as np
import xarray as xr
import pandas as pd
import os


"""This code calculates analogues of each day in the TARGET member, comparing with other members.
For clarity, "target file" refers to the file containing days for which analogues are sought, while "analogue file" refers to the file where analogues are searched."""

TARGET = 'r5i1p1f1' 


def calcul_analogues(begin,end,member):
    #begin = first year in the target period (e.g., for 1950-1960, begin=1950 and end=1960)
    #end = last year in the target period
    #member = ID of the analogue members to compare (e.g., r10i1p1f1, r11i1p1f1, etc.)
    
    # Load the analogue file containing slp data
    data_psl = xr.open_dataset('psl_day_MIROC6_unet_'+str(member)+'_noleapday.nc') 
    
    # Calculate weights to correct for latitude-based distortions in high-latitude regions
    latitude = data_psl['lat'].values
    wgt = np.sqrt(np.cos(np.deg2rad(latitude))) 
    n_lat = len(data_psl['lat'].values)
    n_lon = len(data_psl['lon'].values)
    weights = np.tile(wgt[:, np.newaxis], n_lon)
    
    ntime = len(data_psl['time'].values)
    total_period = [] # List to store all days available in the analogue file
    for i in range(ntime):
        date = data_psl['time'].values
        total_period.append(str(date[i])[0:10])
     
    # Define time windows for the target period and analogue search period  
    begin_window = total_period.index(str(begin)+'-01-01') 
    end_window = total_period.index(str(end)+'-12-31')
    begin_ana = total_period.index('1880-01-01')  # Analogue search period start (1880)
    end_ana = total_period.index('2100-12-31') # Analogue search period end (2100)
    
    # Extract date and slp data for the analogue and target periods
    dates_analogues = xr.open_dataset('psl_day_MIROC6_unet_'+str(member)+'_noleapday.nc')['time'].values[begin_ana:end_ana+1]
    dates_analogues = np.datetime_as_string(dates_analogues,unit='D')
    
    psl_analogue = xr.open_dataset'psl_day_MIROC6_unet_'+str(member)+'_noleapday.nc')['psl'].values[begin_ana:end_ana+1] #analogue file
    psl_target = xr.open_dataset('psl_day_MIROC6_unet_'+str(TARGET)+'_noleapday.nc')['psl'].values[begin_window:end_window+1] #target file

    target_analogues = []  # List to store the results of analogue calculation
    
    # Find analogue days within a +/-15-day window around the target date each year
    for id_date,j in enumerate(total_period[begin_window:end_window+1]) :
        print(id_date)
        psl_comp=[] # List to store SLP maps for analogue dates
        date_ana= [] # List to store analogue dates
        id_d_ana = list(dates_analogues).index(j) # Index of target day within the analogue file
        
        for i in range(15):
            date_ana.append(list(dates_analogues[id_d_ana-i::-365])+list(dates_analogues[id_d_ana-i::365][1:])) #analogues are found on a +/-15days window centered on the target day (every year)
            date_ana.append(list(dates_analogues[id_d_ana+i+1::-365])+list(dates_analogues[id_d_ana+i+1::365][1:]))
            
            psl_comp.append(list(psl_analogue[id_d_ana-i::-365,:,:])+list(psl_analogue[id_d_ana-i::365,:,:][1:])) #slp maps corresponding to date_ana
            psl_comp.append(list(psl_analogue[id_d_ana+i+1::-365,:,:])+list(psl_analogue[id_d_ana+i+1::365,:,:][1:]))
           
        m=np.shape(date_ana)[0]
        n=np.shape(date_ana)[1] 
        
        psl_comp = np.array(psl_comp).reshape(m*n,n_lat,n_lon)
        date_ana = np.array(date_ana).reshape(m*n)
        
        # Apply latitude weights to both target and analogue SLP data
        weights3D=[weights for i in range(m*n)]
        target = psl_target[id_date,:,:]*weights
        psl_comp = psl_comp*weights3D
        
        # Calculate Euclidean distances between target SLP map and analogue maps
        distances = np.linalg.norm(target-psl_comp,axis=(1,2))
        
        id_dis = np.argsort(distances)[:20] # Keep only the 20 smallest distances
        
        # Record dates and distances of the closest analogues
        analogues_dates = date_ana[id_dis]
        analogues_dis = distances[id_dis]

    
        for a in range(len(analogues_dates)):
            target_analogues.append((j,analogues_dates[a],analogues_dis[a]))
    
    dataframe = pd.DataFrame(target_analogues)
    dataframe.to_csv('TARGET'+str(TARGET)+'_'+str(begin)+'_'+str(end)+'_'+str(member)+'.csv',mode='a',index=False, header=False)

calcul_analogues(1950,2022,'r10i1p1f1')

   
    
