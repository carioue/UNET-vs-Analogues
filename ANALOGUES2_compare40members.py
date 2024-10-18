import numpy as np
import xarray as xr
import pandas as pd
from pandas import*
import matplotlib.pyplot as plt
import os

""" This code compare the analogues of the 40 members and select for each target day the 20 smallests analogues """
PATH = os.getcwd()
     
TARGET = 'r1i1p1f1'

# Load CSV files calculated with ANALOGUES1_find_analogues.py (for every days for the Target member, it contains the dates of its 20 analogues)
r10i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r10i1p1f1.csv',header=None) 
r11i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r11i1p1f1.csv',header=None) 
r12i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r12i1p1f1.csv',header=None) 
r13i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r13i1p1f1.csv',header=None) 
r14i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r14i1p1f1.csv',header=None) 
r15i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r15i1p1f1.csv',header=None) 
r16i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r16i1p1f1.csv',header=None) 
r17i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r17i1p1f1.csv',header=None) 
r18i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r18i1p1f1.csv',header=None) 
r19i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r19i1p1f1.csv',header=None) 
r20i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r20i1p1f1.csv',header=None) 
r21i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r21i1p1f1.csv',header=None) 
r22i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r22i1p1f1.csv',header=None) 
r23i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r23i1p1f1.csv',header=None) 
r24i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r24i1p1f1.csv',header=None) 
r25i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r25i1p1f1.csv',header=None) 
r26i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r26i1p1f1.csv',header=None) 
r27i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r27i1p1f1.csv',header=None) 
r28i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r28i1p1f1.csv',header=None) 
r29i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r29i1p1f1.csv',header=None) 
r30i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r30i1p1f1.csv',header=None) 
r31i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r31i1p1f1.csv',header=None) 
r32i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r32i1p1f1.csv',header=None) 
r33i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r33i1p1f1.csv',header=None) 
r34i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r34i1p1f1.csv',header=None) 
r35i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r35i1p1f1.csv',header=None) 
r36i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r36i1p1f1.csv',header=None) 
r37i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r37i1p1f1.csv',header=None) 
r38i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r38i1p1f1.csv',header=None) 
r39i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r39i1p1f1.csv',header=None) 
r40i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r40i1p1f1.csv',header=None) 
r41i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r41i1p1f1.csv',header=None) 
r43i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r43i1p1f1.csv',header=None) 
r44i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r44i1p1f1.csv',header=None) 
r45i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r45i1p1f1.csv',header=None) 
r46i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r46i1p1f1.csv',header=None) 
r47i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r47i1p1f1.csv',header=None) 
r48i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r48i1p1f1.csv',header=None) 
r49i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r49i1p1f1.csv',header=None) 
r50i1p1f1 = read_csv(PATH+'/data/TARGET'+str(TARGET)+'_1950_2022_r50i1p1f1.csv',header=None) 

files=[r10i1p1f1,r11i1p1f1,r12i1p1f1,r13i1p1f1,r14i1p1f1,r15i1p1f1,r16i1p1f1,r17i1p1f1,r18i1p1f1,r19i1p1f1,r20i1p1f1,r21i1p1f1,r22i1p1f1,r23i1p1f1,r24i1p1f1,r25i1p1f1,r26i1p1f1,r27i1p1f1,r28i1p1f1,r29i1p1f1,r30i1p1f1,r31i1p1f1,r32i1p1f1,r33i1p1f1,r34i1p1f1,r35i1p1f1,r36i1p1f1,r37i1p1f1,r38i1p1f1,r39i1p1f1,r40i1p1f1,r41i1p1f1,r43i1p1f1,r44i1p1f1,r45i1p1f1,r46i1p1f1,r47i1p1f1,r48i1p1f1,r49i1p1f1,r50i1p1f1]
name_files=["r10i1p1f1","r11i1p1f1","r12i1p1f1","r13i1p1f1","r14i1p1f1","r15i1p1f1","r16i1p1f1","r17i1p1f1","r18i1p1f1","r19i1p1f1","r20i1p1f1","r21i1p1f1","r22i1p1f1","r23i1p1f1","r24i1p1f1","r25i1p1f1","r26i1p1f1","r27i1p1f1","r28i1p1f1","r29i1p1f1","r30i1p1f1","r31i1p1f1","r32i1p1f1","r33i1p1f1","r34i1p1f1","r35i1p1f1","r36i1p1f1","r37i1p1f1","r38i1p1f1","r39i1p1f1","r40i1p1f1","r41i1p1f1","r43i1p1f1","r44i1p1f1","r45i1p1f1","r46i1p1f1","r47i1p1f1","r48i1p1f1","r49i1p1f1","r50i1p1f1"]

for i in files:
    i.columns=['cible','analogue','distance']

#add a column with the name of the member
name=[]
for j in range(len(name_files)):
    name=[name_files[j] for a in range(len(files[0]['cible']))]
    files[j]['membre']=name
    name=[]

#for each day find the 20 smallest distances among the 20*40 analogues    
n=0
table = pd.concat([files[i][n:n+20] for i in range(len(files))])
smallest_distances = table.sort_values(by='distance')[0:20] 
smallest_distances_all_day = smallest_distances   

n=20
while n<len(files[0]['cible'])-20:  
    table = pd.concat([files[i][n:n+20] for i in range(len(files))])
    smallest_distances = table.sort_values(by='distance')[0:20]
    smallest_distances_all_day = pd.concat([smallest_distances_all_day,smallest_distances])
    n=n+20 

smallest_distances_all_day.to_csv(PATH+'/data/TARGET'+str(TARGET)+'_analogues_40membres_1950_2022.csv')
    
    
