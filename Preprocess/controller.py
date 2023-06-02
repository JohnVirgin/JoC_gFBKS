#!/usr/bin/env python3

print('read in packages')
import numpy as np
import xarray as xr
import pickle as pk
import natsort as ns
import pandas as pd
import datetime as dt
from pathlib import Path
import glob
import sys

def controller(kp,ki,te,sumte,tsi):
    Fc = (kp*te) + (ki*sumte) #forcing correction, in % SOL
    delta_s0 = tsi*Fc #forcing correction, in Wm^-2
    new_tsi = tsi-(delta_s0/100) #new TSI in Wm^-2
    return new_tsi


print('Run Start Details')
case = sys.argv[1]
reftime = sys.argv[2]
ctl_root = sys.argv[3]
refyear = reftime[:-6].lstrip("0") #grab the first year, strip leading zeros if there are any

print(f'\tCase - {case}\n\tStart Year - {refyear}')

print('\nReading in the previous year output')
path = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'+case+'/atm/hist/'
files  = ns.natsorted(glob.glob(path+case+'.cam.h0.*')) #monthly history files only
year_n1 = files[-12:] #only the last 12 months

ds = xr.open_mfdataset(year_n1)

#surface temp, TSI, and datetime values
ts = ds['TREFHT'].values
tsi = np.round(np.mean(ds['sol_tsi'].values),3)
date = str(ds['date'][0].values)[:-4]
j = int(date)

#weights for area averaging
lat = ds['lat'].values
lon = ds['lon'].values
y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat,(lon.size,1)).T

#global, annual mean TAS from the previous year
ts_gam = np.average(np.mean(ts,axis=0),weights=coslat)

print('\nDiagnostics for this past year')
print(f'\tYear number - {j}\n\tGlobal Mean tas - {ts_gam},\n\tTSI - {tsi}')

d = {
    'year':[date],
    'TS': [ts_gam],
}

#sample gains
#kp = 1.2 #proportional gain, in units of %SOL K^-1
#ki = 1.8 #integral gain, in units of %SOL K^-1 Year ^-1

#CESM1.2 @T31 Atmos resolution and 3deg ocean resolution
kp = 0.9
ki = 1.02

#CanESM5
#kp = 0.87
#ki = 0.55

if date == refyear:
    print('\nThis is the first year of the simulation, creating new .csv with params')

    d['te'] = [0] #temperature error is zero for year 1
    te = d['te'][0]
    sumte = te #integrated error is equal to this years temperature error
    tsi_out = controller(kp,ki,te,sumte,tsi) #run controller, this will return a 0% change in the TSI by contruction
    d['tsi_out'] = [np.round(tsi_out,3)] #write out the TSI
    df_current = pd.DataFrame.from_dict(d) #make a dataframe
    df_current.to_csv(ctl_root+'/'+case+'-params.csv',index=False) #drop it into a csv

else:
    print('\nloading in params csv')

    df_all = pd.read_csv(ctl_root+'/'+case+'-params.csv')
    target = df_all.iloc[0]['TS'] #whatever the temp was from year 1
    te = d['TS']-target #temperature error from this year
    sumte = df_all['te'].sum()+te #integrated temperature error since controller start
    tsi_out = np.round(controller(kp,ki,te,sumte,df_all.iloc[0]['tsi_out']),3) #new TSI produced from controller
    
    #assign variables
    d['te'] = te 
    d['tsi_out'] = tsi_out
    
    df_current = pd.DataFrame.from_dict(d) #new dataframe with this years information
    df_all = df_all.append(df_current) #append to the bottom
    df_all.to_csv(ctl_root+'/'+case+'-params.csv', index=False) #write out to a CSV

#notes for the log file
print('\nController I/O')
print(f'\tTemperature Error - {te}\n\tIntegrated since year start - {sumte}')
print(f'\tGains\n\t>Proportional - {kp}\n\t>Integral - {ki}')
print(f'\tNew TSI - {tsi_out}')
print('\nDone')







