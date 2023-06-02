#!/usr/bin/env python3

print('\tread in packages')
import numpy as np 
import xarray as xr
import pickle as pk
import natsort as ns 
import glob 
import sys

case = sys.argv[1]

print('\tcase -',case)

wd = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'

print('\treading in data')
files = ns.natsorted(glob.glob(wd+case+'/atm/hist/'+case+'.cam.h0.*'))
data = xr.open_mfdataset(files)

lat = data['lat'].values
lon = data['lon'].values

tau = data['cosp_tau'].values
plevs = data['cosp_prs'].values

print('\taveraging variables')

print('\t\t- surface temp')
tas = data['TS'].values
length = int(len(tas[:,0,0])/12)
tas = np.nanmean(tas.reshape(length,12,48,96)[-30:],axis=0)

print('\t\t- ISCCP cloud fraction')
cl = data['FISCCP1_COSP'].values
cl = np.nansum(cl.reshape(length,12,7,7,48,96),axis=(2,3))
cl = np.nanmean(cl[-30:],axis=0)

print('\t\t- Net TOA flux')
fnet = data['FSNT'].values-data['FLNT'].values
fnet = fnet.reshape(length,12,48,96)
fnet = np.nanmean(fnet[-30:],axis=0)

print('\t\t- Total Sky Surface Albedo')
salb = (data['FSDS'].values-data['FSNS'].values)/data['FSDS'].values
salb = salb.reshape(length,12,48,96)*100
salb = np.nanmean(salb[-30:],axis=0)

print('\t\t - Precipitation')
precip = data['PRECL'].values+data['PRECC'].values
precip = precip.reshape(length,12,48,96)
precip = np.nanmean(precip[-30:],axis=0)

print('\t\t - Sea Ice')
ice = data['ICEFRAC'].values
ice = np.nanmean(ice.reshape(length,12,48,96)[-30:],axis=0)

print('\taggregate variables')

output_full = {}
output_full['tas'] = tas
output_full['cl'] = cl
output_full['fnet'] = fnet
output_full['salb'] = salb
output_full['precip'] = precip
output_full['ice'] = ice
output_full['tau'] = tau
output_full['ctp'] = plevs
output_full['lat'] = lat
output_full['lon'] = lon

print('\tsaving\n')
pk.dump(output_full,open(case+'_picon_diag.pk','wb'))
