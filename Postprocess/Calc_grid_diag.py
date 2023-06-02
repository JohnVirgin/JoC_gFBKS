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
tas = tas.reshape(length,12,48,96)

print('\t\t- air temp')
ta = data['T'].values
ta = ta.reshape(length,12,26,48,96)

print('\t\t- cloud radiative effects')

print('\t\t- ISCCP cloud fraction')
cl = data['FISCCP1_COSP'].values
cl = cl.reshape(length,12,7,7,48,96)

print('\t\t- Net longwave TOA flux')
flnt = data['FLNT'].values
flnt = flnt.reshape(length,12,48,96)

print('\t\t- Net shortwave TOA flux')
fsnt = data['FSNT'].values
fsnt = fsnt.reshape(length,12,48,96)

print('\t\t- Net TOA flux')
fnet = data['FSNT'].values-data['FLNT'].values
fnet = fnet.reshape(length,12,48,96)

print('\t\t- Clear sky Net TOA flux')
fnetc = data['FSNTC'].values-data['FLNTC'].values
fnetc = fnetc.reshape(length,12,48,96)

print('\t\t- Total Sky Surface Albedo')
salb = (data['FSDS'].values-data['FSNS'].values)/data['FSDS'].values
salb = salb.reshape(length,12,48,96)*100

print('\t\t- Clear Sky Surface Albedo')
csalb = (data['FSDSC'].values-data['FSNSC'].values)/data['FSDSC'].values
csalb = csalb.reshape(length,12,48,96)*100

print('\t\t- Specific Humidity')
q = data['Q'].values
q = np.log(q.reshape(length,12,26,48,96))

print('\t\t- Surface Pressure')
ps = data['PS'].values
ps = ps.reshape(length,12,48,96)/100

print('\taggregate variables')

output_full = {}
output_full['tas'] = tas
output_full['ta'] = ta
output_full['isccp'] = cl
output_full['flnt'] = flnt
output_full['fsnt'] = fsnt
output_full['fnet'] = fnet
output_full['fnetc'] = fnetc
output_full['salb'] = salb
output_full['csalb'] = csalb
output_full['lnQ'] = q
output_full['ps'] = ps
output_full['tau'] = tau
output_full['ctp'] = plevs
output_full['lat'] = lat
output_full['lon'] = lon

print('\tsaving\n')
pk.dump(output_full,open(case+'_diag_grid.pk','wb'))