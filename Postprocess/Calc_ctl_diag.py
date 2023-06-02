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

case = sys.argv[1]

path = '/scratch/c/cgf/jgvirgin/cesm1_2_2/archive/'+case+'/atm/hist/'

print('reading in data')
files  = ns.natsorted(glob.glob(path+case+'.cam.h0.*'))
ds = xr.open_mfdataset(files)

ts = ds['TREFHT'].values
tsi = ds['sol_tsi'].values
flut = ds['FLUT'].values
co2 = ds['co2vmr'].values

lat = ds['lat'].values
lon = ds['lon'].values

length = int(len(ts[:,0,0])/12)

ts_resh = np.mean(ts.reshape(length,12,48,96),axis=1)
tsi_resh = np.mean(tsi.reshape(length,12),axis=1)
flut_resh = np.mean(flut.reshape(length,12,48,96),axis=1)
co2_resh = np.mean(co2.reshape(length,12),axis=1)

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat[None,:,None],(length,1,lon.size))

print('Taking the global mean')
ts_gam = np.average(ts_resh,weights=coslat,axis=(1,2))
flut_gam = np.average(flut_resh,weights=coslat,axis=(1,2))

data = {
    'TSI': tsi_resh,
    'tas':ts_gam,
    'OLR':flut_gam,
    'CO2':co2_resh
}

print('saving')
pk.dump(data,open(case+'_ctl_diag.pk','wb'))
