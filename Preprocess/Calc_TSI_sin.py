#!/usr/bin/env python3

import numpy as np
import xarray as xr
import cftime
from matplotlib import pyplot as plt
import sys

path = '/Volumes/eSSD0/Papers/GRL_SOLu/Data/Inputs/'
hist_tsi = xr.open_dataset(path+'CESM/forcing/SOLAR_TSI_Lean_1610-2007_annual_c090324.nc')

ry_s = sys.argv[1]
w = float(ry_s)

s0 = 1360.89
amp = 0.01*s0
period = 2*np.pi/w
nyrs = int(round(period,0))*2
x = np.arange(nyrs)
tsi = amp*np.sin(w*x)+s0

years_xr = []
date = np.zeros([nyrs])
for i in range(nyrs):
    years_xr.append(cftime.DatetimeNoLeap(i,7,1,0,0,0,0))
    date[i] = int(cftime.datetime.strftime(years_xr[i], '%Y%m%d'))

ready = np.asarray(years_xr)

nested = {
    "coords": {
        "time": {
            "dims":"time",
            "data":ready
            }
        },
    "data_vars": {
        "tsi": {
            "dims":"time",
            "data":tsi
        },
        "date":{
            "dims":"time",
            "data":date
        }
    }
}


dataset = xr.Dataset.from_dict(nested)
dataset.to_netcdf(path+'CESM/forcing/TSI_pm1per_'+ry_s+'rady_'+str(nyrs)+'yrs.nc')