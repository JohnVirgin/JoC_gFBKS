#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pk
from scipy.interpolate import griddata
import sys
import multiprocessing as mp
import time
from itertools import repeat

def interpolation(data,native_plevs,cmip_plevs):
        data_interp = np.zeros([12,17,48,96])
        for i in range(12):
            for j in range(48):
                for k in range(96):

                    profile = np.squeeze(data[i,:,j,k])
                    data_interp[i,:,j,k] = griddata(\
                        np.squeeze(native_plevs[i,:,j,k]),profile,cmip_plevs,method="nearest")
        
        return data_interp

if __name__ ==  '__main__':

    print('Setting up worker pools using',mp.cpu_count(),'cpu cores')
    cpus = mp.cpu_count()
    Pools = mp.Pool(cpus)

    print('read in data')

    compset = sys.argv[1]
    wd = '/Volumes/eSSD0/Papers/JoC_gFBK/Data/'

    data = pk.load(open(wd+'Outputs/'+compset+'_diag_grid.pk','rb'))
    ta = data.pop('ta')
    q = data.pop('lnQ')
    rh = data.pop('rh')

    print('read in kernel dimensions')

    cmip_plevs = np.asarray([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    native_plevs = pk.load(open(wd+'plevs_T31.pk','rb'))

    print('vertical interpolation')
    start_time = time.time()

    ta_flatten = [i for i in ta]
    q_flatten = [i for i in q]
    rh_flatten = [i for i in rh]

    print(f'Sim length? (years) - {len(ta_flatten)}')

    ta_interp = Pools.starmap(interpolation,zip(ta_flatten,repeat(native_plevs),repeat(cmip_plevs)))
    q_interp = Pools.starmap(interpolation,zip(q_flatten,repeat(native_plevs),repeat(cmip_plevs)))
    rh_interp = Pools.starmap(interpolation,zip(rh_flatten,repeat(native_plevs),repeat(cmip_plevs)))

    end_time = time.time() - start_time
    print(round(end_time/60,2), 'minutes for complete interpolation to finish')

    Pools.close()
    Pools.join()

    print('Rebuilding...')
    ta_rebuild = np.stack(ta_interp[:],axis=0)
    q_rebuild = np.stack(q_interp[:],axis=0)
    rh_rebuild = np.stack(rh_interp[:],axis=0)

    print('Final shapes')
    print(f'\t- {ta_rebuild.shape}\n\t- {q_rebuild.shape}\n\t- {rh_rebuild.shape}')

    print('Done, Saving to new files')
    data['ta_int'] = ta_rebuild
    data['lnQ_int'] = q_rebuild
    data['rh_int'] = rh_rebuild

    pk.dump(data,open(wd+'Outputs/'+compset+'_diag_grid_int.pk','wb'))