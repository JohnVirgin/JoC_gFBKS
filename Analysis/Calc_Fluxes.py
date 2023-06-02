
print('import packages')

import numpy as np
import pandas as pd
import pickle as pk
import sys
from scipy.interpolate import interp1d

print('Define functions')

def calc_dp(ps,plevs,p_top):

    if min(plevs) == plevs[0]:
        plevs = plevs
    else:
        plevs = plevs[::-1]

    Upper = 0
    Mid = 1
    Lower = 2

    dp = np.empty([plevs.size])
    dp[:] = np.nan

    def lower_boundary():
        for i,value in enumerate(plevs):
            if value > ps:
                    return i

        return plevs.size

    for i in range(lower_boundary()):

        if i == 0:
            dp[i] = -(p_top-(((plevs[1]+plevs[2])/2)-plevs[0]))

        elif Lower < lower_boundary():
            dp[i] = ((plevs[Mid]+plevs[Lower])/2)-((plevs[Upper]+plevs[Mid])/2)

            Upper += 1
            Mid += 1
            Lower += 1

        else:
            dp[i] = ps-(((plevs[lower_boundary()-1]+plevs[lower_boundary()-2])/2))

    return dp

def Calc_huss(ta,plevs):

    ta = ta-273.15 #convert temperature to degress celsius
    MWratio = 0.622 #molecular weight ratio of water vapour to dry air

    #equation 1: saturation vapour pressure with respect to water
    #the hard coded values here are coefficients for general usage
    Sat_P_water = (1.0007+(3.46e-6*plevs))*6.1121*(np.exp((17.502*ta)/(240.97+ta)))

    # saturation mixing ratio with respect to liquid water (g of water/kg of dry air)
    ws_liquid = MWratio*Sat_P_water/(plevs-Sat_P_water)

    #equation 2: saturation vapour pressure with respect to ice
    #the hard coded values here are coefficients for general usage
    Sat_P_ice = (1.0003+(4.18e-6*plevs))*6.1115*(np.exp((22.452*ta)/(272.55+ta)))

    # saturation mixing ratio with respect to ice (g of water/kg of dry air)
    ws_ice = MWratio*Sat_P_ice/(plevs-Sat_P_ice)

    ws = ws_liquid
    ws[ta<0] = ws_ice[ta<0]

    #saturation specific humidity (g/kg)
    Qs = ws/(1+ws)

    return Qs

ctl_file = sys.argv[1]
response_file = sys.argv[2]

wd = '/Volumes/eSSD0/Papers/JoC_gFBK/Data/'

print('Read in control and response data')

control = pk.load(open(wd+'Outputs/'+ctl_file+'_diag_grid_int.pk','rb'))
ctl_mean = {}
for keys in control.keys():
    ctl_mean[keys] = np.mean(control[keys][-30:],axis=0)

print(f"Check the shape of the control simulation means - {ctl_mean['tas'].shape}")

lat = control.pop('lat')
lon = control.pop('lon')
ctp = control.pop('ctp')
tau = control.pop('tau')

response = pk.load(open(wd+'Outputs/'+response_file+'_diag_grid_int.pk','rb'))
rlength = int(response['tas'][:,0,0,0].size)

del response['lat']
del response['lon']
del response['ctp']
del response['tau']
del response['talb']

print('Calculate the Climate Reponse')

ctl_stack = {}
for keys in control.keys():
    ctl_stack[keys] = np.moveaxis(np.tile(np.expand_dims(ctl_mean[keys],axis=-1),rlength),-1,0)

delta = {}
for var in response.keys():
    delta[var] = response[var]-np.moveaxis(np.tile(np.expand_dims(ctl_mean[var],axis=-1),rlength),-1,0)

print('Read in Radiative Kernels')
kernels = pk.load(open(wd+'Kernels/CAM3_Kernels_T31.pk','rb'))

for keys in kernels:
    if kernels[keys].ndim == 4:
        kernels[keys] = np.tile(kernels[keys][None,:,:,:,:], (rlength,1,1,1,1))
    else:
        kernels[keys] = np.tile(kernels[keys][None,:,:,:], (rlength,1,1,1))

print('Define Pressure levels and midpoint deltas, as well as tropopause height')
cmip_plevs = np.asarray([10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
plevs = np.tile(cmip_plevs[None,:,None,None],(12,1,48,96))
ps = np.nanmean(control['ps'][-30:],axis=0)

dp = np.zeros([12,17,48,96])
for i in range(12):
    for j in range(48):
        for k in range(96):
            dp[i,:,j,k] = calc_dp(ps = ps[i,j,k],plevs=cmip_plevs,p_top = min(cmip_plevs))

dp_100 = np.tile(dp[None,:,:,:,:],(rlength,1,1,1,1))/100

p_tropo_NH = np.linspace(300,100,24)
p_tropo_SH = np.linspace(100,300,24)
p_tropo = np.concatenate((p_tropo_NH,p_tropo_SH))
p_tropo_full = np.tile(p_tropo[None,None,None,:,None],(rlength,12,17,1,96))

print('Calculate the case-specific delta in specific humidity required to increase temp by 1K')

print('\t- huss for the baseline')
huss_base = Calc_huss(ctl_stack['ta_int'],plevs)

print('\t- huss for the response')
huss_resp = Calc_huss(response['ta_int'],plevs)

print('\t- rate of change of huss with respect to ta')
dqsdt = (huss_resp-huss_base)/delta['ta_int']

print('\t- relative humidity')
hurs = (1000*np.exp(ctl_stack['lnQ_int']))/huss_base

print('\t- delta huss with respect to ta')
dqdt = dqsdt*hurs

print('\t- log delta huss with respect to ta')
dlnqdt = dqdt/(1000*np.exp(ctl_stack['lnQ_int']))

delta['lnQ_norm'] = delta['lnQ_int']/dlnqdt

print('Create isothermal temperature profile response')
ta_iso = np.tile(delta['tas'][:,:,None,:,:],(1,1,17,1,1))
ta_dep = delta['ta_int']-ta_iso

print('Separate WV and T responses in stratosphere and troposphere portions')
q_strato = delta['lnQ_norm']*(plevs<p_tropo_full)
q_tropo = delta['lnQ_norm']*(plevs>=p_tropo_full)

ta_strato = delta['ta_int']*(plevs<p_tropo_full)
ta_dep_tropo = ta_dep*(plevs>=p_tropo_full)
ta_iso_tropo = ta_iso*(plevs>=p_tropo_full)

adj = {}

print('\t- Surface albedo')
adj['salb'] = np.nanmean(delta['salb']*kernels['Alb_TOA'],axis=1)

print('\t- Stratosphere temperature')
adj['ta_strato'] = np.nanmean(np.nansum(ta_strato*kernels['Ta_TOA']*dp_100,axis=2),axis=1)

print('\t- Lapse rate')
adj['lapse'] = np.nanmean(np.nansum(ta_dep_tropo*kernels['Ta_TOA']*dp_100,axis=2),axis=1)

print('\t- Planck')
surface_rad = delta['tas']*kernels['Ts_TOA']
iso_rad = np.nansum(ta_iso_tropo*kernels['Ta_TOA']*dp_100,axis=2)
adj['Planck'] = np.nanmean(surface_rad+iso_rad,axis=1)

print('\t- Stratosphere water vapour')
q_strato_lw = np.nansum(q_strato*kernels['WVlw_TOA']*dp_100,axis=2)
q_strato_sw = np.nansum(q_strato*kernels['WVsw_TOA']*dp_100,axis=2)
adj['q_strato'] = np.nanmean(q_strato_lw+q_strato_sw,axis=1)

print('\t- Troposphere water vapour')
q_tropo_lw = np.nansum(q_tropo*kernels['WVlw_TOA']*dp_100,axis=2)
q_tropo_sw = np.nansum(q_tropo*kernels['WVsw_TOA']*dp_100,axis=2)
adj['q_tropo'] = np.nanmean(q_tropo_lw+q_tropo_sw,axis=1)

print('Read in cloud kernels')
cld_kernel = pk.load(open(wd+'Kernels/cloud_kernels_T31.pk','rb'))
cld_kernel['LWkernel_swap'] = np.tile(np.swapaxes(cld_kernel['LWkernel'],1,2)[None,:,:,:,:,:], (rlength,1,1,1,1,1))

adj['cld_lw'] = np.nanmean(np.nansum(cld_kernel['LWkernel_swap']*delta['isccp'], axis=(2,3)),axis=1)
adj['cld_lw_lo'] = np.nanmean(np.nansum(cld_kernel['LWkernel_swap'][:,:,:2,:,:,:]*delta['isccp'][:,:,:2,:,:,:], axis=(2,3)),axis=1)
adj['cld_lw_hi'] = np.nanmean(np.nansum(cld_kernel['LWkernel_swap'][:,:,2:,:,:,:]*delta['isccp'][:,:,2:,:,:,:], axis=(2,3)),axis=1)

print('Remapping the SW kernel to appropriate values based on control albedo meridional climatology')

albcs = np.arange(0.0, 1.1, 0.5)

SWkernel_map = np.zeros([12, 7, 7, 48, 96])
for m in range(12):  # loop through months
    for la in range(46):  # loop through longitudes

        # pluck out a zonal slice of clear sky surface albedo
        alb_lon = np.nanmean(control['csalb'],axis=0)[m, la, :]/100

        #remap the kernel onto the same grid as the model output
        function = interp1d(albcs, cld_kernel['SWkernel'][m, :, :, la, :], axis=2, kind='linear')
        new_kernel_lon = function(alb_lon)
        SWkernel_map[m, :, :, la, :] = new_kernel_lon

SWkernel_map = np.tile(SWkernel_map[None,:,:,:,:,:],(rlength,1,1,1,1,1))

print('Calculating Shortwave cloud adjustment and setting values to zero in the polar night')
sundown = np.nansum(SWkernel_map, axis=(2, 3))
#set the SW feedbacks to zero in the polar night
night = np.where(sundown == 0)

cld_kernel['SWkernel_swap'] = np.swapaxes(SWkernel_map,2,3)

adj['cld_sw'] = np.nansum(cld_kernel['SWkernel_swap']*delta['isccp'], axis=(2,3))
adj['cld_sw_lo'] = np.nansum(cld_kernel['SWkernel_swap'][:,:,:2,:,:,:]*delta['isccp'][:,:,:2,:,:,:], axis=(2,3))
adj['cld_sw_hi'] = np.nansum(cld_kernel['SWkernel_swap'][:,:,2:,:,:,:]*delta['isccp'][:,:,2:,:,:,:], axis=(2,3))

adj['cld_sw'][night] = 0
adj['cld_sw_lo'][night] = 0
adj['cld_sw_hi'][night] = 0

adj['cld_sw'] = np.nanmean(adj['cld_sw'],axis=1)
adj['cld_sw_lo'] = np.nanmean(adj['cld_sw_lo'],axis=1)
adj['cld_sw_hi'] = np.nanmean(adj['cld_sw_hi'],axis=1)

pop_SWlo = adj.pop('cld_sw_lo')
pop_SWhi = adj.pop('cld_sw_hi')
pop_LWlo = adj.pop('cld_lw_lo')
pop_LWhi = adj.pop('cld_lw_hi')


print('Calculating instantaneous RF as a residual')

adj_sum = np.nansum(np.stack(list(adj.values()),axis=0),axis=0)

adj['IRF'] = np.nanmean(delta['fnet'],axis=1)-adj_sum

adj['cld_sw_lo'] = pop_SWlo
adj['cld_sw_hi'] = pop_SWhi
adj['cld_lw_lo'] = pop_LWlo
adj['cld_lw_hi'] = pop_LWhi

print('Calculating global means')

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat[None,:,None],(rlength,1,lon.size))

adj_gam = {}
for var in adj.keys():
    adj_gam[var] = np.average(adj[var],weights=coslat, axis=(1,2))

print('Checking final var list, shapes, and global mean values')

for var in adj.keys():
    print('\t- ',var,' - ',adj[var].shape,' - ',round(np.nanmean(adj_gam[var]),2))

print('Saving output')

pk.dump(adj,open(wd+'Outputs/EB/'+response_file+'_EB_Grid.pk','wb'))
pk.dump(adj_gam,open(wd+'Outputs/EB/'+response_file+'_EB_gam.pk','wb'))