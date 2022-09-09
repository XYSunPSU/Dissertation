"""
@author:
@file  : pred_to_nc.py
@time  : 2022/8/22 9:51 上午
@desc  : step 4: convert predition result to nc file
"""
import netCDF4
from netCDF4 import Dataset
import datetime
import pandas as pd
import numpy as np
from datetime import datetime
import os
pd.set_option('display.float_format', lambda x: '%.15f' % x)

def write_nc(file_path):
    """
    :param file_path: file named by date,ex:200608290030-200608292330
    :return: nc file with predition data
    """
    pred = netCDF4.Dataset(os.path.join(file_path, 'pred.nc'), 'w', format='NETCDF4')
    pred.createDimension('time', 24)
    pred.createDimension('leadtime', 24)
    pred.createDimension('longitude', 236)
    pred.createDimension('latitude', 366)
    pred.createVariable('time', np.float64, ('time'))
    pred.createVariable('leadtime', np.float64, ('leadtime'))
    pred.createVariable('longitude', np.float32, ('longitude'))
    pred.createVariable('latitude', np.float32, ('latitude'))
    pred.createVariable('a04203', np.float32, ('time', 'latitude', 'longitude'))
    # 1.time
    base_time = datetime.strptime('1970-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    data_all = pd.read_csv(os.path.join(file_path, 'data.csv'))
    dates = [datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S") for i in list(data_all['time'].unique())]
    time1 = [(i - base_time) for i in dates]
    time = [j.days * 24 + j.seconds / 3600 for j in time1]
    pred.variables['time'][:] = time
    # 2.leatime
    leadtime = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5,
                19.5, 20.5, 21.5, 22.5, 23.5]
    pred.variables['leadtime'][:] = leadtime
    # 3.longitude
    longitude = pd.read_csv('./result/longitude.csv')
    pred.variables['longitude'][:] = longitude['longitude'].values
    # 4.latitude
    latitude = pd.read_csv('./result/latitude.csv')
    pred.variables['latitude'][:] = latitude['latitude'].values
    # 5.a04203
    a04203_orig = pd.read_csv(os.path.join(file_path, 'pred.csv'), index_col=0)
    temp1 = a04203_orig.reindex(list(range(2073024)), fill_value=-1073741800.0)
    a04203_data = np.array(temp1['pred'].values.tolist()).reshape(24, 366, 236)
    temp2 = a04203_orig
    temp2['pred'] = False
    temp2 = temp2.reindex(list(range(2073024)), fill_value=True)
    a04203_mask = np.array(temp2['pred'].values.tolist()).reshape(24, 366, 236)
    a04203 = np.ma.array(a04203_data, mask=a04203_mask, fill_value=-1073741800.0)
    pred.variables['a04203'][:] = a04203
    pred.close()
def get_file(path):
    """
    :param path: path
    :return: sub file list
    """
    file = []
    for root, dirs, files in os.walk(path):
        file.append(dirs)
    return file[0]

if __name__== '__main__':
    root_path = './result'
    file_names = get_file(root_path)
    print(file_names)
    for file_name in file_names:
        file_path = os.path.join(root_path,file_name)
        write_nc(file_path)
