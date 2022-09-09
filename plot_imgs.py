
"""
@author:
@file  : plot_imgs.py
@time  : 2022/8/20 4:29 下午
@desc  : step 5: plot predition、true、input images
"""

import numpy as np
import matplotlib
matplotlib.use('agg') # This is needed to suppress the need for a DISPLAY
import matplotlib.pyplot as plt
import cftime as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from fileselector import file_selector
from netCDF4 import Dataset
import os

def hh1970(yy,mm,dd,hh):
  return 24.*360.*(yy-1970)+30.*24.*(mm-1)+24.*(dd-1)+hh

def ymd1970(hh):
  hd=hh/24.
  yy=np.trunc(hd/360.)
  mm=np.trunc((hd-yy*360.)/30.)+1
  dd=np.trunc(hd-yy*360.-(mm-1)*30)+1
  yy=yy+1970
  return yy,mm,dd


def plot_images(stash,data_path,result_path):
  """
  :param stash: 'a01208' or 'a04203'
  :param data_path: nc file path
  :return: images and txt file
  """
  lon1,lat1=2.2+360.,13.6
  fc="C"
  clevs=[.1,5.,10.,20.]
  xlims=[-15.,15.] # west af
  ylims=[4.,24.] # west af]
  xlims=[35.,57.] # madagascar
  ylims=[-25.,-5.] #
  rcolors=['blue','yellow','red']

  #output a04203_A1hr_mean_aj514_25-4km_199707010030-199707012330.nc
  #input a01208_A1hr_mean_aj514_4-25km_199707010030-199707012330.nc

  time1h=np.empty([0])
  rain1h=np.empty([0]) # accumulated rain over 1h
  series=np.empty([0])

  yearstart,monthstart,daystart = 1997, 7, 1
  hour1970=hh1970(yearstart,monthstart,daystart,0)

  yearend,monthend,dayend = 1997, 7, 1
  hourend1970=hh1970(yearend,monthend,dayend+1,0)

  year,month,day=ymd1970(hour1970)

  nc_fid = Dataset(data_path, 'r')

  lat = nc_fid.variables['latitude'][:]  # extract/copy the data
  lon = nc_fid.variables['longitude'][:]
  time = nc_fid.variables['time'][:]
  field = nc_fid.variables[stash][:]
  print ('min, max '),np.min(field),np.max(field)

  dlat=np.float(lat[3]-lat[2])
  dlon=np.float(lon[3]-lon[2])
  ilat,ilon=1,1
  lati=lat[ilat]
  loni=lon[ilon]
  ilat=ilat+int((lat1-lati)/dlat+0.5) # check this especially pos/neg jumps
  ilon=ilon+int((lon1-loni)/dlon+0.5)
  print(ilat,lat[ilat],ilon,lon[ilon])

  time1h=np.append(time1h,time)
  series=np.append(series,field[:,ilat,ilon])

  for ig in range(np.size(time)):
    gfile = os.path.join(result_path,str(int(hour1970+ig))+'_'+fc+'.png')
    hh=(int(hour1970)+ig)%24
    #hourstr=str(np.where(hh<10,'0'+str(hh),str(hh)))
    year1,month1,day1=ymd1970(hour1970+ig)
    hourstr=str(hh).zfill(2)
    daystr=str(int(day1)).zfill(2)
    monthstr=str(int(month1)).zfill(2)
    datestr=hourstr+'00 '+daystr+' '+monthstr+' '+str(int(year1))
    #print (day,daystr,month,monthstr)
    #print (datestr)
    #print gfile
    f=plt.figure(1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.axis(np.concatenate((xlims,ylims),axis=0))
    #plt.contourf(lon-360.,lat,3600.*field[ig,:,:],clevs,colors=rcolors,extend='max')
    plt.contourf(lon-360.,lat,field[ig,:,:])
    #plt.contourf(lon-360.,lat,3600.*field[ig,:,:],clevs,cmap="jet")
    ax.add_feature(cfeature.COASTLINE,lw=0.5, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS,  lw=0.5, edgecolor='gray')
    #plt.text(xlims[0]+1.,ylims[1]-1.,datestr)

    plt.savefig(gfile)

    plt.clf()
  hour1970=int(time[-1]+1.)
  model = 'CP4'
  if model == 'R25':
    stashprec='a05216'
  elif model == 'CP4_regridded':
    stashprec='a04203'
  elif model == 'CP4':
    stashprec='a04203'
  if stash == stashprec: series=series*3600. # convert precip to kg m-2 hour-1
  dataout=np.array([time1h,series])
  txt_path = os.path.join(result_path,'data_'+model+'_'+fc+'_'+stash+'.txt')
  fout = open(txt_path, 'w')
  np.savetxt(fout,np.transpose(dataout),fmt="%.1f,%.3f")
  #fout.write(dataout)
  fout.close()

def create_result_files(root_path):
  """
  :param root_path: time file
  :return: create predition files,true file and input file
  """
  pred_file = os.path.join(root_path,'imgs_pred')
  if not os.path.exists(pred_file):
    os.mkdir(pred_file)
  true_file = os.path.join(root_path,'imgs_true')
  if not os.path.exists(true_file):
    os.mkdir(true_file)
  input_file = os.path.join(root_path, 'imgs_input')
  if not os.path.exists(input_file):
    os.mkdir(input_file)

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

  # stash = 'a04203'
  # data_path = './200608290030-200608292330/pred.nc'
  # result_path = './200608290030-200608292330/images_pred'
  # plot_images(stash, data_path,result_path)

  root_path = './result'
  file_names = get_file(root_path)
  print(file_names )
  for file_name in file_names:
    file_path = os.path.join(root_path, file_name)
    create_result_files(file_path)
    # 1.plot pred imgs
    pred_result_path = os.path.join(file_path,'imgs_pred')
    pred_data_path = os.path.join(file_path, 'pred.nc')
    plot_images('a04203', pred_data_path,pred_result_path)
    # 2.plot input imgs
    input_root_path = './a01208'
    input_data_path = os.path.join(input_root_path,'a01208_A1hr_mean_aj575_4-25km_'+file_name+'.nc')
    input_result_path = os.path.join(file_path,'imgs_input')
    plot_images('a01208', input_data_path, input_result_path)
    # 3.2.plot true a04203 imgs
    true_root_path = './a04203'
    true_data_path = os.path.join(true_root_path, 'a04203_A1hr_mean_aj575_25-4km_' + file_name + '.nc')
    true_result_path = os.path.join(file_path,'imgs_true')
    plot_images('a04203', true_data_path, true_result_path)




