"""
@author: WYP
@file  : pred_lstm.py
@time  : 2022/8/15 10:05 上午
@desc  : step 3: pred new data
"""
import netCDF4
from netCDF4 import Dataset
import datetime
import pandas as pd
import numpy as np
import os
import shutil
from keras.models import load_model
pd.set_option('display.float_format', lambda x: '%.15f' % x)


def nc_to_csv(nc_path,data_type):
    """
    :param nc_path: path of nc file
    :param data_type: 'a01207','a03332','a04203','a01208'
    :return: csv file
    """
    nc = Dataset(nc_path)
    nc_shape = nc.variables[data_type][:].data.shape
    #print(nc_shape)
    shape_proc = int(nc_shape[0] * nc_shape[1] * nc_shape[2])

    # 1.create empty dataframe
    data = pd.DataFrame(index=range(0, shape_proc),
                        columns=['time', 'latitude', 'longitude', data_type,data_type+'mask'], dtype='object')
    # 2.fill time column
    time = netCDF4.num2date(nc.variables['time'][:], 'hours since 1970-01-01 00:00:00').data
    time = [str(time[i]) for i in range(24)]
    data['time'] = list(np.repeat(time, 366*236))

    # 3.fill latitude column
    latitude = list(nc.variables['latitude'][:].data)
    data['latitude'] = list(np.repeat(latitude, 236))*24

    # 4.fill longitude column
    longitude = list(nc.variables['longitude'][:].data)
    data['longitude'] = longitude*366*24

    # 5.fill datatype column
    data_series = list(nc.variables[data_type][:].data.reshape(shape_proc))
    data.loc[:, data_type] = pd.Series(data_series)

    # 6. fill mask column
    mask = list(nc.variables[data_type][:].mask.reshape(shape_proc))
    data.loc[:, data_type+'mask'] = pd.Series(mask)
    return data

def all_ncs_to_csvs(data_path,nc_number):
    """
    :param data_path: './data
    :param nc_number: the number of nc files used to be trained
    :return: a nc file corresponding to a csv file,save in a01207(or a03332、a04203、a01208)
    """
    data_types = ['a01207', 'a01208','a03332', 'a04203']
    for data_type in data_types:
        file_path = os.path.join(data_path, data_type)
        nc_format = ['.nc']
        nc_lists = [name for name in os.listdir(file_path) for item in nc_format if os.path.splitext(name)[1] == item]
        print(sorted(nc_lists))
        for nc_name in sorted(nc_lists)[0:nc_number]:  # 只用到了两个nc文件
            nc_path = os.path.join(file_path, nc_name)
            data = nc_to_csv(nc_path, data_type)
            data.to_csv(nc_path.replace('.nc', '.csv'), index=None)


def move_csv(result_path):
    """
    :param result_path: path of result file
    :param data_types: ['a01207','a03332','a04203','a01208']
    :return: move the same time csv files to the same file
    """
    data_types = ['a01207', 'a03332', 'a04203', 'a01208']
    data_format = ['.csv']
    for data_type in data_types[0:1]:
        file_path = os.path.join(data_path, data_type)
        csv_lists = [name for name in os.listdir(file_path) for item in data_format if os.path.splitext(name)[1] == item]
        for csv in csv_lists:
            time_name = csv.split('_')[5][0:-4]
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            if not os.path.exists(result_path + str(time_name) + '/'):
                os.mkdir(result_path + str(time_name) + '/')
    data_format = ['.csv']
    for data_type in data_types:
        file_path = os.path.join(data_path, data_type)
        csv_lists = [name for name in os.listdir(file_path) for item in data_format if os.path.splitext(name)[1] == item]
        for csv_name in csv_lists:
            time_name = csv_name.split('_')[5][0:-4]
            tar_path = os.path.join(result_path, time_name)
            try:
                shutil.move(file_path + '/' + csv_name,tar_path + '/' + csv_name)
            except:
                pass

def merge_data(merge_path):
    """
    :param data_path: './pred/result'
    :return: merged dataframe
    """
    data_types = ['a01207', 'a01208','a03332', 'a04203']
    df1 = pd.read_csv(os.path.join(merge_path,data_types[0]+'.csv'))
    df2 = pd.read_csv(os.path.join(merge_path,data_types[1]+'.csv'))
    df3 = pd.read_csv(os.path.join(merge_path,data_types[2]+'.csv'))
    df4 = pd.read_csv(os.path.join(merge_path,data_types[3]+'.csv'))
    data = df1
    data['a01208'] = df2['a01208']
    data['a01208mask'] = df2['a01208mask']
    data['a03332'] = df3['a03332']
    data['a03332mask'] = df3['a03332mask']
    data['a04203'] = df4['a04203']
    data['a04203mak'] = df4['a04203mask']
    data.to_csv(os.path.join(merge_path,'data.csv'), index=None)
    return data

def get_file(path):
    """
    :param path: path
    :return: sub file list
    """
    file = []
    for root, dirs, files in os.walk(path):
        file.append(dirs)
    return file[0]

def merge_csvs(result_path):
    """
    :param result_path: the file to save the same day csvs
    :return: data.csv
    """
    file_names = get_file(result_path)
    nc_middle = '_A1hr_mean_aj575_4-25km_'
    a04203_middle = '_A1hr_mean_aj575_25-4km_'
    for file_name in file_names:
        path = os.path.join(result_path,file_name)
        data_format = ['.csv']
        csv_lists = [name for name in os.listdir(path) for item in data_format if os.path.splitext(name)[1] == item]
        if len(csv_lists)!=4:
            print('check the data!')
        data_types = ['a01207', 'a01208','a03332', 'a04203']
        df1 = pd.read_csv(os.path.join(path,data_types[0]+ nc_middle + file_name + '.csv'))
        df2 = pd.read_csv(os.path.join(path,data_types[1]+ nc_middle + file_name + '.csv'))
        df3 = pd.read_csv(os.path.join(path,data_types[2]+ nc_middle + file_name + '.csv'))
        df4 = pd.read_csv(os.path.join(path,data_types[3]+ a04203_middle + file_name + '.csv'))
        data = df1
        print(len(data))
        data['a01208'] = df2['a01208']
        data['a01208mask'] = df2['a01208mask']
        data['a03332'] = df3['a03332']
        data['a03332mask'] = df3['a03332mask']
        data['a04203'] = df4['a04203']
        data['a04203mak'] = df4['a04203mask']
        data.to_csv(os.path.join(path,'data.csv'), index=None)

def df_process(data):
    """
    :param data: dataframe after merge_data(data_path) step
    :return: processed dataframe
    """
    # print('data shape:',data.shape)
    # delete the rows which mask column is True
    df = data.loc[(data['a01207mask'] == False) & (data['a01208mask'] == False) & (data['a03332mask'] == False) & (
                data['a04203mak'] == False), :]
    #print('df shape:', df.shape)
    # delete the rows which a01207 colunmn is 0
    df_1 = df.loc[(df['a01207']!=0),:]
    print('data shape:', df_1.shape)
    # create the column 'ratio' which means a01208 / a01027
    df_1['ratio'] = df_1['a01208'] / df_1['a01207']
    # delete unused columns
    df_2 = df_1[['time','latitude','longitude','ratio','a01207','a01208','a03332','a04203']]
    return df_2

def add_time_features(data):
    """
    :param data: dataframe after df_process step
    :return: dataframe after add_time_features
    """
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    # when train data include different year or different month
    # You can use the following four lines of commented code
    #data['month'] = data['time'].dt.month
    #data['day_of_year'] = data['time'].dt.dayofyear
    #data['week_of_year'] = data['time'].dt.weekofyear
    #df = data[['time', 'latitude', 'longitude', 'ratio','a03332','a04203','hour','month', 'day_of_year', 'week_of_year']]
    df = data[['time','latitude','longitude','ratio','a01207','a01208','a03332','a04203','hour']]
    return df

def max_min_scaler(s):
    """
    :param s: dataframe
    :return: Determine whether the maximum and minimum values are equal, if they are equal, return 0, if they are not equal, return the normalized result
    """
    if float(s.min()) == float(s.max()):
        return s-s.min()
    else:
        return (s - s.min()) / (s.max() - s.min())
def Normalize_data(data):
    """
    :param data: dataframe
    :return: normalize data
    """
    #data = data.set_index('date')
    data[['latitude']] = max_min_scaler(data[['latitude']])
    data[['longitude']] = max_min_scaler(data[['longitude']])
    data[['ratio']] = max_min_scaler(data[['ratio']])
    data[['a01207']] = max_min_scaler(data[['a01207']])
    data[['a01208']] = max_min_scaler(data[['a01208']])
    data[['a03332']] = max_min_scaler(data[['a03332']])
    data[['hour']] = max_min_scaler(data[['hour']])
    return data
def get_slide_window(samples,slide_windows):
    """
    :param data: dataframe
    :param slide_windows: num of slide_windows
    :return: dataset
    """
    x0 = samples.values
    n = int(len(x0))
    x = np.array([x0[k:k+slide_windows] for k in range(n-slide_windows+1)])
    return x

if __name__== '__main__':
    slide_windows = 10   # this para need to be same as train step
    nc_number = 1
    data_path = './pred'
    result_path = './pred/result/'
    all_ncs_to_csvs(data_path, nc_number)
    move_csv(result_path)
    merge_csvs(result_path)
    file_names = get_file(result_path)
    print(file_names)
    for file_name in file_names:
        df_path = os.path.join(result_path, file_name + '/data.csv')
        data = pd.read_csv(df_path)
        data = df_process(data)
        data = add_time_features(data)
        data = Normalize_data(data)
        df = pd.DataFrame(columns=['a04203', 'pred'], dtype='object')
        df['a04203'] = data['a04203']
        df = df.iloc[slide_windows - 1:, :]
        del data['a04203']
        del data['time']
        data = get_slide_window(samples=data, slide_windows=slide_windows)
        model = load_model('./pred/best.pkl')
        preds = model.predict(x=data, batch_size=64, verbose=0)
        print(preds)
        df['pred'] = preds/10000
        print(df['pred'])
        df.to_csv(os.path.join(result_path, file_name + '/pred.csv'))