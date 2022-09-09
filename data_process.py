"""
@author:
@file  : data_process.py
@time  : 2022/8/12 14:53
@desc  : step 1: data process
"""

import netCDF4
from netCDF4 import Dataset
import datetime
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
    data_types = ['a01207', 'a03332', 'a04203', 'a01208']
    for data_type in data_types:
        file_path = os.path.join(data_path, data_type)
        nc_format = ['.nc']
        nc_lists = [name for name in os.listdir(file_path) for item in nc_format if os.path.splitext(name)[1] == item]
        print(sorted(nc_lists))
        for nc_name in sorted(nc_lists)[0:nc_number]:  # 只用到了两个nc文件
            nc_path = os.path.join(file_path, nc_name)
            data = nc_to_csv(nc_path, data_type)
            data.to_csv(nc_path.replace('.nc', '.csv'), index=None)


def merge_csvs(data_path):
    """
    :param data_path: './data'
    :return: merged csv files which saved in ./data
    """
    data_types = ['a01207', 'a03332', 'a04203', 'a01208']
    for data_type in data_types:
        file_path = os.path.join(data_path, data_type)
        csv_format = ['.csv']
        csv_lists = [name for name in os.listdir(file_path) for item in csv_format if os.path.splitext(name)[1] == item]
        #print(sorted(csv_lists))
        data = pd.read_csv(os.path.join(file_path, csv_lists[0]))
        for csv_name in csv_lists[1:2]:
            df = pd.read_csv(os.path.join(file_path, csv_name))
            data = data.append(df, ignore_index=True)
        data.to_csv(os.path.join(data_path, data_type + '.csv'), index=None)

def merge_data(data_path):
    """
    :param data_path: './data'
    :return: merged dataframe
    """
    data_types = ['a01207', 'a01208','a03332', 'a04203']
    df1 = pd.read_csv(os.path.join(data_path,data_types[0]+'.csv'))
    df2 = pd.read_csv(os.path.join(data_path,data_types[1]+'.csv'))
    df3 = pd.read_csv(os.path.join(data_path,data_types[2]+'.csv'))
    df4 = pd.read_csv(os.path.join(data_path,data_types[3]+'.csv'))
    data = df1
    data['a01208'] = df2['a01208']
    data['a01208mask'] = df2['a01208mask']
    data['a03332'] = df3['a03332']
    data['a03332mask'] = df3['a03332mask']
    data['a04203'] = df4['a04203']
    data['a04203mak'] = df4['a04203mask']
    data.to_csv(os.path.join(data_path,'data.csv'), index=None)
    return data

def df_process(data):
    """
    :param data: dataframe after merge_data(data_path) step
    :return: processed dataframe
    """
    print('data shape:',data.shape)
    # delete the rows which mask column is True
    df = data.loc[(data['a01207mask'] == False) & (data['a01208mask'] == False) & (data['a03332mask'] == False) & (
                data['a04203mak'] == False), :]
    print('df shape:', df.shape)
    # delete the rows which a01207 colunmn is 0
    df_1 = df.loc[(df['a01207']!=0),:]
    print('df_1 shape:', df_1.shape)
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

def get_slide_window(samples,targets,slide_windows):
    """
    :param data: dataframe
    :param slide_windows: num of slide_windows
    :return: dataset
    """
    x0 = samples.values
    n = int(len(x0))
    x = np.array([x0[k:k+slide_windows] for k in range(n-slide_windows+1)])
    y = np.array(targets[slide_windows-1:])
    return x,y



def dataset_divide(samples,targets,test_size,validation_size):
    """
    :param samples: samples
    :param targets: label
    :param test_size: test dataset ratio
    :param validation_size: validation dataset ratio
    :return: train dataset、validation dataset、test dataset
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(samples, targets, test_size=test_size,
                                                                random_state=0, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=validation_size, random_state=0,
                                                     shuffle=True)
    return X_train,X_val, X_test,y_train,y_test,y_val


def  write_config(Folder, test_size, validation, lr, epochs, batch_size, samples_len, plot_num, nc_number, sequence_len):
    """
    :param Folder: save path
    :return: txt file
    """
    config_file = open(os.path.join(Folder, 'params_config.txt'), 'w')
    config_file.write('test_size:' + str(test_size) + '\n')
    config_file.write('validation:' + str(validation) + '\n')
    config_file.write('lr:' + str(lr) + '\n')
    config_file.write('epochs:' + str(epochs) + '\n')
    config_file.write('batch_size:' + str(batch_size) + '\n')
    config_file.write('samples_len:' + str(samples_len) + '\n')
    config_file.write('plot numbers:' + str(plot_num) + '\n')
    config_file.write('nc file numbers:' + str(nc_number) + '\n')
    config_file.write('sequence length:' + str(sequence_len) + '\n')
    config_file.close()

def plot_loss(Folder,history):
    loss = pd.DataFrame(history.history['loss'])
    loss.to_csv(os.path.join(Folder, 'Loss.csv'))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.legend()
    fig_name = os.path.join(Folder, 'Loss.png')
    plt.savefig(fig_name, dpi=600)
    plt.close()


def write_result(Folder,mean_squared_error,mean_absolute_error,r2_score):
    """
    :param Folder: file path
    :param mean_squared_error: mean_squared_error
    :param mean_absolute_error: mean_absolute_error
    :param r2_score: r2_score
    :return: result txt file
    """
    result_file = open(os.path.join(Folder, 'result.txt'), 'w')
    result_file.write('mean_squared_error:' + str(mean_squared_error) + '\n')
    result_file.write('mean_absolute_error:' + str(mean_absolute_error) + '\n')
    result_file.write('r2_score:' + str(r2_score) + '\n')
    result_file.close()

def result_plot(Folder,y_pred,y_test,plot_num,type):
    plt.plot(y_pred[0:plot_num], 'r', label='prediction')
    plt.plot(y_test[0:plot_num], 'b', label='real')
    plt.xlabel(type + ' sample number')
    plt.ylabel('Stratiform rainfall rate(*e-4)')
    plt.title(type)
    plt.legend(loc='best')
    plt.savefig(os.path.join(Folder, '{}.png'.format(type)), format='png', dpi=200)
    plt.show()

if __name__== '__main__':
    # 1.nc data to csv
    data_path = './data'
    # all_ncs_to_csvs(data_path)

    # 2.merge csv files,save in ./data
    # merge_csvs(data_path)

    # 3.merge data to a DataFrame
    data = merge_data(data_path)

    # 4.data process
    data = df_process(data)

    # 5.add time features
    data = add_time_features(data)




















