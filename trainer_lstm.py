"""
@author:
@file  : trainer_lstm.py
@time  : 2022/8/12 14:54
@desc  : step 2: train model
"""
#30 60 90 120

from data_process import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# When training with gpu, do not comment the following code
#import keras.backend.tensorflow_backend as K
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#K.set_session(sess)

# hyper-parameter
test_size =  0.3      # tarin dataset and test dataset ratio
validation = 0.4      # validation ratio
lr = 0.001            # learning rate
epochs = 100           # epoch numbers
batch_size = 64       # deep learing batch_size
plot_num = 10000      # the number of points in result plot
nc_number = 2       # the number of nc files used to train
slide_windows = 10    # parameter in data process
sequence_len = 10     # parameter in lstm


# 1.data preprocess
# (1)nc data to csv
data_path = './data'
all_ncs_to_csvs(data_path,nc_number)
# (2)merge csv files,save in ./data
data_path = './data'
print('merge_csvs:')
merge_csvs(data_path)
# (3)merge data to a DataFrame
print('merge data to a DataFrame:')
data = merge_data(data_path)

data_path = './data'
#data = pd.read_csv('/Users/sun/Desktop/Code1/Code/model/data/data.csv')
# (4)data process
print('data process:')
data = df_process(data)
# (5)add time features
print('add time features:')
data = add_time_features(data)
# (6)normalize data
print('normalize data:')
data = Normalize_data(data)
data.to_csv(os.path.join(data_path,'data_process.csv'), index=None)



# 2.train、test、validation test dataset divide
#data = pd.read_csv('/Users/sun/Desktop/Code1/Code/model/data/data_process.csv')


data['label'] = data['a04203']
del data['a04203']
del data['time']

# Use the following three lines of code to test whether the model works，only 600 samples for training
samples = data.iloc[0:600,:-1]
targets = data['label']*10000
targets = targets.head(600)

# Use the following code to run all the samples
#samples = data.iloc[:,:-1]
#targets = data['label']*10000


X,Y = get_slide_window(samples,targets,slide_windows)
print(X.shape)
print(Y.shape)

X_train,X_val, X_test,y_train,y_test,y_val = dataset_divide(X,Y,test_size,validation)
# X_train = np.array(X_train).reshape(X_train.shape[0],1,5)
# X_test = np.array(X_test).reshape(X_test.shape[0],1,5)
# X_val = np.array(X_val).reshape(X_val.shape[0],1,5)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# 3.model structure
model = Sequential()
model.add(LSTM(input_shape=(sequence_len,7),units=256,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=128,return_sequences=True))
model.add(LSTM(units=64,return_sequences=True))
model.add(LSTM(units=8,return_sequences=False))
model.add(Dense(units=1, activation='linear'))
print(model.summary())
adam = tf.keras.optimizers.Adam(lr=lr)
model.compile(loss='mse', optimizer='adam')


# 4.model saving: the path is the file named timestamp
current = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#current = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y_%m_%d-%H_%M_%S") ) #linux time
Folder = os.path.join('saved_model', current)
if not os.path.exists(Folder):
    os.makedirs(Folder)

# save config setting
write_config(Folder,test_size,validation,lr,epochs,batch_size,len(samples),plot_num,nc_number,sequence_len)


# save the best model
checkpoint1 = ModelCheckpoint(os.path.join(Folder,'best.pkl'),monitor='val_loss',save_best_only=True,save_weights_only=False, mode='min', period=1,verbose=1)
# save all models
model_path = os.path.join(Folder,'model-{epoch:02d}-{loss:.10f}.pkl')
checkpoint2 = ModelCheckpoint(model_path, monitor='val_loss',save_best_only=False,save_weights_only=False,mode='min', verbose=0,period=1)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint1,checkpoint2], validation_data=(X_val,y_val))

#  save loss and plot
plot_loss(Folder,history)


# 5.model test
model = load_model(os.path.join(Folder, 'best.pkl'))
y_pred = model.predict(X_test)
print('mean_squared_error:',mean_squared_error(y_test,y_pred))
print('mean_absolute_error:',mean_absolute_error(y_test,y_pred))
print('r2_score:',r2_score(y_test,y_pred))

# save regress result
write_result(Folder,mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_absolute_error(y_test,y_pred))

# result plot
y_pred = [x for y in y_pred for x in y]
result_plot(Folder,y_pred,y_test,plot_num,'test')
result_plot(Folder,model.predict(X_train),y_train,12000,'train')