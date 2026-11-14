# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:30:36 2022

@author: Alireza Mostafavi and Youngjin Cha
"""

from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

scaler = MinMaxScaler(feature_range=(-1, 1))
samplerate =48000
#%%
def  data_concat(path):
    file_names = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for i in range(0,len(file_names)):
        samplerate, noise = wavfile.read(path+file_names[i])
        if samplerate !=48000:
            print('Sample rate is not equal to 48p kHz')
            return 0
        if noise.ndim>1:
            noise =scaler.fit_transform( noise[:,0].reshape((noise.shape[0],1)))
        else:
            noise =scaler.fit_transform( noise.reshape((len(noise),1)))  
        
        data.append(noise)
        
    data = np.concatenate( data, axis=0 ,dtype=np.float32)
    return data

path = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/noise dataset/48kHz/'
bulldozer = data_concat(path+'bulldozer/')
data_review = pd.DataFrame(len(bulldozer)/samplerate,index = ['Bulldozer'],columns=['Time [s]'])

ambience = data_concat(path+'ambience/')
data_review=data_review.append(pd.DataFrame(len(ambience)/samplerate,index = ['Ambience'],columns=['Time [s]']))

cementMixer = data_concat(path+'cementMixer/')
data_review=data_review.append(pd.DataFrame(len(cementMixer)/samplerate,index = ['cementMixer'],columns=['Time [s]']))

electricDrill = data_concat(path+'electricDrill/')
data_review=data_review.append(pd.DataFrame(len(electricDrill)/samplerate,index = ['electricDrill'],columns=['Time [s]']))

electricSander = data_concat(path+'electricSander/')
data_review=data_review.append(pd.DataFrame(len(electricSander)/samplerate,index = ['electricSander'],columns=['Time [s]']))


electricSaw = data_concat(path+'electricSaw/')
data_review=data_review.append(pd.DataFrame(len(electricSaw)/samplerate,index = ['electricSaw'],columns=['Time [s]']))

electricScrewDriver = data_concat(path+'electricScrewDriver/')
data_review=data_review.append(pd.DataFrame(len(electricScrewDriver)/samplerate,index = ['electricScrewDriver'],columns=['Time [s]']))

excavator = data_concat(path+'excavator/')
data_review=data_review.append(pd.DataFrame(len(excavator)/samplerate,index = ['excavator'],columns=['Time [s]']))

filingWood = data_concat(path+'filingWood/')
data_review=data_review.append(pd.DataFrame(len(filingWood)/samplerate,index = ['filingWood'],columns=['Time [s]']))

grader = data_concat(path+'grader/')
data_review=data_review.append(pd.DataFrame(len(grader)/samplerate,index = ['grader'],columns=['Time [s]']))

grinder = data_concat(path+'grinder/')
data_review=data_review.append(pd.DataFrame(len(grinder)/samplerate,index = ['grinder'],columns=['Time [s]']))


hammering = data_concat(path+'hammering/')
data_review=data_review.append(pd.DataFrame(len(hammering)/samplerate,index = ['hammering'],columns=['Time [s]']))

handSaw = data_concat(path+'handSaw/')
data_review=data_review.append(pd.DataFrame(len(handSaw)/samplerate,index = ['handSaw'],columns=['Time [s]']))

jackhammer = data_concat(path+'jackhammer/')
data_review=data_review.append(pd.DataFrame(len(jackhammer)/samplerate,index = ['jackhammer'],columns=['Time [s]']))

sanding = data_concat(path+'sanding/')
data_review=data_review.append(pd.DataFrame(len(sanding)/samplerate,index = ['sanding'],columns=['Time [s]']))

#path = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/noise dataset/96kHz/screwDriver/'
#screwDriver = data_concat(path)
#data_review=data_review.append(pd.DataFrame(len(screwDriver)/samplerate,index = ['screwDriver'],columns=['Time [s]']))

truck = data_concat(path+'truck/')
data_review=data_review.append(pd.DataFrame(len(truck)/samplerate,index = ['truck'],columns=['Time [s]']))

welding = data_concat(path+'welding/')
data_review=data_review.append(pd.DataFrame(len(welding)/samplerate,index = ['welding'],columns=['Time [s]']))
print(data_review)
#%%
train_data = []
test_data = []
def train_test_split(data,fraction,n_train,n_test):
    train = data[0:math.floor(len(data)*fraction)]
    test = data[math.floor(len(data)*fraction):]
    
    a=np.random.randint(0,len(train)-n_train*samplerate,1)
    b=np.random.randint(0,len(test)-n_test*samplerate,1)
    train = train[a[0]:a[0]+n_train*samplerate]
    test = test[b[0]:b[0]+n_test*samplerate]
    return train,test

train,test = train_test_split(bulldozer, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(ambience, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(cementMixer, 0.7, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(electricDrill, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(electricSander, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(electricSaw, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(electricScrewDriver, 0.7, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(excavator, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(filingWood, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(grader, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(grinder, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(hammering, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(handSaw, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(jackhammer, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(sanding, 0.7, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(truck, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train,test = train_test_split(welding, 0.6, 60, 20)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))

train_data = np.concatenate( train_data, axis=0 ,dtype=np.float32)
test_data = np.concatenate( test_data, axis=0 ,dtype=np.float32)

print('training data length in second = ',len(train_data)/samplerate)
print('test data length in second = ',len(test_data)/samplerate)

#test_data = scaler.fit_transform( test_data)  
#train_data = scaler.fit_transform( train_data)  


plt.figure(figsize=(12,8))
plt.plot(test_data, label= 'X test')
plt.legend(loc="upper right",fontsize='large')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.figure(figsize=(12,8))
plt.plot(train_data, label= 'X train')
plt.legend(loc="upper right",fontsize='large')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


#%%
path = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/noise dataset/48kHz/'
wavfile.write(path+'train_noise.wav', samplerate, train_data)
#np.savetxt(path+'train_noise.csv', train_data, delimiter=",")

wavfile.write(path+'test_noise.wav', samplerate, test_data)
#np.savetxt(path+'test_noise.csv', test_data, delimiter=",")
wavfile.write(path+'validation_noise.wav', samplerate, test_data)

#%% for DNoiseNet

train_data = []
test_data = []
def train_test_split(data,fraction,n_train,n_test):
    train = data[0:math.floor(len(data)*fraction)]
    test = data[math.floor(len(data)*fraction):]
    
    a=np.random.randint(0,len(train)-n_train*48000,1)
    b=np.random.randint(0,len(test)-n_test*48000,1)
    train = train[a[0]:a[0]+n_train*48000]
    test = test[b[0]:b[0]+n_test*48000]
    return train,test

train,test = train_test_split(bulldozer, 0.6, 10, 5)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(grinder, 0.6, 10, 5)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train,test = train_test_split(hammering, 0.6, 10, 5)
train_data.append(train-np.mean(train))
test_data.append(test-np.mean(test))


train_data = np.concatenate( train_data, axis=0 ,dtype=np.float32)
test_data = np.concatenate( test_data, axis=0 ,dtype=np.float32)

print('training data length in second = ',len(train_data)/samplerate)
print('test data length in second = ',len(test_data)/samplerate)

plt.figure(figsize=(12,8))
plt.plot(test_data, label= 'X test')
plt.legend(loc="upper right",fontsize='large')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.figure(figsize=(12,8))
plt.plot(train_data, label= 'X train')
plt.legend(loc="upper right",fontsize='large')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

#%%

path = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/GPU_2022_6_6/Previous paper/Experiments/Data/48kHz/'
wavfile.write(path+'train_BGH.wav', 48000,train_data)
#np.savetxt(path+'test_noise.csv', test_data, delimiter=",")
#%%

wavfile.write(path+'test_BGH.wav', 48000, test_data)
#np.savetxt(path+'train_noise.csv', train_data, delimiter=",")


