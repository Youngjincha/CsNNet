# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:29:43 2023

@author: Alireza Mostafavi and Youngjin Cha
"""

#%% Import libraries
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary as summary2
torch.cuda.is_available()
device2 = torch.device('cuda:2')
device = torch.device('cuda:0')
print(device)
#CUDA_LAUNCH_BLOCKING=1
torch.cuda.empty_cache()
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import rir_generator as rir
from scipy import signal
#%% load the data 
path =  'C:/Users/Alireza/OneDrive - University of Manitoba/Desktop/noise dataset/16kHz/'
scaler = MinMaxScaler(feature_range=(-1, 1))

samplerate, train_data = wavfile.read(path+'train_noise.wav')
samplerate, test_data = wavfile.read(path+'test_noise.wav')

train_data =scaler.fit_transform( train_data.reshape((len(train_data),1)))
test_data =scaler.fit_transform( test_data.reshape((len(test_data),1)))

train_data = train_data-np.mean(train_data)
test_data = test_data - np.mean(test_data)

train_data = train_data.reshape((len(train_data),1))
test_data = test_data.reshape((len(test_data),1))

plt.plot(test_data)

#%% Model measurement noise with the desired SNR

sc  = 90000 # change this number in order to reach your desire SNR
white_noise_train1 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))
white_noise_train2 = np.random.normal(loc = 0, scale = 1, size = (len(train_data),1))

white_noise_train1 = white_noise_train1/sc
white_noise_train2 = white_noise_train2/sc
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train1**2)))
print('SNR train data',10*math.log10(np.sum(train_data**2)/np.sum(white_noise_train2**2)))

#plt.plot(white_noise_train1)

#for test data
white_noise_test1 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test1 = white_noise_test1/sc
white_noise_test2 = np.random.normal(loc = 0, scale = 1, size = (len(test_data),1))
white_noise_test2 = white_noise_test2/sc

print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test1**2)))
print('SNR test data',10*math.log10( np.sum(test_data**2)/np.sum(white_noise_test2**2)))

#plt.plot(white_noise_test1)



#%% Generate room impulse response 

fs = 16000
n_sample = 512

def RIR(fs,rb,n_sample):
    h_s = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [1.5, 3, 1]
    ],
    s=[1.5, 2.5, 1],          # Source position [x y z] (m)
    dim = 3,
    L=[3, 4, 2],            # Room dimensions [x y z] (m)
    reverberation_time=rb, # Reverberation time (s)
    nsample=n_sample,           # Number of output samples
    mtype= rir.mtype.omnidirectional, # Type of microphone
    order = 3,
    hp_filter = False
    )


    h_p = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [1.5, 3, 1]
    ],
    s=[1.5, 1, 1],          # Source position [x y z] (m)
    dim = 3,
    L=[3, 4, 2],            # Room dimensions [x y z] (m)
    reverberation_time=rb, # Reverberation time (s)
    nsample=n_sample,           # Number of output samples
    mtype= rir.mtype.omnidirectional, # Type of microphone
    order = 3,
    hp_filter = False
    )

    h_p = h_p/ np.linalg.norm(h_p)
    h_s = h_s/ np.linalg.norm(h_s)
    return h_p,h_s

#%% SIngle PW RIR
h_p,h_s = RIR(fs,0.2,n_sample)

h_s_tns = torch.from_numpy(h_s.astype(np.float32)).to(device)
h_p_tns = torch.from_numpy(h_p.astype(np.float32)).view(1,1,-1)

plt.figure(figsize=(12,8))
plt.plot(h_p, label='h_p')
plt.plot(h_s, label='h_s')
plt.legend(loc="upper right",fontsize='large')
plt.title('Room Impulse Response, reverberation_time=0.2')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


#%% creating a function to generate suitable input and target data for network training 


def input_output_generator_LSTM(z,len_window,len_hop):
    x_train = []
    y_train = []
    z_train = z[:math.floor(trainfrac*len(z))]
    for i in range(0,len(z_train)-len_window-predicting,len_hop):
        x_train.append(z[i:i+len_window])
        y_train.append(z[i+predicting:i+len_window+predicting])
    
    x_train = np.array(x_train,dtype=np.float32)
    x_train_lstm = np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
    
    y_train=np.array(y_train,dtype=np.float32)
    y_train_lstm = np.reshape(y_train,(y_train.shape[0],1,y_train.shape[1]))
    return x_train_lstm, y_train_lstm

trainfrac = 1
len_window = 512  #number of input samples to the network
len_hop = 128     #number of samples overlapping between two consecutive frames
predicting = 200  #Number of samples you want the network to predict for respecting causality constraint
LB = len_window



#%% Filtering the input signal with primary path to calculate target signal
# Moreover, generating the suitable input and output for the network
d_train = signal.lfilter(torch.from_numpy(h_p[:,0]), 1, torch.from_numpy(train_data[:,0])).reshape(len(train_data),1)
d_test = signal.lfilter(torch.from_numpy(h_p[:,0]), 1, torch.from_numpy(test_data[:,0])).reshape(len(test_data),1)


X_noise_train, _ = input_output_generator_LSTM(train_data[0:len(d_train)]+white_noise_train1[0:len(d_train)], len_window,len_hop)
X_noise_test, _ = input_output_generator_LSTM(test_data[0:len(d_test)]+white_noise_test1[0:len(d_test)], len_window,len_hop)

_,Y_noise_train = input_output_generator_LSTM(d_train+white_noise_train2[0:len(d_train)], len_window,len_hop)
_,Y_noise_test = input_output_generator_LSTM(d_test+white_noise_test2[0:len(d_test)], len_window,len_hop)


X_noise_train_tns = torch.from_numpy(X_noise_train.astype(np.float32))
Y_noise_train_tns = torch.from_numpy(Y_noise_train.astype(np.float32))
X_noise_test_tns = torch.from_numpy(X_noise_test.astype(np.float32))
Y_noise_test_tns = torch.from_numpy(Y_noise_test.astype(np.float32))


plt.figure(figsize=(12,8))
plt.plot(X_noise_test_tns[:,0,256].numpy(), label= 'X train')
plt.plot(Y_noise_test_tns[:,0,256].numpy(), label= 'Y train')
plt.legend(loc="upper right",fontsize='large')
plt.title('Colvolved X with h_p')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


#%% CsNNet Model
h_s_tns = torch.from_numpy(h_s.astype(np.float32)).view(1,1,-1)

class Attention(nn.Module):
    def __init__(self,n_channel):
        super(Attention,self).__init__()
        self.fc =  nn.Sequential(nn.Linear(n_channel, n_channel // 4, 1),
                               nn.ReLU(),
                               nn.Linear(n_channel // 4, n_channel, 1))
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        
    def forward(self,x):
       
        avg_pool = F.adaptive_avg_pool1d(x, 1)
        avg_pool = torch.flatten(avg_pool, 1)
        
        max_pool = F.adaptive_max_pool1d(x, 1)
        max_pool = torch.flatten(max_pool, 1)
        
        max_pool = self.fc(max_pool)
        avg_pool = self.fc(avg_pool)
        
        channel_attention = torch.sigmoid(avg_pool + max_pool)
        
        x2 = x*channel_attention.view(channel_attention.size()[0],channel_attention.size()[1],1)
    
        return x2
            
class FW_DANC(nn.Module):
    def __init__(self):
        super(FW_DANC,self).__init__()
        
        self.conv1 = nn.Conv1d(1, 8, kernel_size=16,dilation=1,stride=2,padding=7)
        self.bn1 = nn.BatchNorm1d(num_features=8)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv1d(1, 8, kernel_size=16,dilation=2,stride=2,padding=7+8)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv1d(1, 8, kernel_size=16,dilation=4,stride=2,padding=7*2+8*2)
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.prelu3 = nn.PReLU()
        
        self.cat1 = nn.Conv1d(3*8,3*8, kernel_size=16,dilation=1,stride=1,padding='same',groups = 3*8)
        self.cat1_1 = nn.Conv1d(3*8,8, kernel_size=1,dilation=1,stride=1,padding='same')
        self.prelu1_1 = nn.PReLU()
        
        ##
        self.conv4 = nn.Conv1d(8, 8, kernel_size=16,dilation=1,stride=2,padding=7,groups=8)
        self.conv4_1 = nn.Conv1d(8, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm1d(num_features=8)
        self.prelu4 = nn.PReLU()
        
        self.conv5 = nn.Conv1d(8, 8, kernel_size=16,dilation=2,stride=2,padding=7+8,groups=8)
        self.conv5_1 = nn.Conv1d(8, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn5 = nn.BatchNorm1d(num_features=8)
        self.prelu5 = nn.PReLU()
        
        self.conv6 = nn.Conv1d(8, 8, kernel_size=16,dilation=4,stride=2,padding=7*2+2*8,groups=8)
        self.conv6_1 = nn.Conv1d(8, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn6 = nn.BatchNorm1d(num_features=8)
        self.prelu6 = nn.PReLU()
        
        #self.cat2 = nn.Conv1d(3*16,16, kernel_size=16,dilation=1,stride=1,padding='same')
        self.cat2 = nn.Conv1d(3*16,3*16, kernel_size=16,dilation=1,stride=1,padding='same',groups = 3*16)
        self.cat2_1 = nn.Conv1d(3*16,16, kernel_size=1,dilation=1,stride=1,padding='same')
        self.prelu2_1 = nn.PReLU()
        ##
        self.conv7 = nn.Conv1d(16, 16, kernel_size=16,dilation=1,stride=2,padding=7,groups=16)
        self.conv7_1 = nn.Conv1d(16, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn7 = nn.BatchNorm1d(num_features=16)
        self.prelu7 = nn.PReLU()
        
        self.conv8 = nn.Conv1d(16, 16, kernel_size=16,dilation=2,stride=2,padding=7+8,groups=16)
        self.conv8_1 = nn.Conv1d(16, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn8 = nn.BatchNorm1d(num_features=16)
        self.prelu8 = nn.PReLU()
        
        self.conv9 = nn.Conv1d(16, 16, kernel_size=16,dilation=4,stride=2,padding=7*2+2*8,groups=16)
        self.conv9_1 = nn.Conv1d(16, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn9 = nn.BatchNorm1d(num_features=16)
        self.prelu9 = nn.PReLU()
        
        self.cat3 = nn.Conv1d(3*32,3*32, kernel_size=16,dilation=1,stride=1,padding='same',groups = 3*32)
        self.cat3_1 = nn.Conv1d(3*32,32, kernel_size=1,dilation=1,stride=1,padding='same')
        self.prelu3_1 = nn.PReLU()
        ##
        self.conv10 = nn.Conv1d(32, 32, kernel_size=16,dilation=1,stride=2,padding=7,groups=32)
        self.conv10_1 = nn.Conv1d(32, 64, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn10_1 = nn.BatchNorm1d(num_features=32)
        self.prelu10_1 = nn.PReLU()
        
        self.conv11 = nn.Conv1d(32, 32, kernel_size=16,dilation=2,stride=2,padding=7+8,groups=32)
        self.conv11_1 = nn.Conv1d(32, 64, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn11_1 = nn.BatchNorm1d(num_features=32)
        self.prelu11_1 = nn.PReLU()
        
        self.conv12 = nn.Conv1d(32, 32, kernel_size=16,dilation=4,stride=2,padding=7*2+2*8,groups=32)
        self.conv12_1 = nn.Conv1d(32, 64, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn12_1 = nn.BatchNorm1d(num_features=32)
        self.prelu12_1 = nn.PReLU()
        
        self.cat4 = nn.Conv1d(3*64,3*64, kernel_size=16,dilation=1,stride=1,padding='same',groups = 3*64)
        self.cat4_1 = nn.Conv1d(3*64,64, kernel_size=1,dilation=1,stride=1,padding='same')
        self.prelu4_1 = nn.PReLU()
        ##
        self.LSTM = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        
        ##
        self.attention1 = Attention(64)
        
        self.up1 = nn.ConvTranspose1d(64+64, 128, kernel_size=8,dilation=1,stride=2,padding=3,groups =128)
        self.up1_1 = nn.Conv1d(128, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn10 = nn.BatchNorm1d(num_features=128)
        self.prelu10 = nn.PReLU()
        
        self.up2 = nn.ConvTranspose1d(64+64, 128, kernel_size=8,dilation=2,stride=2,padding=7,groups =128,output_padding = 1)
        self.up2_1 = nn.Conv1d(128, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn11 = nn.BatchNorm1d(num_features=128)
        self.prelu11 = nn.PReLU()
        
        self.up3 = nn.ConvTranspose1d(64+64, 128, kernel_size=8,dilation=4,stride=2,padding=14,groups =128,output_padding = 1)
        self.up3_1 = nn.Conv1d(128, 32, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn12 = nn.BatchNorm1d(num_features=128)
        self.prelu12 = nn.PReLU()
        
        ##
        self.attention2 = Attention(32)
        self.up4 = nn.ConvTranspose1d(32+32, 64, kernel_size=8,dilation=1,stride=2,padding=3,groups =64)
        self.up4_1 = nn.Conv1d(64, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn13 = nn.BatchNorm1d(num_features=64)
        self.prelu13 = nn.PReLU()
        
        self.up5 = nn.ConvTranspose1d(32+32, 64, kernel_size=8,dilation=2,stride=2,padding=7,groups =64,output_padding = 1)
        self.up5_1 = nn.Conv1d(64, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn14 = nn.BatchNorm1d(num_features=64)
        self.prelu14 = nn.PReLU()
        
        self.up6 = nn.ConvTranspose1d(32+32, 64, kernel_size=8,dilation=4,stride=2,padding=14,groups =64,output_padding = 1)
        self.up6_1 = nn.Conv1d(64, 16, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn15 = nn.BatchNorm1d(num_features=64)
        self.prelu15 = nn.PReLU()
        
        ##
        self.attention3 = Attention(16)
        
        self.up7 = nn.ConvTranspose1d(16+16, 32, kernel_size=8,dilation=1,stride=2,padding=3,groups=32)
        self.up7_1 = nn.Conv1d(32, 8, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn16 = nn.BatchNorm1d(num_features=32)
        self.prelu16 = nn.PReLU()
        
        self.up8 = nn.ConvTranspose1d(16+16, 32, kernel_size=8,dilation=2,stride=2,padding=7,groups=32,output_padding = 1)
        self.up8_1 = nn.Conv1d(32, 8, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn17 = nn.BatchNorm1d(num_features=32)
        self.prelu17 = nn.PReLU()
        
        self.up9 = nn.ConvTranspose1d(16+16, 32, kernel_size=8,dilation=4,stride=2,padding=14,groups=32,output_padding = 1)
        self.up9_1 = nn.Conv1d(32, 8, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn18 = nn.BatchNorm1d(num_features=32)
        self.prelu18 = nn.PReLU()
        ##
        self.attention4 = Attention(8)
        
        self.up10 = nn.ConvTranspose1d(8+8, 16, kernel_size=8,dilation=1,stride=2,padding=3,groups=16)
        self.up10_1 = nn.Conv1d(16, 4, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn19 = nn.BatchNorm1d(num_features=16)
        self.prelu19 = nn.PReLU(4)
        
        self.up11 = nn.ConvTranspose1d(8+8, 16, kernel_size=8,dilation=2,stride=2,padding=7,groups=16,output_padding = 1)
        self.up11_1 = nn.Conv1d(16, 4, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn20 = nn.BatchNorm1d(num_features=16)
        self.prelu20 = nn.PReLU(4)
        
        self.up12 = nn.ConvTranspose1d(8+8, 16, kernel_size=8,dilation=4,stride=2,padding=14,groups=16,output_padding = 1)
        self.up12_1 = nn.Conv1d(16, 4, kernel_size=1,dilation=1,stride=1,padding=0)
        self.bn21 = nn.BatchNorm1d(num_features=16)
        self.prelu21 = nn.PReLU(4)
        
        ##
        self.final_conv = nn.Conv1d(4, 1, kernel_size=1,dilation=1,stride=1,padding='same')
        
        
    def forward(self,x):
        
        x1 = self.prelu1(self.bn1(self.conv1(x)))
        x2 = self.prelu2(self.bn2(self.conv2(x)))
        x3 = self.prelu3(self.bn3(self.conv3(x)))
        first_layer = self.prelu1_1(self.cat1_1(self.cat1(torch.cat([x1,x2,x3],1))))
        
        x4 = self.prelu4(self.conv4_1(self.bn4(self.conv4(first_layer))))
        x5 = self.prelu5(self.conv5_1(self.bn5(self.conv5(first_layer))))
        x6 = self.prelu6(self.conv6_1(self.bn6(self.conv6(first_layer))))
        second_layer = self.prelu2_1(self.cat2_1(self.cat2(torch.cat([x4,x5,x6],1))))
        
        x7 = self.prelu7(self.conv7_1(self.bn7(self.conv7(second_layer))))
        x8 = self.prelu8(self.conv8_1(self.bn8(self.conv8(second_layer))))
        x9 = self.prelu9(self.conv9_1(self.bn9(self.conv9(second_layer))))
        embedding_layer = self.prelu3_1(self.cat3_1(self.cat3(torch.cat([x7,x8,x9],1))))
        
        x10_1 = self.prelu10_1(self.conv10_1(self.bn10_1(self.conv10(embedding_layer))))
        x11_1 = self.prelu11_1(self.conv11_1(self.bn11_1(self.conv11(embedding_layer))))
        x12_1 = self.prelu12_1(self.conv12_1(self.bn12_1(self.conv12(embedding_layer))))
        embedding_layer2 = self.prelu4_1(self.cat4_1(self.cat4(torch.cat([x10_1,x11_1,x12_1],1))))
        
        lstm_input = embedding_layer2.permute(0,2,1)

        lstm_out, (hn, cn) = self.LSTM(lstm_input)
        lstm_out = lstm_out.permute(0,2,1)
        
        #x0 = lstm_out*embedding_layer2
        #x0 = embedding_layer2
        
        skip1 = self.attention1(embedding_layer2)
        
        x10 = self.prelu10(self.up1_1(self.bn10(self.up1(torch.cat([skip1,lstm_out],1)))))
        x11 = self.prelu11(self.up2_1(self.bn11(self.up2(torch.cat([skip1,lstm_out],1)))))
        x12 = self.prelu12(self.up3_1(self.bn12(self.up3(torch.cat([skip1,lstm_out],1)))))
        first_up_layer = x10+x11+x12
        
        skip2 = self.attention2(embedding_layer)
        
        x13 = self.prelu13(self.up4_1(self.bn13(self.up4(torch.cat([first_up_layer,skip2],1)))))
        x14 = self.prelu14(self.up5_1(self.bn14(self.up5(torch.cat([first_up_layer,skip2],1)))))
        x15 = self.prelu15(self.up6_1(self.bn15(self.up6(torch.cat([first_up_layer,skip2],1)))))
        second_up_layer = x13+x14+x15
        
        skip3 = self.attention3(second_layer)
        
        x16 = self.prelu16(self.up7_1(self.bn16(self.up7(torch.cat([second_up_layer,skip3],1)))))
        x17 = self.prelu17(self.up8_1(self.bn17(self.up8(torch.cat([second_up_layer,skip3],1)))))
        x18 = self.prelu18(self.up9_1(self.bn18(self.up9(torch.cat([second_up_layer,skip3],1)))))
        third_up_layer = x16+x17+x18
        
        skip4 = self.attention4(first_layer)
        
        x19 = self.prelu19(self.up10_1(self.bn19(self.up10(torch.cat([third_up_layer,skip4],1)))))
        x20 = self.prelu20(self.up11_1(self.bn20(self.up11(torch.cat([third_up_layer,skip4],1)))))
        x21 = self.prelu21(self.up12_1(self.bn21(self.up12(torch.cat([third_up_layer,skip4],1)))))
        final_layer = x19+x20+x21

        x19 = self.final_conv(final_layer)
        
        ## apply loudspeaker nonlinearity. uncomment of the F_05 or F_01 line and comment the other one
        # F_05 represent the nonlinearity of 0.5 and F_01 represent the nonlinearity of 0.1
        F_05 = 0.5*1.772453*torch.erf(x19)
        #F_01 = 0.22360679774*torch.sqrt(torch.tensor(torch.pi))*torch.erf(2.2360679774*x19)
        output2 = torch.nn.functional.conv1d(F_05,h_s_tns.to(x19.device),padding='same')
        
        return output2

mynetwork = FW_DANC()
batch_size = 100

print('Model Outputs')
print(summary2(mynetwork.cpu(), torch.rand((batch_size, 1,512)), show_input=False))

torch.cuda.empty_cache()
mynetwork = FW_DANC().to(device)

#%% Loss function, optimizer, number of epochs, and first learning rate

MSE = nn.MSELoss(reduction='mean').to(device) 
criterion = MSE
lr = 0.002
optimizer = torch.optim.Adam(mynetwork.parameters(), lr=lr, betas=(0.9, 0.999) , amsgrad=False)
num_epochs = 100

#%% Training the model
train = torch.utils.data.TensorDataset(X_noise_train_tns, Y_noise_train_tns)
test = torch.utils.data.TensorDataset(X_noise_test_tns, Y_noise_test_tns)

train_loader = torch.utils.data.DataLoader(train, batch_size=40, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=40, shuffle=True)

batch_size = 40
n_iterations = math.ceil(len(X_noise_train)/batch_size)
print(n_iterations)

validation_iteration = math.ceil(len(X_noise_test)/batch_size)

## path to save the model during trainig 
PATH = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/Saved_models/test21_2023_1_27_2.pth'

epoch_loss_saved = []
valid_loss_saved = []
since = time.time()
for epoch in range(num_epochs):
    train_loss = 0.0
    mynetwork.train()
    for step, (X, Y) in enumerate(train_loader):
    
        optimizer.zero_grad()
        outputs = mynetwork(X.to(device))
        loss = criterion(outputs, Y.to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #
        
        if (step+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{n_iterations}], Loss: {train_loss/step:.5f}')
            print('/n', outputs[0,0,0],Y[0,0,0])
        
    mynetwork.eval() 
    valid_loss = 0.0
    with torch.no_grad():
        for step, (X_val, Y_val) in enumerate(test_loader):

          # Forward Pass
          pred_val = mynetwork(X_val.to(device))
          # Find the Loss
          loss = criterion(pred_val,Y_val.to(device))
          # Calculate Loss
          valid_loss += loss.item()
          
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / n_iterations} \t\t Validation Loss: {valid_loss / validation_iteration}')
    epoch_loss_saved.append(train_loss / n_iterations)
    valid_loss_saved.append(valid_loss / validation_iteration)

    if valid_loss_saved[epoch] <= np.min(valid_loss_saved):
        torch.save(mynetwork, PATH)
        print('model saved')
    
    if (epoch+1)%3==0 and sum(valid_loss_saved[epoch-2:epoch+1] > np.min(valid_loss_saved))==3:
        lr = lr/2
        optimizer.param_groups[0]['lr'] = lr
        print("learning rate = ", optimizer.param_groups[0]['lr'])
 
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#%% After training run this to plot the training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(epoch_loss_saved, "-b", label="Training Loss")
plt.plot(valid_loss_saved, "-r", label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc="upper right",fontsize='large')
plt.show()

#%% If you want, you can save losses
np.savetxt('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/results/Matlab/training_loss_16_0.5_2.csv', epoch_loss_saved, delimiter=",")
np.savetxt('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/results/Matlab/validation_loss_16_0.5_2.csv', valid_loss_saved, delimiter=",")

#%% If you have saved model, you can load it here
PATH = 'C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/Saved_models/test21_2023_1_27_1.pth'

#If you want to save the final trained model, uncomment the line below
#torch.save(mynetwork, PATH)
mynetwork = torch.load(PATH)
#%% Test the model using test data
pred_test = torch.zeros(size=Y_noise_test_tns.shape)
valid_loss =0
mynetwork.eval() 
with torch.no_grad():
     #for step, (X_val, Y_val) in enumerate(test_loader):
     for i in range(validation_iteration-1):
       X_val = X_noise_test_tns[i*batch_size:(i+1)*(batch_size),:,:]
       Y_val = Y_noise_test_tns[i*batch_size:(i+1)*(batch_size),:]

       # Forward Pass
       pred_val = mynetwork(X_val.to(device))
       torch.cuda.synchronize()
       pred_test[i*batch_size:(i+1)*batch_size,:] = pred_val
       # Find the Loss
       loss = criterion(pred_val,Y_val.to(device))
       # Calculate Loss
       valid_loss += loss.item()

print('Test loss = ',valid_loss/validation_iteration)
pred_test_np = pred_test.cpu().numpy()

#%% convert the model data to one dimentional array
def rebuild_data(z,len_window,len_hop):
    x=[];
    x.append(z[0,0,:])
    for i in range(1,z.shape[0]):
        x.append(z[i,0,len_window-len_hop:len_window])
    x = np.concatenate( x, axis=0 ,dtype=np.float32)
    return x
    
def rebuild_dataE(z,len_window,len_hop):
    x=[];
    x.append(z[0,0,:])
    for i in range(1,z.shape[0]):
        x.append(z[i,0,len_window-len_hop:len_window]-np.mean(z[i,0,len_window-len_hop:len_window]))
    x = np.concatenate( x, axis=0 ,dtype=np.float32)
    return x
error = rebuild_dataE(Y_noise_test_tns.numpy()-pred_test_np,len_window,len_hop)
primary_noise = rebuild_data(Y_noise_test_tns.numpy(),len_window,len_hop)

print(len(test_data)-len(primary_noise))
#%% Plot the results

primary_noise = primary_noise - np.mean(primary_noise)
plt.figure(figsize=(12, 8))
plt.plot(primary_noise, label='Primary noise')
plt.plot(error,label = 'Attenuated noise')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc="upper right",fontsize='large')
plt.show()

noise_attenuation = 10*math.log10(np.sum(primary_noise**2)/np.sum(error**2))
print('Noise Attenuation= ',noise_attenuation)

#%% Save the results for further processing in matlab 
wavfile.write('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/results/Matlab/primary_noise_MN_0.1_16_2.wav', 16000, primary_noise)
wavfile.write('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/CRN/results/Matlab/error_MN_0.1_16_2.wav', 16000, error)

#%% calculate inference time of network developed by pytorch
device = torch.device("cuda")
mynetwork.eval().to(device)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

dummy_input = torch.randn(1, 1,512, dtype=torch.float32).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(50):
    _ = mynetwork(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        dummy_input.to(device)
        _ = mynetwork(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        dummy_input.cpu()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        dummy_input.cpu()
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)

print('average inference time [ms]: ',mean_syn)
print('number of samples delay: ',16000*mean_syn/1000)
