import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2,pow
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History
from sklearn.model_selection import train_test_split 
import os

PATH = os.getcwd()

#GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)
		
		
# Code from http://gitlab.meteo.fr/haradercoustaue/dourya_weather/-/blob/main/emu_funs.py?ref_type=heads		

#-----------------------------------------UNET construction-----------------------------------------------
def highestPowerof2(n):
    res = 0;
    for i in range(n, 0, -1):
        # If i is a power of 2
        if ((i & (i - 1)) == 0):

            res = i;
            break;
    return res;
    
#Let's define a basic CNN with few convolutions and MaxPooling
def block_conv(conv, filters):
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def block_up(conv, filters):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = block_conv(conv, filters)
    return conv

def block_up_conc(conv, filters,conv_conc):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = concatenate([conv,conv_conc])
    conv = block_conv(conv, filters)
    return conv
        
    
shape_inputs=[32,86]
size=np.min([highestPowerof2(shape_inputs[0]),highestPowerof2(shape_inputs[1])]) #used to transform the image into an image with dimensions to the power of 2 
diff_lat=shape_inputs[0]-size+1
diff_lon=shape_inputs[1]-size+1

inputUnet = Input(shape=(32,86,1))    
    
    
c1 = Conv2D(32, (diff_lat,diff_lon), padding = "valid", activation = "tanh")(inputUnet) #transform the Input image into 32 channels of 32x32 images 
c1=BatchNormalization()(c1)
c1=Activation('relu')(c1)

conc_list=[]

# three blocks containing convolution and maxpooling operations. The image is transformed into a 4x4x128 image
for i in range(3):
    c2 = block_conv(c1, 32*int(pow(2,i)))
    c1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(c2)
    conc_list.append(c2)
    
c1=Conv2D(128, 3, padding='same')(c1)

#three blocks of up-convolutions : the 4x4x128 image is transformed into a 32x32x32 image
for i in range(3,0,-1):
    c1 = block_up_conc(c1, 32*int(pow(2,i-1)),conc_list[int(i-1)])

#last convolution to get a 32x32x1 image    
c5 = Conv2D(1, 3, activation='linear', padding='same')(c1)


model = Model(inputs=inputUnet, outputs=c5)    
#---------------------------------------------------------------------------------------------

#----------------------------------------- Load Data -----------------------------------------
"""MIROC6 sea level pressure of 40 simulations (members) are available thanks to CMIP6 working groups. Here a pre-processing is done on raw data to normalize data: mean and standard deviation 
are computed for each grid point on the daily data of the 40 concatenated training members.
Same for raw temperatures and then a pre-process is done to calculate anomalies: non stationary normals are calculated with routines from https://gitlab.com/ribesaurelien/france_study.
"""

slp_ano = xr.open_dataset('/MIROC6/psl_day_MIROC6_unet_r10_r50_18802100_CentreReduit.nc').psl # normalized sea level pressure of the 40 training members (input)
tas_ano = xr.open_dataset('/MIROC6/anomalies_40membres.nc').tasano # temperature anomalies of the 40 training members (output)
    
X_train,X_test,Y_train,Y_test=train_test_split(slp_ano,tas_ano,test_size=0.2)  #The dataset is randomly split into 80% for training and 20% for evaluation.   
    

model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3),metrics=['mse'])    
    
callback = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1), EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                 ModelCheckpoint('model_40members_input3286', monitor='val_loss', verbose=1, save_best_only=True),History()] 

tf.config.run_functions_eagerly(True)

model.fit(x=X_train, y=Y_train, batch_size=1000,epochs=120,shuffle=True,callbacks=callback, validation_data=(X_test,Y_test))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
