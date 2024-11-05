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
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

#GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)
		
		
#-------------------------------------------CNN layers-----------------------------------------------------------
model_trained = load_model('/model_40membres_input3286') #load an already trained UNET		
encoder_output = model_trained.get_layer('conv2d_7').output #get the last layer of the UNET's encoder 
x = Flatten()(encoder_output)

#Season prediction
output = Dense(4, activation='softmax')(x) #the output of the new CNN is a vector of 4 values

##Day prediction
#output = Dense(2, activation='linear')(x)

##Year prediction
#output = Dense(221, activation='softmax')(x)

new_model = Model(inputs=model_trained.input, outputs=output)

for layer in model_trained.layers: # UNET encoder parameters are trainable, only those of the dense layer are
    layer.trainable = False
    
#------------------------------------------------------------------------------------------------------------------


#----------------------------------------Load Data------------------------------------------------------------------
#Input
slp_ano = xr.open_dataset('/MIROC6/psl_day_MIROC6_unet_r10_r50_18802100_CentreReduit.nc').psl[0:80665]  # normalized sea level pressure of the 1 training members (input)

#Output
#IF season prediction
saison = [] #output for the training
for i in range(len(slp_ano)):
    if int(str(slp_ano.time.values[i])[5:7]) == 1 or int(str(slp_ano.time.values[i])[5:7]) == 2 or int(str(slp_ano.time.values[i])[5:7]) == 12:
        saison.append(1)
    if int(str(slp_ano.time.values[i])[5:7]) == 3 or int(str(slp_ano.time.values[i])[5:7]) == 4 or int(str(slp_ano.time.values[i])[5:7]) == 5:
        saison.append(2)
    if int(str(slp_ano.time.values[i])[5:7]) == 6 or int(str(slp_ano.time.values[i])[5:7]) == 7 or int(str(slp_ano.time.values[i])[5:7]) ==8:
        saison.append(3)
    if int(str(slp_ano.time.values[i])[5:7]) == 9 or int(str(slp_ano.time.values[i])[5:7]) == 10 or int(str(slp_ano.time.values[i])[5:7]) == 11:
        saison.append(4)
        
enc=OneHotEncoder() #the output is encoded with One Hot Encoder
target_saison = np.array(saison).reshape(-1, 1)
target_saison_ohe = enc.fit_transform(target_saison)
target = target_saison_ohe.toarray()

#IF day prediction
#cos_j = [np.cos((2*math.pi*d)/365) for d in range(len(slp_ano))]
#sin_j = [np.sin((2*math.pi*d)/365) for d in range(len(slp_ano))]  
#cos_sin = xr.Dataset({'cos': (['time'], cos_j),
#                      'sin': (['time'], sin_j)},
#                     coords={'time': slp_ano.time})
#data_array = cos_sin.to_array(dim='variable')
#target=data_array.transpose('time', 'variable')


#IF year prediction
#annee =[]
#for i in range(len(slp_ano)):
#    annee.append(int(str(slp_ano.time.values[i])[0:4]))
        
#enc=OneHotEncoder()
#target_annee = np.array(annee).reshape(-1, 1)
#target_annee_ohe = enc.fit_transform(target_annee)
#target = target_annee_ohe.toarray()

X_train, X_test, Y_train, Y_test = train_test_split(slp_ano, target, test_size = 0.2)

#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------Training----------------------------------------------------------
new_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy') #the categorical_crossentropy is used for classification problems (for seasons and years predictions)
									    # use loss='mse' for day prediction

callback = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1), EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                 ModelCheckpoint('model_season', monitor='val_loss', verbose=1, save_best_only=True),History()] #model_season or model_day or model_year
      
tf.config.run_functions_eagerly(True)
new_model.fit(x=X_train, y=Y_train, batch_size=1000,epochs=20,callbacks=callback, validation_data=(X_test,Y_test))





