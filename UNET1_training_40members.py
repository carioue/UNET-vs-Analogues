mport tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import xarray as xr
from tqdm import tqdm
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, History
 

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

def block_conv(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet(input_shape=(40, 42, 1)):
    inputs = layers.Input(shape=input_shape)
   
    # Initial Conv
    x = layers.Conv2D(32, (diff_lat, diff_lon), padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.tanh(x)

    # Encoder path
    enc1 = block_conv(x, 32)
    enc1_1 = layers.MaxPooling2D(pool_size=2)(enc1)
    enc2 = block_conv(enc1_1, 64)
    enc2_1 = layers.MaxPooling2D(pool_size=2)(enc2)
    enc3 = block_conv(enc2_1, 128)
    enc3_1 = layers.MaxPooling2D(pool_size=2)(enc3)
   
    # Bottleneck
    bottleneck = layers.Conv2D(128, 3, padding='same')(enc3_1)

    # Decoder path
    dec3 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(bottleneck)
    dec3 = layers.Concatenate()([dec3, enc3])
    dec3 = block_conv(dec3, 128)

    dec2 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(dec3)
    dec2 = layers.Concatenate()([dec2, enc2])
    dec2 = block_conv(dec2, 64)

    dec1 = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(dec2)
    dec1 = layers.Concatenate()([dec1, enc1])
    dec1 = block_conv(dec1, 32)

    # Output layer
    outputs = layers.Conv2D(1, 3, padding='same')(dec1)

    return models.Model(inputs, outputs)
    
class TemperatureSLPDataset(tf.keras.utils.Sequence):
    def __init__(self, slp_file, tas_file, batch_size=100):
        self.slp_data = xr.open_dataset(slp_file)['psl']
        self.tas_data = xr.open_dataset(tas_file)['tasano']
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.slp_data) / self.batch_size))

    def __getitem__(self, idx):
        slp = self.slp_data[idx * self.batch_size:(idx + 1) * self.batch_size].values
        tas = self.tas_data[idx * self.batch_size:(idx + 1) * self.batch_size].values

        # Add channel dimension
        slp_tensor = np.expand_dims(slp, axis=-1)
        tas_tensor = np.expand_dims(tas, axis=-1)
      
        return slp_tensor, tas_tensor
  
      
shape_inputs=[32,86]
size=np.min([highestPowerof2(shape_inputs[0]),highestPowerof2(shape_inputs[1])]) #used to transform the image into an image with dimensions to the power of 2 
diff_lat=shape_inputs[0]-size+1
diff_lon=shape_inputs[1]-size+1

#---------------------------------------------------------------------------------------------

#----------------------------------------- Load Data -----------------------------------------
"""MIROC6 sea level pressure of 40 simulations (members) are available thanks to CMIP6 working groups. Here a pre-processing is done on raw data to normalize data: mean and standard deviation are computed for each grid point on the daily data of the 40 concatenated training members.
Same for raw temperatures and then a pre-process is done to calculate anomalies: non stationary normals are calculated with routines from https://gitlab.com/ribesaurelien/france_study. 
80% of the data is randomly selected to form the training dataset and the remaining 20% for the validation dataset.
"""

slp_file_train = '/MIROC6/psl_day_MIROC6_1880_2100_TRAIN.nc' # normalized sea level pressure of 80% of the 40 training members (input)
tas_file_train = '/MIROC6/anomalies_1880_2100_TRAIN.nc' # temperature anomalies of 80% of the 40 training members (output)

slp_file_val = '/MIROC6/psl_day_MIROC6_1880_2100_VAL.nc' # normalized sea level pressure of 20% of the 40 training members (input)
tas_file_val = '/MIROC6/anomalies_1880_2100_VAL.nc' # temperature anomalies of 20% of the 40 training members (output)

# Datasets
dataset_train = TemperatureSLPDataset(slp_file_train, tas_file_train, batch_size=1000)
dataset_val = TemperatureSLPDataset(slp_file_val, tas_file_val, batch_size=1000)

#---------------------------------------------------------------------------------------------

#----------------------------------------- Model config --------------------------------------  
   
# Configuration
model = build_unet()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse',metrics=['mse'])    
    
callback = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1), EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                 ModelCheckpoint('model_40members_input3286', monitor='val_loss', verbose=1, save_best_only=True),History()] 

tf.config.run_functions_eagerly(True)

model.fit(dataset_train,epochs=100,callbacks = callback, validation_data=dataset_val)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
