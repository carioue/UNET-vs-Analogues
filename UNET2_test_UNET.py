import xarray as xr
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import os

""" This code test the model 'model_40membres_input3286' on a member not used in the training """
PATH = os.getcwd()

slp_test = xr.open_dataset(PATH+'/data/psl_day_MIROC6_unet_r5i1p1f1_CR.nc').psl[36500:36500+26645] #get SLP from 1950 to 2022 (from a member not used in the training)
T_target_test = xr.open_dataset(PATH+'/data/anomalies_r5.nc').tasano[25550:25550+26645] #get corresponding temperature anomalies from 1950 to 2022


UNET = load_model(PATH+'/trained_models/model_40members_input3286') #load the UNET already trained
T_pred = UNET.predict(slp_test) #apply the UNET on testing data
T_pred = T_pred.reshape((T_pred.shape[0],T_pred.shape[1],T_pred.shape[2])) 

np.save(PATH+'/data/Tpred_unet_r5.npy',T_pred)
