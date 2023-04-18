# Imports

import numpy as np
import pandas as pd

import rasterio

import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import sys
from time import sleep
import random

# Functions


def cloud_only_mae(true,pred):
    # Compute MAE only on cloudy pixels. True values have to be INVERTED
    # compared to normal setup, i.e., NON-cloudy pixels are NaNs
    # Included but should not be used, as we don't do any training anymore here
    diff = tf.math.abs(tf.subtract(true,pred))
    #test = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
    no_nans = tf.where(tf.math.is_nan(diff), tf.zeros_like(diff), diff)
    mean = tf.math.reduce_mean(no_nans)
    return(mean)


def NN_interpolation(model,batch_in,batch_f):
    pred = model.predict([batch_in,batch_f])
    return(pred)


def load_data(base_path,scene,patch_name,feature_name="all",plot=False,
                       possible_features=["msi_1w","msi_1m","msi_6m","msi_12m"],cloud_threshold=0.3,buffer_mask=False,mask_buffer_size=5):
   
    
    target_path = base_path + "scenes/" + scene + "/" + patch_name + "_msi_target.tif"
    feature_path = base_path + "scenes/" + scene + "/" + patch_name + "_" + feature_name + ".tif"
    mask_path = base_path + "scenes/" + scene + "/" + patch_name + "_mask.tif"
    
    fp = rasterio.open(mask_path)
    mask_grid = np.moveaxis(fp.read().astype(float),0,-1)
    fp.close()
    
    if(buffer_mask):
        mask_grid = mask_buffer(mask_grid,mask_buffer_size)
    
    suitable = check_patch(mask_grid)
    if(not(suitable)):
        return(False)
    
    fp = rasterio.open(target_path)
    target_grid_true = np.moveaxis(fp.read().astype(float),0,-1) # to have height,width,band order
    fp.close()
    
    target_grid = target_grid_true.copy()
    #target_grid = np.nan_to_num(target_grid,nan=np.nanmean(target_grid))
    # TODO: vectorise
    for i in range(0,target_grid.shape[0]):
        for j in range(0,target_grid.shape[1]):
            if(mask_grid[i,j] > cloud_threshold):
                a = np.ones(target_grid.shape[2]) * np.nan
                target_grid[i,j,:] = a
                #for b in range(0,target_grid.shape[2]):
                #    target_grid[i,j,b] = np.nan
                    
                    
    if(feature_name == "all"):
        f_grid = np.zeros((target_grid.shape[0],target_grid.shape[1],target_grid.shape[2],len(possible_features)))
        f = 0
        for f_name in possible_features:
            feature_path = base_path + "scenes/" + scene + "/" + patch_name + "_" + f_name + ".tif"
            fp = rasterio.open(feature_path)
            f_grid[:,:,:,f] = np.moveaxis(fp.read().astype(float),0,-1)
            fp.close()
            f += 1
        feature_grid = np.mean(f_grid,axis=3)
    
    else:
        fp = rasterio.open(feature_path)
        feature_grid = np.moveaxis(fp.read().astype(float),0,-1)
        feature_grid = feature_grid.reshape((feature_grid.shape[0],feature_grid.shape[1],feature_grid.shape[2],1))
        fp.close()
    
    feature_grid = np.nan_to_num(feature_grid,nan=np.nanmean(feature_grid))
    
    
    
    return(target_grid,feature_grid)
    

    
def check_patch(grid):
    training_size = (~np.isnan(grid)).sum()
    if(training_size == 0):
        return(False)

    h = grid.shape[0] - 1
    w = grid.shape[1] - 1

    c = 0
    for i in range(0,grid.shape[0]):
        for j in range(0,grid.shape[1]):
            y2 = grid[i,j]
            if(not(np.isnan(y2))):
                if(i > 0):
                    # Top
                    y1 = grid[i,j]
                    if(not(np.isnan(y1))):
                        c += 1
                if(j < w):
                    # Right
                    y1 = grid[i,j+1]
                    if(not(np.isnan(y1))):
                        c += 1                        
                if(i < h):
                    # Bottom                       
                    y1 = grid[i+1,j]
                    if(not(np.isnan(y1))):
                        c += 1  
                if(j > 0):
                    # Left                       
                    y1 = grid[i,j-1]
                    if(not(np.isnan(y1))):
                        c += 1
                        
    if(c > 0):
        return(True)
    else:
        return(False)

def get_patch_names(meta_path,cloud_prop_min,cloud_prop_max):
    df = pd.read_csv(meta_path)
    df = df.loc[df['cloud_prop'] >= cloud_prop_min]
    df = df.loc[df['cloud_prop'] <= cloud_prop_max]
    names = df['patch_name'].values
    return(names)



# Setup

if(len(sys.argv) != 4):
    print("Usage: python SEN2-MSI-T-repression.py [benchmark base path] [result base path] [cloud confidence threshold]")


cloud_threshold = float(sys.argv[3])

save_path = sys.argv[2]
base_path = sys.argv[1]



features = ["msi_1w","msi_1m","msi_6m","msi_12m"]
scenes = ["vegetation_ukraine","dry_australia","urban_tokyo","flood_brazil","water_baikal"]
props = [(0.1,0.3),(0.3,0.7),(0.7,0.9)]

replace = True # Set to True to overwrite existing runs

batch_size = 16 # Hardcoded for convenience



currdir = save_path + "results"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    

        
np.set_printoptions(threshold=np.inf)

log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.FATAL)



# Run
        
batch_size = 16
batch_t = np.zeros((batch_size,256,256,10))
batch_f = np.zeros((batch_size,256,256,10))
save_paths = []
i = 0

# Load model

model = tf.keras.models.load_model('/scratch/arp/results/models/meraner_adaptation_original')#, compile=False)
# custom_objects={'loss': cloud_only_mae}

# Main loop

for scene in scenes:
    print(scene)
    currdir1 = currdir + "/" + scene
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass

    for prop in props: 
        currdir2 = currdir1 + "/NN_" + str(prop[0]) + "_" + str(prop[1])

        if(not(os.path.exists(currdir2))):
            try:
                os.mkdir(currdir2)
            except:
                pass
                
        for feature in features:
            meta_path = "/scratch/arp/data/SEN2-MSI-T/scenes/" + scene + "_full.csv"
            
            names = get_patch_names(meta_path,prop[0],prop[1])
            if(len(names)==0):
                print("WARNING: no patches satisfy criteria for scene " + scene + " at cloud proportion " + str(prop[0]) + " - " + str(prop[1]))
            random.shuffle(names) # For running multiple tasks to speed up
                
            for name in names:
                currdir3 = currdir2 + "/" + name
                if(not(os.path.exists(currdir3))):
                    try:
                        os.mkdir(currdir3)
                    except:
                        pass

                currdir4 = currdir3 + "/" + feature + ".npy"
                if(replace or not(os.path.exists(currdir4))):
                    save_paths.append(currdir4)
                    grid_t, grid_f = load_data(base_path,scene,name,feature_name=feature,plot=False,buffer_mask=False)
                    batch_t[i,:,:,:] = grid_t
                    batch_f[i,:,:,:] = grid_f[:,:,:,0]
                    batch_t = np.nan_to_num(batch_t,nan=np.nanmax(batch_t))
                    i += 1
                    
                    
                    if(i>=16):
                        pred_grids = NN_interpolation(model,batch_t,batch_f)
                        for j in range(0,16):
                            save_path = save_paths[j]
                            pred_grid = pred_grids[j,:,:,:]
                            
                            if(type(pred_grid) != type(False)): # just because 
                                np.save(save_path,pred_grid)
                        i = 0
                        save_paths = []
            
