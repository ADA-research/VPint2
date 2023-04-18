# Imports

import numpy as np
import pandas as pd

import rasterio

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import autosklearn.regression

import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from VPint.WP_MRP import WP_SMRP
import VPint.utils.baselines_2D

from skimage.exposure import match_histograms

import os
import sys
from time import sleep
import random

# Functions

def VPint_interpolation(target_grid,feature_grid,method="exact"):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        #MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b,:],LinearRegression(),max_gamma=5,min_gamma=0.1)
        MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b])
        mu = np.nanmean(target_grid[:,:,b]) + 2*np.nanstd(target_grid[:,:,b])
        if(method=='exact'):
            #pred_grid[:,:,b] = MRP.run(method='exact',prioritise_identity=True, 
            #                    priority_intensity='auto',auto_priority_max_iter=100,
            #                    auto_priority_strategy='grid',auto_priority_min=1,
            #                    auto_priority_proportion=0.8,auto_priority_max=7,auto_priority_subsample_strategy='max_contrast',
            #                    resistance=True,epsilon=0.12,mu=mu) # TODO: automate 
            pred_grid[:,:,b] = MRP.run(method='exact',
                           auto_adapt=True, auto_adaptation_verbose=False,
                           auto_adaptation_epochs=50, auto_adaptation_max_iter=100,
                           auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, 
                           resistance=True,prioritise_identity=True)
        else:
            MRP.train()
            pred_grid[:,:,b] = MRP.run(method=method)
    return(pred_grid)
    
def naive_replacement(target_grid,feature_grid):
    # Replace all non-missing pixels from target by feature grid
    
    # Flatten (can't use argwhere with 3D)
    pred_grid = feature_grid.copy().reshape((target_grid.shape[0]*target_grid.shape[1]*target_grid.shape[2]))
    flattened_original = target_grid.copy().reshape((target_grid.shape[0]*target_grid.shape[1]*target_grid.shape[2]))
    
    # Replace
    pred_grid[np.argwhere(~np.isnan(flattened_original))] = flattened_original[np.argwhere(~np.isnan(flattened_original))]
    
    # Back to 3D
    pred_grid = pred_grid.reshape((target_grid.shape[0],target_grid.shape[1],target_grid.shape[2]))
        
    return(pred_grid)
    
def histogram_matching_replacement(target_grid,feature_grid):
    # Replace all non-missing pixels from target by feature grid
    
    # Flatten (can't use argwhere with 3D)
    pred_grid = feature_grid.copy().reshape((target_grid.shape[0]*target_grid.shape[1]*target_grid.shape[2]))
    flattened_original = target_grid.copy().reshape((target_grid.shape[0]*target_grid.shape[1]*target_grid.shape[2]))
    
    # Replace
    pred_grid[np.argwhere(~np.isnan(flattened_original))] = flattened_original[np.argwhere(~np.isnan(flattened_original))]
    
    # Back to 3D
    pred_grid = pred_grid.reshape((target_grid.shape[0],target_grid.shape[1],target_grid.shape[2]))
    
    matched = match_histograms(pred_grid,target_grid,channel_axis=2)
        
    return(matched)

def regression_interpolation(target_grid,feature_grid): 
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        # While True try statements are bad, but these autosklearn delete errors
        # can sneakily ruin entire runs without being noticed.
        while True:
            try:
                autoskl_current_id = np.random.randint(0,999999)
                
                model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=120,
                    per_run_time_limit=30,
                    #memory_limit=1920,
                    tmp_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "temp",
                    output_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "out",
                    delete_tmp_folder_after_terminate=False,
                    delete_output_folder_after_terminate=False,
                )
                #model = SVR() # TODO: use auto-sklearn
            
                model = VPint.utils.baselines_2D.regression_train(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                if(not(model)):
                    print("AAAAAAA")
                pred_grid[:,:,b] = VPint.utils.baselines_2D.regression_run(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                break
            except:
                pass

    return(pred_grid)
    
def ARMA_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        # While True try statements are bad, but these autosklearn delete errors
        # can sneakily ruin entire runs without being noticed.
        while True:
            try:
                autoskl_current_id = np.random.randint(0,999999)
                
                model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=120,
                    per_run_time_limit=30,
                    #memory_limit=1920,
                    tmp_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "temp",
                    output_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "out",
                    delete_tmp_folder_after_terminate=False,
                    delete_output_folder_after_terminate=False,
                )
                #model = SVR() # TODO: use auto-sklearn
                sub_model = LinearRegression()
                
                model, sub_model, sub_error_grid = VPint.utils.baselines_2D.ARMA_train(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model,sub_model)
                pred_grid[:,:,b] = VPint.utils.baselines_2D.ARMA_run(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model,sub_model,sub_error_grid)
                break
            except:
                pass

    return(pred_grid)
    
def MA_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        while True:
            try:
                autoskl_current_id = np.random.randint(0,999999)
                
                model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=120,
                    per_run_time_limit=30,
                    #memory_limit=1920,
                    tmp_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "temp",
                    output_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "out",
                    delete_tmp_folder_after_terminate=False,
                    delete_output_folder_after_terminate=False,
                )
                #model = SVR() # TODO: use auto-sklearn
                sub_model = LinearRegression()
                
                model, sub_model, sub_error_grid = VPint.utils.baselines_2D.MA_train(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model,sub_model)
                pred_grid[:,:,b] = VPint.utils.baselines_2D.MA_run(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model,sub_model,sub_error_grid)
                break
            except:
                pass
                
    return(pred_grid)
    
    
def SAR_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        while True:
            try:
                autoskl_current_id = np.random.randint(0,999999)
                
                model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=120,
                    per_run_time_limit=30,
                    #memory_limit=1920,
                    tmp_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "temp",
                    output_folder="/scratch/arp/autosklearn/cld/reg/" + feature_name + "/" + str(autoskl_current_id) + "out",
                    delete_tmp_folder_after_terminate=False,
                    delete_output_folder_after_terminate=False,
                )
                #model = SVR() # TODO: use auto-sklearn
                
                model = VPint.utils.baselines_2D.SAR_train(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                pred_grid[:,:,b] = VPint.utils.baselines_2D.SAR_run(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                break
            except:
                pass

    return(pred_grid)
    

def CNN_interpolation(target_grid,feature_grid):
    pass


def run_patch(base_path,scene,patch_name,feature_name="all",plot=False,
                       possible_features=["msi_1w","msi_1m","msi_6m","msi_12m"],cloud_threshold=0.3,method="exact",buffer_mask=True,mask_buffer_size=5,algorithm="VPint"):

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
    
    if(algorithm=="VPint"):
        pred_grid = VPint_interpolation(target_grid,feature_grid,method=method) 
    elif(algorithm=="replacement"):
        pred_grid = naive_replacement(target_grid,feature_grid)
        mae = np.nanmean(np.absolute(pred_grid-target_grid_true))
        if(mae > 10):
            print("MAE>10: " + str(mae) + " for patch " + patch_name + " on " + feature)
    elif(algorithm=="replacement_matching"):
        pred_grid = histogram_matching_replacement(target_grid,feature_grid)
    elif(algorithm=="regression"):
        pred_grid = regression_interpolation(target_grid,feature_grid) 
    elif(algorithm=="SAR"):
        pred_grid = SAR_interpolation(target_grid,feature_grid)
    elif(algorithm=="MA"):
        pred_grid = MA_interpolation(target_grid,feature_grid)
    elif(algorithm=="ARMA"):
        pred_grid = ARMA_interpolation(target_grid,feature_grid)
    elif(algorithm=="ResNet"):
        pred_grid = ResNet_interpolation(target_grid,feature_grid,mask_grid)
    else:
        pred_grid = None
        print("Warning: invalid method")
        asdlfkjf
    
    return(pred_grid)
    
def mask_buffer(mask,passes=1):
    for p in range(0,passes):
        new_mask = mask.copy()
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if(np.isnan(mask[i,j])):
                    if(i>0):
                        new_mask[i-1,j] = np.nan
                    if(i<mask.shape[0]-1):
                        new_mask[i+1,j] = np.nan
                    if(j>0):
                        new_mask[i,j-1] = np.nan
                    if(j<mask.shape[1]-1):
                        new_mask[i,j+1] = np.nan
        mask = new_mask
    return(mask)
    
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

if(len(sys.argv) != 11):
    print("Usage: python SEN2-MSI-T-repression.py [benchmark base path] [result base path] [meta file path] [scene name] [feature set name] [cloud confidence threshold] [cloud proportion min] [cloud proportion max] [VPint method] [algorithm]")

cloud_prop_min = float(sys.argv[7])
cloud_prop_max = float(sys.argv[8])
if(cloud_prop_max > 0.9):
    cloud_prop_max = 0.9 # I'm sorry it's so many files to change otherwise, should fix
cloud_threshold = float(sys.argv[6])

method = sys.argv[9]

save_path = sys.argv[2]
meta_path = sys.argv[3]
base_path = sys.argv[1]

scene = sys.argv[4]
feature_names = {
    "1":"msi_1w",
    "2":"msi_1m",
    "3":"msi_6m",
    "4":"msi_12m",
    "5":"all"
}
feature_name = feature_names[sys.argv[5]]

algorithm = sys.argv[10]

replace = True # Set to True to overwrite existing runs



names = get_patch_names(meta_path,cloud_prop_min,cloud_prop_max)
if(len(names)==0):
    print("WARNING: no patches satisfy criteria for scene " + scene + " at cloud proportion " + str(cloud_prop_min) + " - " + str(cloud_prop_max))


currdir = save_path + "/results"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
currdir = currdir + "/" + scene
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass

currdir = currdir + "/" + algorithm + "_" + str(cloud_prop_min) + "_" + str(cloud_prop_max)

if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
np.set_printoptions(threshold=np.inf)

log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.FATAL)

# Run
        
random.shuffle(names) # For running multiple tasks to speed up
for name in names:
    currdir1 = currdir + "/" + name
    if(not(os.path.exists(currdir1))):
        try:
            os.mkdir(currdir1)
        except:
            pass

    currdir1 = currdir1 + "/" + feature_name + ".npy"
    if(replace or not(os.path.exists(currdir1))):
        #print("No result yet for " + currdir1)
        pred_grid = run_patch(base_path,scene,name,feature_name=feature_name,plot=False,method=method,algorithm=algorithm,buffer_mask=False)
        if(type(pred_grid) != type(False)): # just because if(pred_grid) is ambiguous
            np.save(currdir1,pred_grid)
        
print("Terminated successfully")