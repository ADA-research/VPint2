# Imports

import numpy as np
import pandas as pd

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import autosklearn.regression

#import tensorflow as tf
#from tensorflow.keras import Model, Input, regularizers
#from tensorflow.keras import layers, models
#from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from VPint.WP_MRP import WP_SMRP
import VPint.utils.baselines_2D

import baselines
import baselines.NSPI
import baselines.WLR

#from skimage.exposure import match_histograms

import os
import sys
import time
import random
import pickle
import multiprocessing

# Helper functions


def regression_band_train(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))
    
    training_size = (~np.isnan(grid[:,:,0])).sum()
    if(training_size == 0):
        return(False)

    X_train = np.zeros((training_size,f_grid.shape[2]))
    y_train = np.zeros((training_size,grid.shape[2]))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j,0]))):
                X_train[c,:] = f_grid[i,j,:]
                y_train[c,:] = grid[i,j,:]
                c += 1
    
    model.fit(X_train, y_train)
    return(model)

def regression_band_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j,0])):
                f = f_grid[i,j,:]
                f = f.reshape((1,len(f)))
                pred = model.predict(f)
                pred_grid[i,j,:] = pred[0]
    
    return(pred_grid)



def regression_train_old(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))
    
    training_size = (~np.isnan(grid[:,:,0])).sum()
    if(training_size == 0):
        return(False)

    X_train = np.zeros((training_size,f_grid.shape[2]))
    y_train = np.zeros((training_size,grid.shape[2]))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j,0]))):
                X_train[c,:] = f_grid[i,j,:]
                y_train[c,:] = grid[i,j,:]
                c += 1
    
    model.fit(X_train, y_train)
    return(model)

def regression_run_old(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j,0])):
                f = f_grid[i,j,:]
                f = f.reshape((1,len(f)))
                pred = model.predict(f)
                pred_grid[i,j,:] = pred[0]
    
    return(pred_grid)
    
    
def SAR_train(grid,f_grid,model):

    # Compute nanmean of grid for mean imputation
    mean_val = np.nanmean(grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))

    training_size = (~np.isnan(grid)).sum()
    if(training_size == 0):
        return(False)
    num_features = f_grid.shape[2] + 4*grid.shape[2]

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size,grid.shape[2]))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j,0]))):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_val(grid,i,j,mean_val)
                X_train[c,0:-4*grid.shape[2]] = f1
                X_train[c,-4*grid.shape[2]:] = f2
                y_train[c,:] = grid[i,j,:]
                c += 1
    
    model.fit(X_train, y_train)
    
    return(model)
    
    
def SAR_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))
    
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j,0])):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_val(grid,i,j,mean_val)
                f = np.zeros((1,len(f1)+len(f2)))
                f[0,0:-4*grid.shape[2]] = f1
                f[0,-4*grid.shape[2]:] = f2
                pred = model.predict(f)
                pred_grid[i,j] = pred[0]
    
    return(pred_grid)



def spatial_lag_error(grid,f_grid,i,j,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    d = grid.shape[2]
    
    vec = np.ones((d,4)) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j,:]
        if(not(np.isnan(val).any())):
            vec[:,0] = val
    if(j < w):
        # Right
        val = grid[i,j+1,:]
        if(not(np.isnan(val).any())):
            vec[:,1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j,:]
        if(not(np.isnan(val).any())):
            vec[:,2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1,:]
        if(not(np.isnan(val).any())):
            vec[:,3] = val
            
    vec = vec.flatten()
            
    return(vec)
    
def spatial_lag_val(grid,i,j,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    d = grid.shape[2]
    
    vec = np.ones((d,4)) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j,:]
        if(not(np.isnan(val).any())):
            vec[:,0] = val
    if(j < w):
        # Right
        val = grid[i,j+1,:]
        if(not(np.isnan(val).any())):
            vec[:,1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j,:]
        if(not(np.isnan(val).any())):
            vec[:,2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1,:]
        if(not(np.isnan(val).any())):
            vec[:,3] = val
            
    vec = vec.flatten()
            
    return(vec)


def ARMA_train(grid,f_grid,model,sub_model):

    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))
    
    sub_model = SAR_train(grid,f_grid,sub_model)
    sub_pred_grid = SAR_run(grid,f_grid,sub_model)
    
    sub_error_grid = sub_pred_grid - grid
    
    mean_error = np.nanmean(sub_error_grid)
    mean_val = np.nanmean(grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    training_size = (~np.isnan(grid)).sum()
    if(training_size == 0):
        return(False)
    num_features = f_grid.shape[2] + 8*grid.shape[2]

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size,grid.shape[2]))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j,0]))):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                f3 = spatial_lag_val(grid,i,j,mean_val)
                X_train[c,0:-8*grid.shape[2]] = f1
                X_train[c,-8*grid.shape[2]:-4*grid.shape[2]] = f2
                X_train[c,-4*grid.shape[2]:] = f3
                y_train[c,:] = grid[i,j,:]
                c += 1
    
    model.fit(X_train, y_train)
    
    return(model,sub_model,sub_error_grid)
    
    
def ARMA_run(grid,f_grid,model,sub_model,sub_error_grid):
    height = grid.shape[0]
    width = grid.shape[1]
    if(len(grid.shape) < 3):
        grid = grid.reshape((grid.shape[0], grid.shape[1], 1))
    
    mean_error = np.nanmean(sub_error_grid)
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j,0])):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                f3 = spatial_lag_val(grid,i,j,mean_val)
                f = np.zeros((1,len(f1)+len(f2)+len(f3)))
                f[0,0:-8*grid.shape[2]] = f1
                f[0,-8*grid.shape[2]:-4*grid.shape[2]] = f2
                f[0,-4*grid.shape[2]:] = f3
                pred = model.predict(f)
                pred_grid[i,j,:] = pred[0]
    
    return(pred_grid)
    

# Main functions

def VPint_interpolation(target_grid,feature_grid,method="exact", use_IP=True, use_EB=True):
    pred_grid = target_grid.copy()
    feature_grid = np.clip(feature_grid, 0, 10000)
    for b in range(0,target_grid.shape[2]): 
        MRP = WP_SMRP(target_grid[:,:,b], feature_grid[:,:,b])
        mu = np.nanmean(target_grid[:,:,b]) + 2*np.nanstd(target_grid[:,:,b])
        if(method=='exact'):
            pred_grid[:,:,b] = MRP.run(method='exact',
                           auto_adapt=True, auto_adaptation_verbose=False,
                           auto_adaptation_epochs=10, auto_adaptation_max_iter=100,
                           auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, clip_val=10000,
                           resistance=use_EB,prioritise_identity=use_IP)
        else:
            MRP.train()
            pred_grid[:,:,b] = MRP.run(method=method)
    return(pred_grid)
    
    
    
def VPint_multi(target_grid, feature_grid, method="exact", use_IP=True, use_EB=True):
    # Multiprocessing VPint version code written by Dean van Laar
    
    manager = multiprocessing.Manager()
    pred_dict = manager.dict()
    pred_grid = target_grid.copy()
    grid_combos = []
    bands = []
    procs = len(range(0, target_grid.shape[2]))

    # Create lists containing the bands in order
    for b in range(0, target_grid.shape[2]):
        targetc = target_grid[:, :, b]
        feature = feature_grid[:, :, b]
        band0 = b
        grid_combos.append([targetc, feature])
        bands.append(band0)

    # Start the processes
    jobs = []
    for i in range(0, procs):
        process = multiprocessing.Process(target=VPint_multi_single,
                                          args=(pred_dict, grid_combos[i], bands[i]))
        jobs.append(process)
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

        # Ensure all of the processes have finished
        if j.is_alive():
            pass
            #print("Job is not finished!")

    #Sort the dictionary after running VPint on the bands
    sorted_dict = dict(sorted(pred_dict.items()))
    pred0 = np.array([*sorted_dict.values()])
    pred1 = pred0.swapaxes(0, 2)
    pred_grid_final = pred1.swapaxes(0, 1)
    return(pred_grid_final)
    
    
    
def VPint_multi_single(pred_dict, grids, band, use_IP=True, use_EB=True):
    MRP = WP_SMRP(grids[0], grids[1])
    pred_dict[band] = MRP.run(method='exact',
                                 auto_adapt=True, auto_adaptation_verbose=False,
                                 auto_adaptation_epochs=25, auto_adaptation_max_iter=100,
                                 auto_adaptation_strategy='random', auto_adaptation_proportion=0.8,
                                 resistance=use_EB, prioritise_identity=use_IP)
    

    
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
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    #for b in range(0,target_grid.shape[2]): 
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    while True:
        try:
            autoskl_current_id = np.random.randint(0,999999)

            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                #memory_limit=1920,
                tmp_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "temp",
                output_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "out",
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
            )
            
            target_grid = target_grid.astype(np.float64) / 10000.0 # This is to force sklearn to recognise dtype as float

            model = regression_train_old(target_grid,feature_grid,model)
            if(not(model)):
                print("AAAAAAA")
            else:
                pred_grid = regression_run_old(target_grid,feature_grid,model)
    
            pred_grid = pred_grid * 10000 # To restore values to the original range
            break
        except Exception as ex:
            print("Error occurred for regression, keep an eye on this: ")
            print(ex)
            print("NanMean: ", np.nanmean(pred_grid))
            print("NanMin: ", np.nanmin(pred_grid))
            print("NanMax: ", np.nanmax(pred_grid))
            print("All nan in pred grid?", np.all(np.isnan(pred_grid)))
            print("Any nan in feature grid?", np.any(np.isnan(feature_grid)))
            print("\n")
            return(False)

    return(pred_grid)
    
    
    
def regression_band_interpolation(target_grid,feature_grid, max_failures=5): 
    feature_grid = np.clip(feature_grid, 0, 10000)

    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        # While True try statements are bad, but these autosklearn delete errors
        # can sneakily ruin entire runs without being noticed.
        fail_count = 0
        while True:
            try:
                autoskl_current_id = np.random.randint(0,999999)
                
                model = autosklearn.regression.AutoSklearnRegressor(
                    time_left_for_this_task=120,
                    per_run_time_limit=30,
                    #memory_limit=1920,
                    tmp_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "temp",
                    output_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "out",
                    delete_tmp_folder_after_terminate=False,
                    delete_output_folder_after_terminate=False,
                )
                
                target_grid = target_grid.astype(np.float64) / 10000.0 # This is to force sklearn to recognise dtype as float
                feature_grid = feature_grid.astype(np.float64) / 10000.0
            
                model = VPint.utils.baselines_2D.regression_train(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                if(not(model)):
                    print("AAAAAAA")
                pred_grid[:,:,b] = VPint.utils.baselines_2D.regression_run(target_grid[:,:,b],feature_grid[:,:,b].reshape(feature_grid.shape[0],feature_grid.shape[1],1),model)
                pred_grid[:,:,b] = pred_grid[:,:,b] * 10000
                break
            except Exception as e:
                fail_count += 1
                if(fail_count >= 5):
                    print("Failure for a patch; returning feature grid as backup.")
                    print(e)
                    pred_grid = feature_grid[:,:,:,0]
                    for i in range(0, pred_grid.shape[0]):
                        for j in range(0, pred_grid.shape[1]):
                            if(not(np.isnan(target_grid[i,j,0]))):
                                pred_grid[i,j,:] = target_grid[i,j,:]
                    return(pred_grid)

    return(pred_grid)
    
    
def regression_RF_interpolation(target_grid,feature_grid): 
    pred_grid = target_grid.copy()
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    #for b in range(0,target_grid.shape[2]):  
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    
    while True:
        try:
            autoskl_current_id = np.random.randint(0,999999)
            
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                #memory_limit=1920,
                tmp_folder="/scratch/arp/autosklearn/cld/regRF/" + str(autoskl_current_id) + "temp",
                output_folder="/scratch/arp/autosklearn/cld/regRF/" + str(autoskl_current_id) + "out",
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
                ensemble_size=1,
                include_estimators=['random_forest'],
            )
            
            target_grid = target_grid.astype(np.float64) / 10000.0 # This is to force sklearn to recognise dtype as float
            
            model = regression_train(target_grid,feature_grid,model)
            if(not(model)):
                print("AAAAAAA")
            else:
                pred_grid = regression_run(target_grid,feature_grid,model)
            break
        except Exception as ex:
            print("Error occurred for regression RF, keep an eye on this: ")
            print(ex)
            print("NanMean: ", np.nanmean(pred_grid))
            print("NanMin: ", np.nanmin(pred_grid))
            print("NanMax: ", np.nanmax(pred_grid))
            print("All nan in pred grid?", np.all(np.isnan(pred_grid)))
            print("Any nan in feature grid?", np.any(np.isnan(feature_grid)))
            print("\n")
            return(False)

    return(pred_grid)
    
    
def regression_MLP_interpolation(target_grid,feature_grid): 
    # This needs to use different env with updated autosklkearn

    pred_grid = target_grid.copy()
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    # But MLP does not support multi-output regression, so need new model per band
    
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    #while True:
    #    try:
    autoskl_current_id = np.random.randint(0,999999)
    
    model = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        #memory_limit=1920,
        tmp_folder="/scratch/arp/autosklearn/cld/regMLP/" + str(autoskl_current_id) + "temp",
        #output_folder="/scratch/arp/autosklearn/cld/regMLP/" + str(autoskl_current_id) + "out",
        delete_tmp_folder_after_terminate=False,
        #delete_output_folder_after_terminate=False,
        ensemble_size=1,
        include={'regressor':['mlp']},
    )

    model = regression_train(target_grid[:,:,b],feature_grid,model)
    if(not(model)):
        print("AAAAAAA")
    else:
        pred_grid = regression_run(target_grid,feature_grid,model)
    #        break
    #    except Exception as ex:
    #        print("Error occurred, keep an eye on this: ")
    #        print(ex)

    return(pred_grid)
    
def ARMA_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    #for b in range(0,target_grid.shape[2]): 
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    while True:
        try:
            autoskl_current_id = np.random.randint(0,999999)
            
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                #ensemble_memory_limit=1920,
                tmp_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "temp",
                output_folder="/scratch/arp/autosklearn/cld/reg/" + str(autoskl_current_id) + "out",
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
            )
            #model = SVR() # TODO: use auto-sklearn
            sub_model = LinearRegression()
            
            model, sub_model, sub_error_grid = ARMA_train(target_grid,feature_grid,model,sub_model)
            pred_grid = ARMA_run(target_grid,feature_grid,model,sub_model,sub_error_grid)
            break
        except Exception as ex:
            print("Error occurred, keep an eye on this: ")
            print(ex)

    return(pred_grid)
    
    
def ARMA_RF_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    #for b in range(0,target_grid.shape[2]):  
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    while True:
        try:
            autoskl_current_id = np.random.randint(0,999999)
            
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                #memory_limit=1920,
                tmp_folder="/scratch/arp/autosklearn/cld/regRF/" + str(autoskl_current_id) + "temp",
                output_folder="/scratch/arp/autosklearn/cld/regRF/" + str(autoskl_current_id) + "out",
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
                ensemble_size=1,
                include_estimators=['random_forest'],
            )
            sub_model = LinearRegression()
            
            model, sub_model, sub_error_grid = ARMA_train(target_grid,feature_grid,model,sub_model)
            pred_grid = ARMA_run(target_grid,feature_grid,model,sub_model,sub_error_grid)
            break
        except Exception as ex:
            print("Error occurred, keep an eye on this: ")
            print(ex)
            

    return(pred_grid)
    
def ARMA_MLP_interpolation(target_grid,feature_grid):
    pred_grid = target_grid.copy()
    feature_grid = feature_grid[:,:,:,0]
    # Probably better to run on all bands at once
    #for b in range(0,target_grid.shape[2]): 
    # While True try statements are bad, but these autosklearn delete errors
    # can sneakily ruin entire runs without being noticed.
    while True:
        try:
            autoskl_current_id = np.random.randint(0,999999)
            
            model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                #memory_limit=1920,
                tmp_folder="/scratch/arp/autosklearn/cld/regMLP/" + str(autoskl_current_id) + "temp",
                output_folder="/scratch/arp/autosklearn/cld/regMLP/" + str(autoskl_current_id) + "out",
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
                ensemble_size=1,
                include_estimators=['mlp'],
            )
            sub_model = LinearRegression()
            
            model, sub_model, sub_error_grid = ARMA_train(target_grid,feature_grid,model,sub_model)
            pred_grid = ARMA_run(target_grid,feature_grid,model,sub_model,sub_error_grid)
            break
        except Exception as ex:
            print("Error occurred, keep an eye on this: ")
            print(ex)

    return(pred_grid)
    
    
    
def load_product(path, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
                        bands_10m={"B2":1, "B3":2, "B4":3, "B8":7},
                        bands_20m={"B5":4, "B6":5, "B7":6, "B8A":8, "B11":10, "B12":11, "CLD":12},
                        bands_60m={"B1":0, "B9":9}):

    grid = None
    size_y = -1
    size_x = -1

    scales = [bands_10m, bands_20m, bands_60m, {}] # For bands that have multiple resolutions

    with rasterio.open(path) as raw_product:
        product = raw_product.subdatasets

    # Initialise grid
    with rasterio.open(product[1]) as bandset:
        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        grid = np.zeros((size_y, size_x, len(keep_bands))).astype(np.uint16)


    # Iterate over band sets (resolutions)
    resolution_index = 0
    for bs in product:
        with rasterio.open(bs, dtype="uint16") as bandset:
            desc = bandset.descriptions
            size_y_local = bandset.profile['height']
            size_x_local = bandset.profile['width']

            band_index = 1
            # Iterate over bands
            for d in desc:
                band_name = d.split(",")[0]

                if(band_name in keep_bands and band_name in scales[resolution_index]):
                
                    if(band_name in bands_10m):
                        b = bands_10m[band_name]
                        upscale_factor = (1/2)
                    elif(band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 1
                    elif(band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 3

                    band_values = bandset.read(band_index, 
                                        out_shape=(
                                            int(size_y_local * upscale_factor),
                                            int(size_x_local * upscale_factor)
                                        ),
                                        resampling=Resampling.bilinear
                                    )

                    #grid[:,:,b] = np.moveaxis(band_values, 0, -1)
                    grid[:,:,b] = band_values
                    
                band_index += 1

        resolution_index += 1

    return(grid)


def load_product_windowed(path, y_size, x_size, y_offset, x_offset, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
                        bands_10m={"B2":1, "B3":2, "B4":3, "B8":7},
                        bands_20m={"B5":4, "B6":5, "B7":6, "B8A":8, "B11":10, "B12":11, "CLD":12},
                        bands_60m={"B1":0, "B9":9}):

    grid = None
    size_y = -1
    size_x = -1

    scales = [bands_10m, bands_20m, bands_60m, {}] # For bands that have multiple resolutions

    with rasterio.open(path) as raw_product:
        product = raw_product.subdatasets

    # Initialise grid
    with rasterio.open(product[1]) as bandset:
        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(np.uint16)


    # Iterate over band sets (resolutions)
    resolution_index = 0
    for bs in product:
        with rasterio.open(bs, dtype="uint16") as bandset:
            desc = bandset.descriptions
            size_y_local = bandset.profile['height']
            size_x_local = bandset.profile['width']

            band_index = 1
            # Iterate over bands
            for d in desc:
                band_name = d.split(",")[0]

                if(band_name in keep_bands and band_name in scales[resolution_index]):
                
                    if(band_name in bands_10m):
                        b = bands_10m[band_name]
                        upscale_factor = (1/2)
                    elif(band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 1
                    elif(band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 3

                    # Output window using target resolution
                    window=Window(x_offset, y_offset, x_size, y_size)

                    # Second window for reading in local resolution
                    res_window = Window(x_offset*x_size/upscale_factor, y_offset*y_size/upscale_factor,
                                        window.width / upscale_factor, window.height / upscale_factor)
                    
                   

                    band_values = bandset.read(band_index, 
                                        out_shape=(
                                            window.height,
                                            window.width
                                        ),
                                        resampling=Resampling.bilinear,
                                        #masked=True, 
                                        window=res_window, 
                                    )

                    grid[:,:,b] = band_values
                    
                band_index += 1

        resolution_index += 1

    return(grid)


def run_patch(base_path, legend, scene, y_size, x_size, y_offset, x_offset, feature_name="img_1m", plot=False,
                       cloud_threshold=20, method="exact", buffer_mask=True, mask_buffer_size=5,algorithm="VPint"):
                       
    key_target = scene + "-target"
    key_feature = scene + "-" + feature_name
    key_mask = scene + "-mask"

    target_path = base_path + legend[scene]['target'] + ".zip"
    feature_path = base_path + legend[scene][feature_name] + ".zip"
    mask_path = base_path + legend[scene]['mask'] + ".zip"
    
    # Load target and mask first, just return target if no cloudy pixels in mask
    target = load_product_windowed(target_path, y_size, x_size, y_offset, x_offset).astype(float)
    if(not(np.any(target > 0))):
        # Black stuff for non-filling scenes?
        #print("All zeros for scene: ", scene)
        #print("\n")
        return(np.zeros(target.shape))
    mask = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
    
    if(not(np.any(mask > cloud_threshold))):
        return(target)
    
    # If there are any cloudy pixels, load features and run algorithms
    features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset).astype(float)
    features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))
    
   
    if(buffer_mask):
        mask_grid = mask_buffer(mask, mask_buffer_size)
    
  
    target_cloudy = target.copy()

    for i in range(0, target_cloudy.shape[0]):
        for j in range(0, target_cloudy.shape[1]):
            if(mask[i,j] > cloud_threshold):
                a = np.ones(target_cloudy.shape[2]) * np.nan
                target_cloudy[i,j,:] = a
                    
                    
    
    features = np.nan_to_num(features, nan=np.nanmean(features))
    
    t1 = time.time()
    
    if(algorithm=="VPint"):
        pred_grid = VPint_interpolation(target_cloudy, features, method=method, use_IP=True, use_EB=True)
    elif(algorithm=="VPint_no_IP"):
        pred_grid = VPint_interpolation(target_cloudy, features, method=method, use_IP=False, use_EB=True)
    elif(algorithm=="VPint_no_EB"):
        pred_grid = VPint_interpolation(target_cloudy, features, method=method, use_IP=True, use_EB=False) 
    elif(algorithm == "VPint_multi"):
        pred_grid = VPint_multi(target_cloudy, features, use_IP=True, use_EB=True)
    elif(algorithm=="replacement"):
        pred_grid = naive_replacement(target_cloudy, features)
    elif(algorithm=="replacement_matching"):
        pred_grid = histogram_matching_replacement(target_cloudy, features)
    elif(algorithm=="regression"):
        pred_grid = regression_interpolation(target_cloudy,features)
    elif(algorithm=="regression_band"):
        pred_grid = regression_band_interpolation(target_cloudy,features)
    elif(algorithm=="regression_RF"):
        pred_grid = regression_RF_interpolation(target_cloudy,features)
    elif(algorithm=="regression_MLP"):
        pred_grid = regression_MLP_interpolation(target_cloudy,features)
    elif(algorithm=="ARMA"):
        pred_grid = ARMA_interpolation(target_cloudy,features)
    elif(algorithm=="ARMA_RF"):
        pred_grid = ARMA_RF_interpolation(target_cloudy,features)
    elif(algorithm=="ARMA_MLP"):
        pred_grid = ARMA_MLP_interpolation(target_cloudy,features)
    elif(algorithm=="ResNet"):
        pred_grid = ResNet_interpolation(target_cloudy, features, mask_grid)
    elif(algorithm == "NSPI"):
        pred_grid = baselines.NSPI.NSPI(target, features[:,:,:,0], mask)
    elif(algorithm == "WLR"):
        pred_grid = baselines.WLR.WLR(target, features[:,:,:,0], mask)
    else:
        pred_grid = None
        print("Warning: invalid method")
        asdlfkjf
        
    t2 = time.time()
    
    return(t2-t1)
    
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
    



# Setup

#conditions_features = ["1w", "1m", "3m", "6m", "12m"]
conditions_features = ["1w", "1m", "3m", "6m"]
conditions_features = ["1m"]

conditions_algorithms_VPint = [
    "VPint", 
    "VPint_multi",
    "VPint_no_IP",
    "VPint_no_EB", 
    #"replacement",
    #"regression",
    #"regression_RF",
    #"regression_MLP",
    #"ARMA",
    #"ARMA_RF",
    #"ARMA_MLP",
]

conditions_algorithms_regression = [
    #"VPint", 
    #"VPint_no_IP",
    #"VPint_no_EB", 
    #"replacement",
    "regression",
    "regression_band",
    "regression_RF",
    #"regression_MLP",
    #"ARMA",
    #"ARMA_RF",
    #"ARMA_MLP",
]

conditions_algorithms_replacement = [
    #"VPint", 
    #"VPint_no_IP",
    #"VPint_no_EB", 
    "replacement",
    #"regression",
    #"regression_RF",
    #"regression_MLP",
    #"ARMA",
    #"ARMA_RF",
    #"ARMA_MLP",
]

conditions_algorithms_interpolation = [
    #"VPint", 
    #"VPint_no_IP",
    #"VPint_no_EB", 
    #"replacement",
    #"regression",
    #"regression_RF",
    #"regression_MLP",
    #"ARMA",
    #"ARMA_RF",
    #"ARMA_MLP",
    "NSPI",
    "WLR",
]

conditions_algorithms_all = [
    "VPint", 
    "VPint_multi",
    #"VPint_no_IP",
    #"VPint_no_EB", 
    "replacement",
    #"regression",
    "regression_band",
    #"ARMA",
    #"ARMA_RF",
    #"ARMA_MLP",
    "NSPI",
    "WLR",
]

if(sys.argv[3] == "VPint"):
    conditions_algorithms = conditions_algorithms_VPint
elif(sys.argv[3] == "regression"):
    conditions_algorithms = conditions_algorithms_regression
elif(sys.argv[3] == "replacement"):
    conditions_algorithms = conditions_algorithms_replacement
elif(sys.argv[3] == "interpolation"):
    conditions_algorithms = conditions_algorithms_interpolation
elif(sys.argv[3] == "all"):
    conditions_algorithms = conditions_algorithms_all
else:
    print("Invalid algorithm set")
    sdlkfj

save_path = sys.argv[2]
base_path = sys.argv[1]

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
    
num_bins = int(sys.argv[4])
num_samples = int(sys.argv[5])
step_size = 100 / num_bins

    

    

conditions_bins = np.arange(0, num_bins)
          
#random.shuffle(conditions_scenes) # Allows for multiple tasks


i = 1
conds = {}

for b in conditions_bins:
    for feature in conditions_features:
        for alg in conditions_algorithms:
            conds[i] = {"features":feature, "algorithm":alg, "bin":b}
            i += 1
            
this_run_cond = conds[int(sys.argv[6])]

bin_min = step_size * this_run_cond['bin']
bin_max = step_size * (this_run_cond['bin']+1)

random.seed(42) # To keep things consistent and reproducible

candidates = []
with open("cloud_proportions.pkl", 'rb') as fp:
    clp = pickle.load(fp)
    for scene, d in clp.items():
        for patch, prop in d.items():
            if(prop > bin_min and prop < bin_max):
                candidates.append(scene + "_" + patch + "_" + str(prop))
random.shuffle(candidates)
candidates = candidates[:min(num_samples, len(candidates))]


# Some parameters

replace = False # Set to True to overwrite existing runs
size_y = 256
size_x = 256



currdir = save_path + "results_efficiency"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    

        
np.set_printoptions(threshold=np.inf)

log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.FATAL)

# Run

# Create file per alg
res_path = currdir + "/" + this_run_cond["algorithm"] + ".csv"
if(replace or not(os.path.exists(res_path))):
    header = "scene,patch,clp,running_time\n"
    with open(res_path, 'w') as fp:
        fp.write(header)
else:
    print("Already done; let's kill this job with very little elegance because I'm too lazy to implement a simple exception")
    aslkdjflksdjf


# Iterate over patches in candidates
for c in candidates:
    c2 = c.split("_") # Ugly but can't help it now
    this_run_cond["scene"] = c2[0] + "_" + c2[1] + "_" + c2[2]
    patch_name = c2[3] + "_" + c2[4]
    y_offset = int(c2[3][1:])
    x_offset = int(c2[4][1:])
    clp = c2[-1]
    
    t = run_patch(base_path, legend, this_run_cond["scene"], size_y, size_x, y_offset, x_offset, feature_name=this_run_cond["features"], plot=False, algorithm=this_run_cond["algorithm"], buffer_mask=False)
    
    s = this_run_cond["scene"] + "," + patch_name + "," + str(clp) + "," + str(t) + "\n"
    with open(res_path, 'a') as fp:
        fp.write(s)

        
print("Terminated successfully")