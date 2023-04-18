import numpy as np
import pandas as pd

import json
import rasterio

import cv2
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

import sys
import os
import math
from math import log10, sqrt
import pickle


# Functions

def compute_measures(patchdir,ground_truth_path,mask_path,metric,scene_patches,aggregation_method='mean'):

    features = ["msi_1w","msi_1m","msi_6m","msi_12m"]
    measures = np.ones(len(features)) * np.nan
    
    i = 0
    for feature in features:
        scene = patchdir.split("/")[-4]
        patch = patchdir.split("/")[-2]
        feature_key = scene + "_" + feature
        invalid_patches = scene_patches[feature_key]
        if(patch + "_" + feature not in invalid_patches):
            #pd = patchdir + feature + ".json"
            pd = patchdir + feature + ".npy"
            #with open(pd,'r') as fp:
            #    pred_vals = np.array(eval(fp.read().replace("nan","np.nan"))).astype(float)
            pred_vals = np.load(pd)
            if(len(pred_vals.shape)!= 3):
                print("Found dims " + str(pred_vals.shape) + "for result " + pd)

            true_vals = np.moveaxis(rasterio.open(ground_truth_path).read(),0,-1)
            
            # Remove edge case nans; should only happen on true (faulty pixels), but
            # do both just in case.
            pred_mean = np.nanmean(pred_vals)
            true_mean = np.nanmean(true_vals)
            pred_vals = np.nan_to_num(pred_vals,nan=pred_mean)
            true_vals = np.nan_to_num(true_vals,nan=true_mean)
                
            mask = np.moveaxis(rasterio.open(mask_path).read(),0,-1)[:,:,0]
            if(metric == "MAE"):
                mae = compute_MAE(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = mae
            elif(metric == "MSE"):
                mse = compute_MSE(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = mse
            elif(metric == "PSNR"):
                psnr = compute_PSNR(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = psnr
            elif(metric == "SSIM"):
                ssim = compute_SSIM(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = ssim
            elif(metric == "Rsquared"):
                r_squared = compute_R_squared(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = r_squared
            elif(metric == "SAM"):
                sam = SAM_grid(true_vals,pred_vals,only_cloud=True,cloud_mask=mask,aggr=aggregation_method)
                measures[i] = sam
            elif(metric == "MAPE"):
                mape = compute_MAPE(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = mape
            elif(metric == "NDVI"):
                ndvi = compute_NDVI(true_vals,pred_vals,mask,aggr=aggregation_method)
                measures[i] = ndvi
            else:
                print("Non-supported metric: " + str(metric))
            i += 1
        else:
            i += 1
            
    return(measures)

    
def compute_MAE(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    diff = np.absolute(true-pred)
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>0.3]
    
    if(aggr=='mean'):
        mae = np.nanmean(flattened_diff)
    elif(aggr=='median'):
        mae = np.nanmedian(flattened_diff)
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    return(mae)
    
def compute_MSE(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    diff = (true-pred)**2
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>0.3]
    
    if(aggr=='mean'):
        mse = np.nanmean(flattened_diff)
    elif(aggr=='median'):
        mse = np.nanmedian(flattened_diff)
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    return(mse)
    
def compute_PSNR(true,pred,mask_2d,aggr='mean'):
    # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>0.3]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>0.3]
    
    if(aggr=='mean'):
        mse = np.nanmean((flattened_true - flattened_pred) ** 2) + 0.001 # 0.001 for smoothing
    elif(aggr=='median'):
        mse = np.nanmedian((flattened_true - flattened_pred) ** 2) + 0.001 # 0.001 for smoothing
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    
    if(mse == 0):
        return(1)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) / 100 # /100 because I want 0-1
    return(psnr)
    
def compute_R_squared_old(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    diff = (true-pred)**2
    SStot = (true-np.nanmean(true))**2
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_SStot = SStot.reshape((SStot.shape[0]*SStot.shape[1]*SStot.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>0.3]
    flattened_SStot = flattened_SStot[flattened_mask>0.3]
    
    r_squared_vec = np.divide(flattened_diff,flattened_SStot)
    
    if(aggr=='mean'):
        r_squared = np.nanmean(r_squared_vec)
    elif(aggr=='median'):
        r_squared = np.nanmedian(r_squared_vec)
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    return(r_squared)
    
def compute_R_squared(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>0.3]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>0.3]
    
    corr_matrix = np.corrcoef(flattened_true,flattened_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    
    #if(aggr=='mean'):
    #    r_squared = np.nanmean(r_squared_vec)
    #elif(aggr=='median'):
    #    r_squared = np.nanmedian(r_squared_vec)
    #else:
    #    print("invalid aggregation method: " + str(aggr))
    #    asdf
    return(R_sq)
    
def compute_SSIM(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()

    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>0.3]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>0.3]
    
    try:
        #(s,d) = compare_ssim(flattened_true,flattened_pred,full=True)
        (s,d) = structural_similarity(flattened_true,flattened_pred,full=True)
    except:
        s = np.nan
    return(s)

def compute_SSIM_old(true,pred,mask_2d,aggr='mean'):
    # Using https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((mask_2d.shape[1],mask_2d.shape[2],true.shape[2])) # mask i0 is band
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[np.isnan(flattened_mask)]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[np.isnan(flattened_mask)]
    
    true = flattened_true.astype(np.float64)
    pred = flattened_pred.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(true, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(true**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(true * pred, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return(np.nanmean(ssim_map))
    
    
def compute_MAPE(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
    
    diff = true-pred
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>0.3]
    flattened_true = flattened_true[flattened_mask>0.3]
    
    if(aggr=='mean'):
        mape = np.nanmean(np.absolute(flattened_diff / flattened_true))
    elif(aggr=='median'):
        mape = np.nanmedian(np.absolute(flattened_diff / flattened_true))
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf

    return(mape)
    
def SAM_pixel(s1,s2):
    # Using code from https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html#SAM
    # TODO: check redistribution license
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        return(0.0)
    return(angle)

def SAM_grid(target,pred,gridded=False,only_cloud=False,cloud_mask=None,aggr='mean'):
    sam_grid = np.zeros((pred.shape[0],pred.shape[1]))
    for i in range(0,pred.shape[0]):
        for j in range(0,pred.shape[1]):
            if(only_cloud):
                if(np.isnan(cloud_mask[i,j])):
                    sam_grid[i,j] = SAM_pixel(pred[i,j,:],target[i,j,:])
                else:
                    sam_grid[i,j] = np.nan # Use nanmean later to filter non-cloudy pixels
            else:
                sam_grid[i,j] = SAM_pixel(pred[i,j,:],target[i,j,:])
    if(gridded):
        return(sam_grid)
    else:
        return(np.nanmean(sam_grid))
        
def compute_NDVI(true,pred,mask_2d,aggr='mean'):
    #mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    #for b in range(0,true.shape[2]):
    #    mask[:,:,b] = mask_2d.copy()
   
    diff_true = true[:,:,7]-true[:,:,3]
    sum_true = true[:,:,7]+true[:,:,3]
    div_true = diff_true / sum_true
    flattened_div_true = div_true.reshape((div_true.shape[0]*div_true.shape[1]))
    
    diff_pred = pred[:,:,7]-pred[:,:,3]
    sum_pred = pred[:,:,7]+pred[:,:,3]
    div_pred = diff_pred / sum_pred
    flattened_div_pred = div_pred.reshape((div_pred.shape[0]*div_pred.shape[1]))

    flattened_mask = mask_2d.copy().reshape((mask_2d.shape[0]*mask_2d.shape[1]))#*mask.shape[2]))

    flattened_div_true = flattened_div_true[flattened_mask>0.3]
    flattened_div_pred = flattened_div_pred[flattened_mask>0.3]
   
    if(aggr=='mean'):
        error = np.nanmean(np.absolute(flattened_div_true-flattened_div_pred))
    elif(aggr=='median'):
        error = np.nanmedian(np.absolute(flattened_div_true-flattened_div_pred))
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf

    return(error)


# Setup

if(len(sys.argv) != 7):
    print("Usage: python compute_performance.py [result file path] [original scene data path] [save path] [scene name] [method] [cloud proportion mode]")

results_path = sys.argv[1]
original_path = sys.argv[2]
save_path = sys.argv[3]
scene = sys.argv[4]
method = sys.argv[5]
clprop = {
    "1":(0.1,0.3),
    "2":(0.3,0.7),
    "3":(0.7,0.9)
}
cloud_prop_min = clprop[sys.argv[6]][0]
cloud_prop_max = clprop[sys.argv[6]][1]


log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.ERROR)

aggregation_method = "mean"

overwrite = True

if(aggregation_method == "mean"):
    currdir = save_path + "performance_measures"
else:
    currdir = save_path + "performance_measures_" + aggregation_method
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
        
# List of invalid patches
fp = open("scene_patches.pkl","rb")
scene_patches = pickle.load(fp)
fp.close()

performance_metrics = ["MAPE","MAE","MSE","Rsquared","SSIM","PSNR","SAM","NDVI"]
#performance_metrics = ["MAE"] # TODO: remove

feature_sets = ["msi_1w","msi_1m","msi_6m","msi_12m"]

for metric in performance_metrics:

    savedir = currdir + "/" + method + "_" + metric + "_" + str(cloud_prop_min) + "_" + str(cloud_prop_max) + ".csv"
    
    existing_runs = []

    # Initialise empty results file if it doesn't exist (or if we are
    # overwriting existing results), get a list of existing results otherwise
    if(not(os.path.exists(savedir)) or overwrite):
        with open(savedir,'w') as fp:
            fp.write("patch,msi_1w,msi_1m,msi_6m,msi_12m\n")
    #else:
    #    df = pd.read_csv(savedir)
    #    existing_runs = list(df['patch'].values)
            
    resdir = results_path + "results/" + scene + "/" + method + "_" + str(cloud_prop_min) + "_" + str(cloud_prop_max) 
    
    if(os.path.exists(resdir)): # This is bad but the results are already in, this is just ensembling prelim
        for patchname in os.listdir(resdir):
            #if(not(patchname in existing_runs)):
            patchdir = resdir + "/" + patchname + "/"
            ground_truth_path = original_path + "scenes/" + scene + "/" + patchname + "_msi_target.tif"
            mask_path = original_path + "scenes/" + scene + "/" + patchname + "_mask.tif"
            target_key = scene + "_msi_target"
            invalid_patches = scene_patches[target_key]
            if(patchname + "_msi_target" not in invalid_patches):
                patch_results = compute_measures(patchdir,ground_truth_path,mask_path,metric,scene_patches,aggregation_method=aggregation_method)
                s = patchname + "," + str(patch_results[0]) + "," + str(patch_results[1]) + "," + str(patch_results[2]) + "," + str(patch_results[3]) + "\n"
                    
                with open(savedir,'a') as fp:
                    fp.write(s)


        
            
            
print("Terminated successfully")