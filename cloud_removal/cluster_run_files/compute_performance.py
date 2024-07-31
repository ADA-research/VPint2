import numpy as np
import pandas as pd
import scipy.stats

import json

import rasterio
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import cv2
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

import sys
import os
import math
from math import log10, sqrt
import pickle


# Functions

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

def compute_measures(patchdir, ground_truth_path, mask_path, metric, this_run_cond, features=["1w", "1m", "3m", "6m"], filter_values=False):

    measures = np.ones(len(features)) * np.nan
    
    i = 0
    for feature in features:
        scene = this_run_cond['scene']
        patch = patchdir.split("/")[-2]
        
        y_offset = int(patch.split("_")[0][1:])
        x_offset = int(patch.split("_")[1][1:])
        
        feature_key = scene + "_" + feature

        #pd = patchdir + feature + ".json"
        pd = patchdir + feature + ".npy"

        pred_vals = np.load(pd)
        
        true_vals = load_product_windowed(ground_truth_path, 256, 256, y_offset, x_offset).astype(float)
        
        if(pred_vals.shape != true_vals.shape):
            print("Mismatched shapes for ", pd, ", True shape:", true_vals.shape, ", pred shape:", pred_vals.shape)
            fkldjsflkj # TODO: remove
        else:
            
            # Remove edge case nans; should only happen on true (faulty pixels), but
            # do both just in case.
            pred_mean = np.nanmean(pred_vals)
            true_mean = np.nanmean(true_vals)
            pred_vals = np.nan_to_num(pred_vals,nan=pred_mean)
            true_vals = np.nan_to_num(true_vals,nan=true_mean)
                
            mask = load_product_windowed(mask_path, 256, 256, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
            
            if(metric == "MAE"):
                mae = compute_MAE(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = mae
            elif(metric == "MSE"):
                mse = compute_MSE(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = mse
            elif(metric == "PSNR"):
                psnr = compute_PSNR(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = psnr
            elif(metric == "SSIM"):
                ssim = compute_SSIM(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = ssim
            elif(metric == "Rsquared"):
                r_squared = compute_R_squared(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = r_squared
            elif(metric == "SAM"):
                sam = SAM_grid(true_vals,pred_vals,only_cloud=True,cloud_mask=mask)
                measures[i] = sam
            elif(metric == "MAPE"):
                mape = compute_MAPE(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = mape
            elif(metric == "NDVI"):
                ndvi = compute_NDVI(true_vals,pred_vals,mask, filter_values=filter_values)
                measures[i] = ndvi
            else:
                print("Non-supported metric: " + str(metric))
        i += 1
    else:
        i += 1
            
    return(measures)


def filter_confidence(mask, pred):
    # Abuse mask to easily filter for confidence interval in pixels
    
    for b in range(0, pred.shape[2]):
        pred_flat = pred[:,:,0].reshape((pred.shape[0] * pred.shape[1]))
        mask_flat = mask[:,:,0].reshape((mask.shape[0] * mask.shape[1]))
        ci_lower = scene_dists[this_run_cond['scene']][b]['ci'+str(ci_interval)+'_lower']
        ci_upper = scene_dists[this_run_cond['scene']][b]['ci'+str(ci_interval)+'_upper']
        
        mask_flat[pred_flat < ci_lower] = 0 # 0 means no cloud, so do not use in computing performance, slightly hacky
        mask_flat[pred_flat > ci_upper] = 0
        mask_flat = mask_flat.reshape((pred.shape[0], pred.shape[1])) # no longer technically flat
        mask[:,:,b] = mask_flat
                
    return(mask)

    
def compute_MAE(true, pred, mask_2d, aggr='mean', filter_values=False):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)
    
    diff = np.absolute(true-pred)
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>20]

    
    if(aggr=='mean'):
        mae = np.nanmean(flattened_diff)
    elif(aggr=='median'):
        mae = np.nanmedian(flattened_diff)
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    return(mae)
    
    

    
def compute_MSE(true,pred,mask_2d,aggr='mean', filter_values=False):
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)
    
    diff = (true-pred)**2
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>20]
    
    if(aggr=='mean'):
        mse = np.nanmean(flattened_diff)
    elif(aggr=='median'):
        mse = np.nanmedian(flattened_diff)
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf
    return(mse)
    
def compute_PSNR(true,pred,mask_2d,aggr='mean', filter_values=False):
    # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)
    
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>20]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>20]
    
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

    
def compute_R_squared(true,pred,mask_2d,aggr='mean', filter_values=False):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)
    
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>20]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>20]
    
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
    
def compute_SSIM(true,pred,mask_2d,aggr='mean', filter_values=False):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)

    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>20]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>20]
    
    try:
        #(s,d) = compare_ssim(flattened_true,flattened_pred,full=True)
        (s,d) = structural_similarity(flattened_true,flattened_pred,full=True)
    except:
        s = np.nan
    return(s)
    
    
def compute_MAPE(true,pred,mask_2d,aggr='mean', filter_values=False):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    if(filter_values):
        mask = filter_confidence(mask, pred)
    
    
    diff = true-pred
    
    flattened_diff = diff.reshape((diff.shape[0]*diff.shape[1]*diff.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))
    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    
    flattened_diff = flattened_diff[flattened_mask>20]
    flattened_true = flattened_true[flattened_mask>20]
    
    if(aggr=='mean'):
        mape = np.nanmean(np.absolute(flattened_diff / flattened_true))
    elif(aggr=='median'):
        mape = np.nanmedian(np.absolute(flattened_diff / flattened_true))
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf

    return(mape*100)
    
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
        
def compute_NDVI(true,pred,mask_2d,aggr='mean', filter_values=False):
    #mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    #for b in range(0,true.shape[2]):
    #    mask[:,:,b] = mask_2d.copy()
    
    if(filter_values):
        mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
        for b in range(0,true.shape[2]):
            mask[:,:,b] = mask_2d.copy()
        mask = filter_confidence(mask, pred)
        mask_2d = mask[:,:,0]
   
    diff_true = true[:,:,7]-true[:,:,3]
    sum_true = true[:,:,7]+true[:,:,3]
    div_true = diff_true / sum_true
    flattened_div_true = div_true.reshape((div_true.shape[0]*div_true.shape[1]))
    
    diff_pred = pred[:,:,7]-pred[:,:,3]
    sum_pred = pred[:,:,7]+pred[:,:,3]
    div_pred = diff_pred / sum_pred
    flattened_div_pred = div_pred.reshape((div_pred.shape[0]*div_pred.shape[1]))

    flattened_mask = mask_2d.copy().reshape((mask_2d.shape[0]*mask_2d.shape[1]))#*mask.shape[2]))

    flattened_div_true = flattened_div_true[flattened_mask>20]
    flattened_div_pred = flattened_div_pred[flattened_mask>20]
   
    if(aggr=='mean'):
        error = np.nanmean(np.absolute(flattened_div_true-flattened_div_pred))
    elif(aggr=='median'):
        error = np.nanmedian(np.absolute(flattened_div_true-flattened_div_pred))
    else:
        print("invalid aggregation method: " + str(aggr))
        asdf

    return(error)


# Setup

if(len(sys.argv) != 6):
    print("Usage: python compute_performance.py [result file path] [original scene data path] [save path] [filter off/on] [run id]")

results_path = sys.argv[1] + "results_base/" # Path of the .npy results
original_path = sys.argv[2] # Path of the original dataset
save_path = sys.argv[3] # Path to save the results
ci_filter = sys.argv[4]
if(ci_filter == "True"):
    ci_filter = True
else:
    ci_filter = False

# Conditions
conditions_pms = ["MAPE", "MAE", "MSE", "Rsquared", "SSIM", "PSNR", "SAM", "NDVI"]
conditions_pms = ["MAE", "SSIM", "NDVI", "MAPE"]
conditions_features = ["1w", "1m", "3m", "6m"]
conditions_algorithms = [
    #"VPint",
    #"VPint_no_IP",
    #"VPint_no_EB",
    #"replacement",
    #"regression",
    #"regression_band",
    #"regression_RF",
    "OracleEnsembleVPint",
    "OracleEnsembleNoVPint",
    "PatchOracleEnsembleVPint",
    "PatchOracleEnsembleNoVPint",
    #"NSPI",
    #"WLR",
]
conditions_scenes = list(os.listdir(results_path))


# Assigning conditions to this run ID
i = 1
conds = {}

for scene in conditions_scenes:
    for feature in conditions_features:
        for alg in conditions_algorithms:
            for pm in conditions_pms:
                conds[i] = {"features":feature, "algorithm":alg, "scene":scene, "pm":pm}
                i += 1
                
            
global this_run_cond # ugly but convenient
this_run_cond = conds[int(sys.argv[5])]

# Random logging stuff
log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.ERROR)

# Load the legend to keep track of which files belong to which scenes
with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
# Filter out misaligned conditions. Can probably be fixed with geo-coords based data loading, but haven't found a way yet to do with with resampling and windowed reading
manual_QA = True
if(manual_QA):
    dset_size = len(legend)*len(conditions_features)
    print("Legend size prior to QA:", dset_size)
    # All misaligned: drop all
    del legend["africa_forest_zambia"]
    del legend["america_forest_quebec"]
    del legend["america_urban_phoenix"]
    dset_size = dset_size - 2*len(conditions_features)
    # Only targed misaligned: drop target, use 1w as target
    legend["africa_herbaceous_southafrica"]['target'] = legend["africa_herbaceous_southafrica"]['1w']
    del legend["africa_herbaceous_southafrica"]['1w']
    legend["asia_cropland_china"]['target'] = legend["asia_cropland_china"]['1w']
    del legend["asia_cropland_china"]['1w']
    legend["australia_shrubs_west"]['target'] = legend["australia_shrubs_west"]['1w']
    del legend["australia_shrubs_west"]['1w']
    legend["europe_cropland_hungary"]['target'] = legend["europe_cropland_hungary"]['1w']
    del legend["europe_cropland_hungary"]['1w']
    dset_size = dset_size - 4
    # Misaligned specific feature sets
    del legend["africa_herbaceous_southafrica"]['6m']
    del legend["america_urban_atlanta"]['1m']
    del legend["america_urban_atlanta"]['3m']
    del legend["asia_cropland_india"]['1w']
    del legend["asia_cropland_india"]['1m']
    del legend["asia_urban_beijing"]['1w']
    del legend["asia_urban_beijing"]['6m']
    del legend["europe_cropland_hungary"]['1m']
    dset_size = dset_size - 8
    print("Legend size after QA:", dset_size)

# Overwrite existing results where relevant
overwrite = True

# Filter most extreme pixel values
overwrite_dist = False
if(ci_filter):
    global ci_interval
    global scene_dists # ugly but convenient

    ci_interval = 95
    # Overwrite if this interval is not already saved
    ci_key_lower = 'ci' + str(ci_interval) + '_lower'
    ci_key_upper = 'ci' + str(ci_interval) + '_upper'
    if(os.path.exists("scene_distributions.pkl")):
        with open("scene_distributions.pkl", 'rb') as fp:
            scene_dists = pickle.load(fp)
            for _,s in scene_dists.items():
                a = s[0].keys()
                if(not(ci_key_lower in a)):
                    overwrite_dist = True
                break

    # Precompute distributions
    if(overwrite_dist or not(os.path.exists("scene_distributions.pkl"))):
        scene_dists = {}
        for scene in conditions_scenes:

            ref_product_path = original_path + legend[scene]['target'] + ".zip"
            prod = load_product(ref_product_path)
            
            band_dists = {}
            for b in range(0, prod.shape[2]):
                b_mean = np.nanmean(prod[:,:,b])
                b_median = np.nanmedian(prod[:,:,b])
                b_std = np.nanmedian(prod[:,:,b])
                b_interval = scipy.stats.norm.interval(ci_interval/100, loc=b_mean, scale=b_std)
                band_dists[b] = {"mean":b_mean, "median":b_median, "std":b_std, ci_key_lower:b_interval[0], ci_key_upper:b_interval[1]}
                
            scene_dists[scene] = band_dists
        with open("scene_distributions.pkl", 'wb') as fp:
            pickle.dump(scene_dists, fp)
    else:
        with open("scene_distributions.pkl", 'rb') as fp:
            scene_dists = pickle.load(fp)
        
    

# Create save directory
if(ci_filter):
    currdir = save_path + "performance_measures_filtered" + str(ci_interval)
else:
    currdir = save_path + "performance_measures"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
# Create directory for this scene
currdir = currdir + "/" + this_run_cond['scene']
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
# Create key (path) for this run's results    
key = currdir + "/" + this_run_cond['algorithm'] + "_" + this_run_cond["pm"] + ".csv"

# Initialise empty list for existing runs; do this to skip unnecessary patches
existing_runs = []

# Initialise empty results file if it doesn't exist (or if we are
# overwriting existing results), get a list of existing results otherwise
if(not(os.path.exists(key)) or overwrite):
    header = "patch"
    for f in conditions_features:
        header = header + "," + f
    header = header + "\n"
    with open(key, 'w') as fp:
        fp.write(header)

        
# Start iterating over results, first check legend
if(this_run_cond['scene'] in legend):
    if(this_run_cond['features'] in legend[this_run_cond['scene']]):
        
        # Path for the result directory for these conditions
        resdir = results_path + this_run_cond['scene'] + "/" + this_run_cond['algorithm']

        # Iterate over patches in the result directory for this run
        if(os.path.exists(resdir)): # This is bad but the results are already in, this is just ensembling prelim
            for patchname in os.listdir(resdir):
                #if(not(patchname in existing_runs)):
                patchdir = resdir + "/" + patchname + "/"
                
                # Collect appropriate paths for original dataset
                ground_truth_path = original_path + legend[this_run_cond['scene']]['target'] + ".zip"
                mask_path = original_path + legend[this_run_cond['scene']]['mask'] + ".zip"
                # Compute the actual measures
                try:
                    patch_results = compute_measures(patchdir, ground_truth_path, mask_path, this_run_cond['pm'], this_run_cond, filter_values=ci_filter)
                    s = patchname
                    for result in patch_results:
                        s = s + "," + str(result)
                    s = s + "\n"
                        
                    with open(key, 'a') as fp:
                        fp.write(s)
                        
                except Exception as e:
                    print(e)

                
            
            
print("Terminated successfully")