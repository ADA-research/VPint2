import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def normalise_and_visualise(img, title="", rgb=[3,2,1], percentile_clip=True, save_fig=False, save_path=""):
    
    new_img = np.zeros((img.shape[0], img.shape[1],3))
    new_img[:,:,0] = img[:,:,rgb[0]]
    new_img[:,:,1] = img[:,:,rgb[1]]
    new_img[:,:,2] = img[:,:,rgb[2]]
    
    if(percentile_clip):
        min_val = np.nanpercentile(new_img, 1)
        max_val = np.nanpercentile(new_img, 99)

        new_img = np.clip((new_img-min_val) / (max_val-min_val), 0, 1)
    
    plt.imshow(new_img,interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if(save_fig):
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    

def load_product_windowed(path, y_size, x_size, y_offset, x_offset, target_res=20, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
                        bands_10m={"B2":1, "B3":2, "B4":3, "B8":7},
                        bands_20m={"B5":4, "B6":5, "B7":6, "B8A":8, "B11":11, "B12":12, "CLD":13},
                        bands_60m={"B1":0, "B9":9, "B10":10},
                        return_bounds=False):

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
                        upscale_factor = 10 / target_res
                    elif(band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 20 / target_res
                    elif(band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 60 / target_res

                    # Output window using target resolution
                    window=Window(x_offset, y_offset, x_size, y_size)

                    # Second window for reading in local resolution
                    res_window = Window(x_offset*x_size/upscale_factor, y_offset*y_size/upscale_factor,
                                        window.width / upscale_factor, window.height / upscale_factor)
                          
                    if(return_bounds and band_name in bands_20m):
                        # Compute bounds for data fusion, needlessly computing for multiple bands but shouldn't be a big deal
                        # First indices, then points for xy, then extract coordinates from xy
                        # BL, TR --> (minx, miny), (maxx, maxy)
                        # Take special care with y axis; with xy indices, 0 should be top (coords 0/min is bottom)
                        left = x_offset*x_size/upscale_factor
                        top = y_offset*y_size/upscale_factor
                        right = left + x_size/upscale_factor
                        bottom = top + y_size/upscale_factor
                        tr = rasterio.transform.xy(bandset.transform, left, bottom)
                        bl = rasterio.transform.xy(bandset.transform, right, top)
                        
                        transformer = Transformer.from_crs(bandset.crs, 4326)
                        bl = transformer.transform(bl[0], bl[1])
                        tr = transformer.transform(tr[0], tr[1])

                        left = bl[0]
                        bottom = bl[1]
                        right = tr[0]
                        top = tr[1]
                        bounds = (left, bottom, right, top, bandset.transform, bandset.crs)
                
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

    if(return_bounds):
        return(grid, bounds)
    else:
        return(grid)
    
    
    
def load_product_windowed_withSAR(path, y_size, x_size, y_offset, x_offset, keep_bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dt=np.uint16):

    grid = None
    size_y = -1
    size_x = -1

    with rasterio.open(path) as bandset:

        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(dt)

        b = 0
        for band_index in range(1, len(keep_bands)+1):

            if(band_index in keep_bands):
                upscale_factor = 1

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
                b += 1
    return(grid)
    
    
    
def load_product_windowed_withSAR_old(path, y_size, x_size, y_offset, x_offset, keep_bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):

    grid = None
    size_y = -1
    size_x = -1

    with rasterio.open(path) as bandset:

        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(np.uint16)

        b = 0
        for band_index in range(1, len(keep_bands)+1):

            if(band_index in keep_bands):
                upscale_factor = 1

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
                b += 1
    return(grid)

def compute_measures(patchdir, ground_truth_path, metric, this_run_cond):

    features = ["reconstruction"]
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
        

        if(os.path.exists(pd)):
            pred_vals = np.load(pd)
            #print(np.mean(pred_vals))
            #print(pred_vals.shape)
            #print(pred_vals[100,:,0])
            #if(this_run_cond['algorithm'] == 'uncrtaints'):
            pred_vals = np.moveaxis(pred_vals[0,0,:,:,:], 0, -1) # TODO: remove again, just for UnCRtain-TS results which were saved as (b, t, c, h, w]
            #normalise_and_visualise(pred_vals, save_fig=True, save_path="reconstruction2.pdf")
            #sldkfj

            if(this_run_cond['algorithm'] in ['dsen2cr', 'dsen2cr_nosar', 'uncrtaints']):
                try: # One scene has weird dimensions, need to include this
                    target = load_product_windowed_withSAR(ground_truth_path, 256, 256, y_offset, x_offset).astype(float)
                    sar = target[:,:,13:]
                    true_vals = target[:,:,:13]
                    #pred_vals = pred_vals / 5 # TODO: remove, this is just to make up for a mistake multiplying dsen2cr results by 10000 instead of 2000, code after this run has been fixed though so should remove if doing new runs
                except:
                    return(None)

            else:
                target = load_product_windowed(ground_truth_path, 256, 256, y_offset, x_offset, target_res=10).astype(float)
                true_vals = target[:,:,:]

                  
            # Remove edge case nans; should only happen on true (faulty pixels), but
            # do both just in case.
            pred_mean = np.nanmean(pred_vals)
            true_mean = np.nanmean(true_vals)
            pred_vals = np.nan_to_num(pred_vals,nan=pred_mean)
            true_vals = np.nan_to_num(true_vals,nan=true_mean)
            
            mask = np.ones((true_vals.shape[0], true_vals.shape[1])) * 100 # We are evaluating on all pixels unfortunately
            
            if(metric == "MAE"):
                mae = compute_MAE(true_vals,pred_vals,mask)
                measures[i] = mae
            elif(metric == "MSE"):
                mse = compute_MSE(true_vals,pred_vals,mask)
                measures[i] = mse
            elif(metric == "PSNR"):
                psnr = compute_PSNR(true_vals,pred_vals,mask)
                measures[i] = psnr
            elif(metric == "SSIM"):
                ssim = compute_SSIM(true_vals,pred_vals,mask)
                measures[i] = ssim
            elif(metric == "Rsquared"):
                r_squared = compute_R_squared(true_vals,pred_vals,mask)
                measures[i] = r_squared
            elif(metric == "SAM"):
                sam = SAM_grid(true_vals,pred_vals,only_cloud=True,cloud_mask=mask)
                measures[i] = sam
            elif(metric == "MAPE"):
                mape = compute_MAPE(true_vals,pred_vals,mask)
                measures[i] = mape
            elif(metric == "NDVI"):
                ndvi = compute_NDVI(true_vals,pred_vals,mask)
                measures[i] = ndvi
            else:
                print("Non-supported metric: " + str(metric))
            i += 1
        else:
            i += 1
        
            
    return(measures)

    
def compute_MAE(true, pred, mask_2d, aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
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
    
def compute_MSE(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
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
    
def compute_PSNR(true,pred,mask_2d,aggr='mean'):
    # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()
    
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
    
    flattened_diff = flattened_diff[flattened_mask>20]
    flattened_SStot = flattened_SStot[flattened_mask>20]
    
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
    
def compute_SSIM(true,pred,mask_2d,aggr='mean'):
    #true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2]))
    for b in range(0,true.shape[2]):
        mask[:,:,b] = mask_2d.copy()

    flattened_mask = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))
    flattened_true = true.reshape((true.shape[0]*true.shape[1]*true.shape[2]))[flattened_mask>20]
    flattened_pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2]))[flattened_mask>20]
    
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

if(len(sys.argv) != 5):
    print("Usage: python compute_performance.py [result file path] [original scene data path] [save path] [run id]")

results_path = sys.argv[1] + "results_l1c/" # Path of the .npy results 
#results_path = sys.argv[1] + "results_l1c_VPint/" # Path of the .npy results # TODO: remove VPint part
original_path = sys.argv[2] # Path of the original dataset
save_path = sys.argv[3] # Path to save the results

# Conditions
conditions_pms = ["MAPE", "MAE", "MSE", "Rsquared", "SSIM", "PSNR", "SAM", "NDVI"]
conditions_pms = ["MAE", "SSIM", "NDVI", "MAPE"]
conditions_features = ["reconstruction"]
conditions_algorithms = [
    #"dsen2cr",
    "uncrtaints",
    #"dsen2cr_nosar",
    #"VPint",
]

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)    

manual_QA = True
if(manual_QA):
    # All misaligned: drop all
    del legend["africa_forest_zambia"]
    del legend["america_forest_quebec"]
    del legend["america_urban_phoenix"]
    # Extra misaligned check for l1c-target
    del legend["africa_herbaceous_southafrica"]
    del legend["asia_cropland_china"]
    # Seems like coverage, but easier to remove...
    del legend["asia_herbaceous_mongoliaeast"]
    del legend["asia_urban_beijing"]
    del legend["australia_shrubs_west"]
    # Misaligned specific feature sets; del all 1m ones
    del legend["america_urban_atlanta"]
    del legend["asia_cropland_india"]
    # Only 6 scenes left over...

conditions_scenes = list(legend.keys())
# Filter out scenes where no SAR data was available
conditions_scenes = [s for s in conditions_scenes if not(legend[s]['sar'] == "")]

# Assigning conditions to this run ID
i = 1
conds = {}

for scene in conditions_scenes:
    for feature in conditions_features:
        for alg in conditions_algorithms:
            for pm in conditions_pms:
                conds[i] = {"features":feature, "algorithm":alg, "scene":scene, "pm":pm}
                i += 1
            
this_run_cond = conds[int(sys.argv[4])]


# Random logging stuff
log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.ERROR)

# Load the legend to keep track of which files belong to which scenes
with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)

# Overwrite existing results where relevant
overwrite = True

# Create save directory
currdir = save_path + "performance_measures_l1c"
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
    with open(key, 'w') as fp:
        fp.write("patch,reconstruction\n")
#else:
#    df = pd.read_csv(savedir)
#    existing_runs = list(df['patch'].values)
        
# Path for the result directory for these conditions
resdir = results_path + this_run_cond['scene'] + "/" + this_run_cond['algorithm']

# Iterate over patches in the result directory for this run
if(os.path.exists(resdir)): # This is bad but the results are already in, this is just ensembling prelim
    for patchname in os.listdir(resdir):
        #if(not(patchname in existing_runs)):
        patchdir = resdir + "/" + patchname + "/"
        
        # Collect appropriate paths for original dataset
        ground_truth_path = original_path + "l1c/collocated_" + this_run_cond['scene'] + ".tif"
        
        if(this_run_cond['algorithm'] in ['dsen2cr', 'dsen2cr_nosar', 'uncrtaints']):
            ground_truth_path = original_path + "l1c/collocated_" + this_run_cond['scene'] + ".tif"
        else:
            ground_truth_path = original_path + legend[this_run_cond['scene']]['l1c'] + ".SAFE.zip"
        
        
        # Compute the actual measures
        patch_results = compute_measures(patchdir, ground_truth_path, this_run_cond['pm'], this_run_cond)
        
        if(patch_results is not None):
            s = patchname + "," + str(patch_results[0]) + "\n"
            with open(key, 'a') as fp:
                fp.write(s)


        
            
            
print("Terminated successfully:", this_run_cond)