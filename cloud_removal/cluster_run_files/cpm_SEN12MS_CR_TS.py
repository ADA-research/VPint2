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



def normalise_and_visualise(img, title="", rgb=[3,2,1], percentile_clip=True, save_fig=False, save_path="", **kwargs):
    import matplotlib.pyplot as plt
    new_img = np.zeros((img.shape[0],img.shape[1],3))
    new_img[:,:,0] = img[:,:,rgb[0]]
    new_img[:,:,1] = img[:,:,rgb[1]]
    new_img[:,:,2] = img[:,:,rgb[2]]
    
    if(percentile_clip):
        min_val = np.nanpercentile(new_img, 1)
        max_val = np.nanpercentile(new_img, 99)

        new_img = np.clip((new_img-min_val) / (max_val-min_val), 0, 1)
    
    plt.imshow(new_img, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if(save_fig):
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def compute_measure(true_vals, pred_vals, measure):

    if(measure == "MAE"):
        m = compute_MAE(true_vals, pred_vals)
    elif(measure == "MSE"):
        m = compute_MSE(true_vals, pred_vals)
    elif(measure == "RMSE"):
        m = compute_RMSE(true_vals, pred_vals)
    elif(measure == "PSNR"):
        m = compute_PSNR(true_vals, pred_vals)
    elif(measure == "SSIM"):
        m = compute_SSIM(true_vals, pred_vals)
    elif(measure == "Rsquared"):
        m = compute_R_squared(true_vals, pred_vals)
    elif(measure == "SAM"):
        m = compute_SAM(true_vals, pred_vals)
    elif(measure == "MAPE"):
        m = compute_MAPE(true_vals, pred_vals)
    elif(measure == "NDVI"):
        m = compute_NDVI(true_vals, pred_vals)
    else:
        print("Non-supported measure: " + str(measure))

            
    return(m)

    
def compute_MAE(true, pred):
    diff = np.absolute(true-pred)
    return(np.nanmean(diff))

    
def compute_MSE(true, pred):
    diff = (true-pred)**2
    return(np.nanmean(diff))
    
 

def compute_PSNR(true, pred):
    # Based on https://github.com/PatrickTUM/UnCRtainTS/blob/main/model/src/learning/metrics.py
    rmse = np.sqrt(np.mean((true - pred)**2))
    psnr = 20 * log10(1 / rmse)
    return(psnr)

 
def compute_PSNR_old(true, pred):
    # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    mse = np.nanmean((true - pred) ** 2) + 0.001
    
    if(mse == 0):
        return(1)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) / 100 # /100 because I want 0-1
    return(psnr)

    
def compute_RMSE(true, pred):
    mse = np.mean((true - pred)**2)
    rmse = math.sqrt(mse)
    return(rmse)
    
    
def compute_SAM(true, pred):
    # Based on https://github.com/PatrickTUM/UnCRtainTS/blob/main/model/src/learning/metrics.py
    mat = true * pred
    mat = np.sum(mat, axis=2) # Use axis=2 to operate on bands
    mat = mat / np.sqrt(np.sum(true * true, axis=2))
    mat = mat / np.sqrt(np.sum(pred * pred, axis=2))
    sam = np.mean(np.arccos(np.clip(mat, a_min=-1, a_max=1))*180/np.pi)
    return(sam)
    
    
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
    
def compute_SSIM(true, pred):   
    try:
        #(s,d) = compare_ssim(flattened_true,flattened_pred,full=True)
        (s, d) = structural_similarity(true, pred, full=True)
    except:
        s = np.nan
    return(s)
    
    
def compute_MAPE(true, pred):
    diff = np.absolute(true-pred)
    mape = np.nanmean(diff / true)
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
        
def compute_NDVI(true, pred):

    diff_true = true[:,:,7] - true[:,:,3]
    sum_true = true[:,:,7] + true[:,:,3]
    div_true = diff_true / sum_true
    
    diff_pred = pred[:,:,7] - pred[:,:,3]
    sum_pred = pred[:,:,7] + pred[:,:,3]
    div_pred = diff_pred / sum_pred

    error = np.nanmean(np.absolute(div_true - div_pred))
    return(error)


# Setup

result_path = sys.argv[1] # Path of the .npy results
dataset_path = sys.argv[2] # Path of the original dataset
save_path = sys.argv[3] # Path to save the results


# Conditions
#conditions_measures = ["MAE", "SSIM", "NDVI", "MAPE"]
conditions_measures = ["RMSE", "SAM", "PSNR", "SSIM"]
conditions_algorithms = [
    "VPint",
    "VPint_multi",
]

overwrite = True # Set to True to overwrite existing runs
size_y = 256
size_x = 256

conditions_rois = {
    "s2_africa_test": ["ROIs2017"],
    "s2_america_test": ["ROIs1158", "ROIs1970"],
    "s2_asiaEast_test": ["ROIs1868", "ROIs1970"],
    "s2_asiaWest_test": ["ROIs1868"],
    "s2_europa_test": ["ROIs1868", "ROIs1970", "ROIs2017"],
}


# Assigning conditions to this run ID
i = 1
conds = {}

for geo, roi_list in conditions_rois.items():
    for roi in roi_list:
        for alg in conditions_algorithms:
            for m in conditions_measures:
                conds[i] = {"algorithm":alg, "roi":roi, "geo":geo, "measure":m}
                i += 1
            
this_run_cond = conds[int(sys.argv[4])]
            
            
            
# Create result dir
currdir = save_path + "performance_measures_SEN12MS_CR_TS"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    

# Create directory per geo
currdir = currdir + "/" + this_run_cond["geo"]
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass


# Create directory per roi
currdir = currdir + "/" + this_run_cond["roi"]
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
        
                
# Prepare patches etc for this job (this geo+roi combination)

roi_path = dataset_path + this_run_cond["geo"] + "/" + this_run_cond["roi"] + "/" 
roi_path_res = result_path + this_run_cond["algorithm"] + "/" + this_run_cond["geo"] + "/" + this_run_cond["roi"] + "/" 


# Not actually sure what these are, but some rois have more dirs inside than just 100, so need to include
roi_nums = os.listdir(roi_path)
for roi_num in roi_nums:

    # Create directory per roi_num
    save_path = currdir + "/" + str(roi_num)
    if(not(os.path.exists(save_path))):
        try:
            os.mkdir(save_path)
        except:
            pass
            
    # Initialise result csv
    save_path = save_path + "/" + this_run_cond["algorithm"] + "_" + this_run_cond["measure"] + ".csv"
    if(overwrite or not(os.path.exists(save_path))):
        s = "patch," + this_run_cond["measure"] + "\n"
        with open(save_path, 'w') as fp:
            fp.write(s)
    
    # Gather result patches
    roi_path2 = roi_path + str(roi_num) + "/S2/"
    roi_path2_res = roi_path_res + str(roi_num)
    #patches = [a.split("_")[-1].split(".")[0] for a in os.listdir(roi_path2 + "0/")]
    #res_patches = [a.split(".")[0] for a in os.listdir(roi_path2_res)]
    
    # Load csv containing ground truth indices (time steps)
    exp_dir = "experiments_meta/"
    roi_key = this_run_cond["geo"] + "_" + this_run_cond["roi"] + "_" + str(roi_num)
    results_gt_path = exp_dir + "results_gt_" + roi_key + ".csv"
    df = pd.read_csv(results_gt_path, header=0)
    patches = df['patch'].values
    patches = [str(s) for s in patches]
    tss = df['index'].values
    results_gt_ts = {patches[i]:tss[i] for i in range(0, len(patches))}
    


    # Iterate over patches
    for patch in patches:
        # Load reconstruction
        res_data = np.load(roi_path2_res + "/" + patch + ".npy")
        # Load ground truth
        ts = str(results_gt_ts[patch])
        ts_path = roi_path2 + ts + "/"
        img_date = os.listdir(roi_path2 + ts + "/")[0].split("_")[5]
        img_path = ts_path + "s2_" + this_run_cond["roi"] + "_" + str(roi_num) + "_ImgNo_" + ts + "_" + img_date + "_patch_" + patch + ".tif"
        with rasterio.open(img_path) as fp:
            gt_data = np.moveaxis(fp.read(), 0, -1)
            
            
        # debug
        #normalise_and_visualise(gt_data, title="gt", save_fig=True, save_path="gt.pdf")
        #normalise_and_visualise(res_data, title="res", save_fig=True, save_path="res.pdf")
            
        measure = compute_measure(gt_data/10000, res_data/10000, measure=this_run_cond["measure"])
        s = patch + "," + str(measure) + "\n"
        with open(save_path, 'a') as fp:
            fp.write(s)
            
        #print(this_run_cond)
        #asdkljf
        
        
            
print("Terminated successfully")