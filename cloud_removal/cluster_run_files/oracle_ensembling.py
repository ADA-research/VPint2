import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import os
import sys
import pickle
import math

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


# Setup

use_VPint = sys.argv[2]

results_path = "/scratch/arp/results/cloud/results_base/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"

conditions_features = ["1w", "1m", "3m", "6m"]

conditions_algorithms_VPint = [
    "VPint", 
    #"VPint_no_IP",
    #"VPint_no_EB", 
    "replacement",
    "regression",
    "regression_band",
    "NSPI",
]

conditions_algorithms_noVPint = [
    "replacement",
    "regression",
    "regression_band",
    "NSPI",
]



if(use_VPint=="True"):
    ensemble_name = "OracleEnsembleVPint"
    conditions_algorithms = conditions_algorithms_VPint    
else:
    ensemble_name = "OracleEnsembleNoVPint"
    conditions_algorithms = conditions_algorithms_noVPint    
    
with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
conditions_scenes = list(legend.keys())

conditions_features = ["1w", "1m", "3m", "6m"]
replace = True
y_size = 256
x_size = 256

i = 1
conds = {}

for scene in conditions_scenes:
    for feature in conditions_features:
        conds[i] = {"features":feature, "algorithm":"VPint", "scene":scene}
        i += 1
            
this_run_cond = conds[int(sys.argv[1])]


# Ensembling specific setup

estimated_num_patches = len(conditions_scenes) * 400



enc = {}
for i in range(0,len(conditions_algorithms)):
    enc[conditions_algorithms[i]] = i
    
#dec = {list(v):k for k,v in enc.items()}
    

#########################################
# Begin proper code
#########################################



scene = this_run_cond['scene']
method = conditions_algorithms[0] # Just reference for os.listdir, real thing added later
feature = this_run_cond['features']

currdir = results_path + scene + "/" + method + "/"
for patchname in os.listdir(currdir):
    currdir1 = currdir + patchname + "/"
    y_offset = int(patchname.split("_")[0][1:])
    x_offset = int(patchname.split("_")[1][1:])

    # Save path, do first to check for existence
    savedir = results_path + scene + "/"
    if(not(os.path.exists(savedir))):
        try:
            os.mkdir(savedir)
        except:
            pass
            
    savedir = savedir + ensemble_name + "/"
    if(not(os.path.exists(savedir))):
        try:
            os.mkdir(savedir)
        except:
            pass
            
    savedir = savedir + patchname + "/"
    if(not(os.path.exists(savedir))):
        try:
            os.mkdir(savedir)
        except:
            pass
    
    fn = savedir + feature + ".npy"
    if(replace or not(os.path.exists(fn))):
        try:

            filenames = []
            filename = currdir1 + feature + ".npy"
            for method_real in conditions_algorithms: # True method iteration, bit awkward
                s = filename.replace(method, method_real)
                if(method_real == 'NSPI'): # Super hacky because I saved new results in new dir
                    s = s.replace('results_base', 'results')
                filenames.append(s)

            
            # Load input, feature and mask
            key_target = scene + "-target"
            key_feature = scene + "-" + feature
            key_mask = scene + "-mask"
            
            target_path = original_path + legend[scene]['target'] + ".zip"
            feature_path = original_path + legend[scene][feature] + ".zip"
            mask_path = original_path + legend[scene]['mask'] + ".zip"

            target = load_product_windowed(target_path, y_size, x_size, y_offset, x_offset).astype(float)
            mask_grid = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
            
            # Load pred grids
            pred_grids = []
            for fp in filenames:
                if(os.path.exists(fp)):
                    pred_grid = np.load(fp)
                else:
                    pred_grid = np.zeros_like(target)
                pred_grids.append(pred_grid)
            
            # Apply mask for f0, rest doesn't need it
            
            shp = target.shape
            size = np.product(shp)
            
            mask = np.zeros(shp)
            for b in range(0,mask.shape[2]):
                mask[:,:,b] = mask_grid.copy()
            mask = mask.reshape(size)
            
            input_grid = target.copy()
            input_grid_flat = input_grid.reshape(size)
            input_grid_flat = input_grid_flat[mask>20]
            
            # Pre-compute means
            f0 = np.nanmean(input_grid_flat)
            #f1 = np.nanmean(feature_grid)
            
            # Initialise ensemble pred grid
            ensemble_pred_grid = np.zeros(input_grid.shape)
            
            # Iterate over pixels
            for i in range(0,input_grid.shape[0]):
                for j in range(0,input_grid.shape[1]):
                    for b in range(0,input_grid.shape[2]):
            
                        # Replace non-cloudy with input, cloudy with predicted method pred
                        if(mask_grid[i,j] > 20):
                            feature_vec = np.zeros(len(conditions_algorithms))
                            
                            for p_i in range(0,len(pred_grids)):
                                offset = p_i
                                feature_vec[offset] = pred_grids[p_i][i,j,b]
                                
                            errs = np.absolute(np.ones(len(feature_vec))*target[i,j,b] - feature_vec)    
                            
                            ind = np.argmin(errs)
                            ensemble_pred_grid[i,j,b] = feature_vec[ind]
                            
                        else:
                            ensemble_pred_grid[i,j,b] = input_grid[i,j,b]
                    
            # Actually save
            np.save(fn, ensemble_pred_grid)
        except Exception as e:
            print("Failure for ", this_run_cond, patchname)
            print(e)
                
print("Finished run for ", this_run_cond)