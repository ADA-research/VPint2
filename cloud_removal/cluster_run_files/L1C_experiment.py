# Imports

import numpy as np
import pandas as pd

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

#import tensorflow as tf
#from tensorflow.keras import Model, Input, regularizers
#from tensorflow.keras import layers, models
#from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from VPint.WP_MRP import WP_SMRP

import os
import sys
import random
import pickle

# Helper functions
  

# Main functions

def VPint_interpolation(target_grid,feature_grid,method="exact", use_IP=True, use_EB=True):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        #MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b,:],LinearRegression(),max_gamma=5,min_gamma=0.1)
        MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b])
        mu = np.nanmean(target_grid[:,:,b]) + 2*np.nanstd(target_grid[:,:,b])
        if(method=='exact'):
            pred_grid[:,:,b] = MRP.run(method='exact',
                           auto_adapt=True, auto_adaptation_verbose=False,
                           auto_adaptation_epochs=50, auto_adaptation_max_iter=100,
                           auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, 
                           resistance=use_EB,prioritise_identity=use_IP)
        else:
            MRP.train()
            pred_grid[:,:,b] = MRP.run(method=method)
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

def load_product_windowed_sar(path, y_size, x_size, y_offset, x_offset):
    # TODO: implement
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
                       cloud_threshold=5, method="exact", buffer_mask=True, mask_buffer_size=10,mask_buffer_passes=10, algorithm="VPint"):
                       
    key_target = scene + "-l1c"
    key_feature = scene + "-" + feature_name
    key_mask = scene + "-mask"

    target_path = base_path + legend[scene]['l1c'] + ".zip"# need to account for the .SAFE.zip stuff
    if(not(os.path.exists(target_path))):
        target_path = base_path + legend[scene]['l1c'] + ".SAFE.zip"
        if(not(os.path.exists(target_path))):
            print("Still didn't find L1C path: ", target_path)
            sdlkfj
    feature_path = base_path + legend[scene][feature_name] + ".zip" 
    sar_path = base_path + legend[scene]['sar'] + ".zip" 
    mask_path = base_path + legend[scene]['mask'] + ".zip"
    
    # Load target and mask first, just return target if no cloudy pixels in mask
    target = load_product_windowed(target_path, y_size, x_size, y_offset, x_offset).astype(float)
    mask = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
    
    if(not(np.any(mask > cloud_threshold))):
        return(target)
    
    # If there are any cloudy pixels, load features and run algorithms
    features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset).astype(float)
    features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))
    
   
    if(buffer_mask):
        mask_grid = mask_buffer(mask, window_size=mask_buffer_size, passes=mask_buffer_passes)
    
  
    target_cloudy = target.copy()

    for i in range(0, target_cloudy.shape[0]):
        for j in range(0, target_cloudy.shape[1]):
            if(mask[i,j] > cloud_threshold):
                a = np.ones(target_cloudy.shape[2]) * np.nan
                target_cloudy[i,j,:] = a
                    
                    
    
    features = np.nan_to_num(features, nan=np.nanmean(features))
    
    if(algorithm=="VPint"):
        pred_grid = VPint_interpolation(target_cloudy, features, method=method, use_IP=True, use_EB=True)

    else:
        pred_grid = None
        print("Warning: invalid method")
        asdlfkjf
    
    return(pred_grid)
    
def mask_buffer(mask, window_size=5, passes=5):
    for p in range(0, passes):
        new_mask = mask.copy()
        for i in range(0, mask.shape[0]):
            for j in range(0, mask.shape[1]):
                if(np.isnan(mask[i,j])):
                    i_min_local = max(0, i-window_size)
                    i_max_local = min(mask.shape[0], i+window_size)
                    j_min_local = max(0, j-window_size)
                    j_max_local = min(mask.shape[1], j+window_size)
                    new_mask[i_min_local:i_max_local, j_min_local:j_max_local] = np.nan
        mask = new_mask
    return(mask)
    



# Setup

if(len(sys.argv) != 4):
    print("Usage: python SEN2-MSI-T-general.py [dataset base path] [result base path] [task number]")

conditions_algorithms = ["VPint", ]

save_path = sys.argv[2]
base_path = sys.argv[1]

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
conditions_scenes = list(legend.keys())
conditions_scenes = [s for s in conditions_scenes if 'sar' in legend[s]] # Only run on scenes where SAR was available

i = 1
conds = {}

for scene in conditions_scenes:
    for alg in conditions_algorithms:
        conds[i] = {"features":"1m", "algorithm":alg, "scene":scene}
        i += 1
            
this_run_cond = conds[int(sys.argv[3])]



# Some parameters

replace = False # Set to True to overwrite existing runs
size_y = 256
size_x = 256



currdir = save_path + "results_l1c"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    

        
np.set_printoptions(threshold=np.inf)

log = rasterio.logging.getLogger()
log.setLevel(rasterio.logging.FATAL)

# Run
        
# Create directory per scene
currdir1 = currdir + "/" + this_run_cond["scene"]
if(not(os.path.exists(currdir1))):
    try:
        os.mkdir(currdir1)
    except:
        pass

# Create directory per algorithm
currdir1 = currdir1 + "/" + this_run_cond["algorithm"]
if(not(os.path.exists(currdir1))):
    try:
        os.mkdir(currdir1)
    except:
        pass

# Check dims, iterate patches
scene_height = -1
scene_width = -1
ref_product_path = base_path + legend[scene]['target'] + ".zip"
with rasterio.open(ref_product_path) as raw_product:
    product = raw_product.subdatasets
with rasterio.open(product[1]) as fp:
    scene_height = fp.height
    scene_width = fp.width

max_row = int(str(scene_height / size_y).split(".")[0])
max_col = int(str(scene_width / size_x).split(".")[0])

# Shuffle indices to allow multiple tasks to run
row_list = np.arange(max_row)
col_list = np.arange(max_col)
np.random.shuffle(row_list)
np.random.shuffle(col_list)

#for y_offset in range(0, max_row):
#    for x_offset in range(0, max_col):

# Iterate
for y_offset in row_list:
    for x_offset in col_list:

        # Create directory for patch
        
        patch_name = "r" + str(y_offset) + "_c" + str(x_offset)

        path = currdir1 + "/" + patch_name
        if(not(os.path.exists(path))):
            try:
                os.mkdir(path)
            except:
                pass
                
        # Path for save file
        path = path + "/" + this_run_cond["features"] + ".npy"
        
        # Run reconstruction
        if(replace or not(os.path.exists(path))):
            #print("Started running for ", path)
            try:
                pred_grid = run_patch(base_path, legend, this_run_cond["scene"], size_y, size_x, y_offset, x_offset, feature_name=this_run_cond["features"], plot=False, algorithm=this_run_cond["algorithm"], buffer_mask=True, mask_buffer_size=5, mask_buffer_passes=5)
            
                if(type(pred_grid) != type(False)): # just because if(pred_grid) is ambiguous
                    np.save(path, pred_grid)
                    
                print("Successfully ran VPint up to point")
                sldkfj
                    
            except:
                print("Failed run for ", this_run_cond, y_offset, x_offset)
                sdlkfj # TODO: remove
                
            #print("Finished running for ", path)
        
print("Terminated successfully")