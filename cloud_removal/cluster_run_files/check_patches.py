# Imports

import numpy as np
import pandas as pd

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import os
import sys
from time import sleep
import random
import pickle


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


def check_patch(base_path, legend, scene, scenes_ci95, scenes_ci99, y_size, x_size, y_offset, x_offset, img, cloud_threshold=20):
    # Assume patch is valid unless found differently later
    valid_coverage = True
    valid_cloud = True
    valid_ci95 = True
    valid_ci99 = True
                       
    key = scene + "-" + img

    path = base_path + legend[scene][img] + ".zip"
       
    # Load image
    patch = load_product_windowed(path, y_size, x_size, y_offset, x_offset, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "CLD"]).astype(float)

    # Check for 0s (incomplete coverage)
    # Not a perfect solution along edges, but should filter some
    window_size = 1
    indices_i, indices_j = np.where(np.sum(patch[:,:,:-1], axis=2) == 0.0)
    for p in range(0, len(indices_i)):
        start_i = max(indices_i[p]-window_size, 0)
        end_i = min(indices_i[p]+window_size, patch.shape[0])
        start_j = max(indices_j[p]-window_size, 0)
        end_j = min(indices_j[p]+window_size, patch.shape[1])
        box = patch[start_i:end_i, start_j:end_j, :-1]
        if(np.sum(box) <= 0.0):
            valid_coverage = False
            break
    
    # Check for clouds (shouldn't be in target+features)
    # Dismissing patches with 10 or more cloudy pixels 
    
    cloudy = patch[:,:,-1] > cloud_threshold
    if(np.any(cloudy)):
        if(np.sum(cloudy) > 10):
            valid_cloud = False
        
        
    if(False): # very hacky but I don't do this anymore
        # Check for 95% CI
        vec_lower = np.zeros(patch.shape[2]-1)
        vec_upper = np.zeros(patch.shape[2]-1)
        for b in range(0, patch.shape[2]-1):
            vec_lower[b] = scenes_ci95[scene][img][b][0]
            vec_upper[b] = scenes_ci95[scene][img][b][1]
            
        lower_diff = patch[:,:,:-1] - vec_lower # should be positive
        avg_lower_diff = np.sum(lower_diff, axis=-1)
        if(np.any(avg_lower_diff < 0)): 
            valid_ci95 = False
        upper_diff = patch[:,:,:-1] - vec_upper # should be negative
        avg_upper_diff = np.sum(upper_diff, axis=-1)
        if(np.any(avg_lower_diff > 0)): 
            valid_ci95 = False
        
        # Check for 99% CI
        vec_lower = np.zeros(patch.shape[2]-1)
        vec_upper = np.zeros(patch.shape[2]-1)
        for b in range(0, patch.shape[2]-1):
            vec_lower[b] = scenes_ci95[scene][img][b][0]
            vec_upper[b] = scenes_ci95[scene][img][b][1]
            
        lower_diff = patch[:,:,:-1] - vec_lower # should be positive
        avg_lower_diff = np.sum(lower_diff, axis=-1)
        if(np.any(avg_lower_diff < 0)): 
            valid_ci95 = False
        upper_diff = patch[:,:,:-1] - vec_upper # should be negative
        avg_upper_diff = np.sum(upper_diff, axis=-1)
        if(np.any(avg_lower_diff > 0)): 
            valid_ci95 = False
    
    return(valid_coverage, valid_cloud, valid_ci95, valid_ci99)



# Setup

if(len(sys.argv) != 2):
    print("Usage: python check_patches.py [dataset base path]")

imgs = ["target", "1w", "1m", "3m", "6m"]


base_path = sys.argv[1]
recompute_percentiles = False

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
scenes = list(legend.keys())

invalid_patches = {} # scene

# Precompute statistics per scene per feature per band for CI filtering

scenes_ci95 = {}
scenes_ci99 = {}
if(recompute_percentiles):
    for scene in scenes:
        scene_ci95 = {}
        scene_ci99 = {}
        for img in imgs:
            path = base_path + legend[scene][img] + ".zip"
            img_vals = load_product(path)
            band_ci95 = {}
            band_ci99 = {}
            for b in range(0, img_vals.shape[-1]):
                percentile_95 = np.percentile(img_vals[:,:,b], (2.5, 97.5))
                percentile_99 = np.percentile(img_vals[:,:,b], (0.5, 99.5))
                band_ci95[b] = percentile_95
                band_ci99[b] = percentile_99
            scene_ci95[img] = band_ci95
            scene_ci99[img] = band_ci99
        scenes_ci95[scene] = scene_ci95
        scenes_ci99[scene] = scene_ci99
    
    with open("ci95_bounds.pkl", 'wb') as fp:
        pickle.dump(scenes_ci95, fp)
        
    with open("ci99_bounds.pkl", 'wb') as fp:
        pickle.dump(scenes_ci99, fp)

else:
    with open("ci95_bounds.pkl", 'rb') as fp:
        scenes_ci95 = pickle.load(fp)
        
    with open("ci99_bounds.pkl", 'rb') as fp:
        scenes_ci99 = pickle.load(fp)

# Some parameters

size_y = 256
size_x = 256

# Iterate
# {"scene":{"feature":{"coverage":["patch1", "patch2"], "cloud":["patch1", "patch2"]}}

for scene in scenes:
    scene_dict = {} # feature
    for img in imgs:
        invalid_dict = {}
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

        row_list = np.arange(max_row)
        col_list = np.arange(max_col)

        invalid_list_coverage = [] # patch1, patch2
        invalid_list_cloud = []
        invalid_list_ci95 = []
        invalid_list_ci99 = []
        
        # Iterate
        print("Running for scene: ", scene)
        for y_offset in row_list:
            for x_offset in col_list:
                
                # Run actual check
                valid_coverage, valid_cloud, valid_ci95, valid_ci99 = check_patch(base_path, legend, scene, scenes_ci95, scenes_ci99, size_y, size_x, y_offset, x_offset, img)
                
                #print(scene, img, y_offset, x_offset, valid_coverage, valid_cloud)
                
                if(not(valid_coverage)):
                    key = "r" + str(y_offset) + "_c" + str(x_offset) 
                    invalid_list_coverage.append(key)
                    
                if(not(valid_cloud)):
                    key = "r" + str(y_offset) + "_c" + str(x_offset) 
                    invalid_list_cloud.append(key)
                    
                if(not(valid_ci95)):
                    key = "r" + str(y_offset) + "_c" + str(x_offset) 
                    invalid_list_ci95.append(key)
                    
                if(not(valid_ci99)):
                    key = "r" + str(y_offset) + "_c" + str(x_offset) 
                    invalid_list_ci99.append(key)
                
        invalid_dict["coverage"] = invalid_list_coverage
        invalid_dict["cloud"] = invalid_list_cloud
        invalid_dict["ci95"] = invalid_list_ci95
        invalid_dict["ci99"] = invalid_list_ci99
        scene_dict[img] = invalid_dict
    invalid_patches[scene] = scene_dict
    
print(invalid_patches)
with open("invalid_patches.pkl", 'wb') as fp:
    pickle.dump(invalid_patches, fp)
        
