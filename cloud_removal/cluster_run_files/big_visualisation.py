# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import os
import random
import pickle

# Helper functions   

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


def visualise_patch(base_path, result_path, save_path, legend, scene, patch, y_size, x_size, feature_name, algorithms, cloud_threshold=20, prefix="", visualise_full=True):

                       
    target_path = base_path + legend[scene]['target'] + ".zip"
    feature_path = base_path + legend[scene][feature_name] + ".zip"
    mask_path = base_path + legend[scene]['mask'] + ".zip"
    
    save_path_target = save_path + prefix + "_target.pdf"
    save_path_features = save_path + prefix + "_features.pdf"
    save_path_cloudy = save_path + prefix + "_cloudy.pdf"
    save_path_mask = save_path + prefix + "_mask.pdf"
    
    y_offset = int(patch.split("_")[0][1:])
    x_offset = int(patch.split("_")[1][1:])
       
    target = load_product_windowed(target_path, y_size, x_size, y_offset, x_offset).astype(float)
    mask = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
    mask_img = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset).astype(float)
    
    features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset).astype(float)
     
    target_cloudy = target.copy()

    for i in range(0, target_cloudy.shape[0]):
        for j in range(0, target_cloudy.shape[1]):
            if(mask[i,j] > cloud_threshold):
                a = np.ones(target_cloudy.shape[2]) * np.nan
                target_cloudy[i,j,:] = a
                    
                    
    # Visualise base now
    normalise_and_visualise(target, save_fig=True, save_path=save_path_target)
    normalise_and_visualise(features, save_fig=True, save_path=save_path_features)
    normalise_and_visualise(mask_img, save_fig=True, save_path=save_path_mask)
    normalise_and_visualise(target_cloudy, save_fig=True, save_path=save_path_cloudy)
    
    # Visualise results now
    for alg in algorithms:
        result_path_alg = result_path + "results/" + scene + "/" + alg + "/" + patch + "/" + feature_name + ".npy"
        save_path_result = save_path + prefix + "_" + alg + ".pdf"
        result = np.load(result_path_alg)
        normalise_and_visualise(result, save_fig=True, save_path=save_path_result)
        
    if(visualise_full):
        save_path_target_full = save_path + prefix + "_target_full.pdf"
        save_path_recon_full = save_path + prefix + "_VPint_full.pdf"
        result_path_recon_full = result_path + "results_full/" + scene + "/VPint/" + feature_name + ".npy"
    
        img = load_product(target_path).astype(float)
        mask = load_product(mask_path, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if(mask[i,j] > cloud_threshold):
                    a = np.ones(img.shape[2]) * np.nan
                    img[i,j,:] = a
        
        normalise_and_visualise(img, save_fig=True, save_path=save_path_target_full)
        
        img = np.load(result_path_recon_full)
        normalise_and_visualise(img, save_fig=True, save_path=save_path_recon_full)
    
    return()
    


# Setup

base_path = "/scratch/arp/data/SEN2-MSI-T/"
result_path = "/scratch/arp/results/cloud/"
save_path = "/scratch/arp/results/cloud/big_visualisation/"

scene = "america_cropland_iowa"
patch = "r9_c6"
feature = "1m"
#algorithms = ["VPint", "regression", "regression_band", "replacement"]
algorithms = ["NSPI"]

prefix = "base"

print_cloud_percentages = False
visualise_full = False

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
if(print_cloud_percentages):
    with open("cloud_proportions.pkl", 'rb') as fp:
        prop = pickle.load(fp)
        
        for k, v in prop.items():
            print(k)
            c = 0
            l = list(v.items())
            random.shuffle(l)
            for pa, perc in l:
                if(perc > 30 and perc < 60):
                    print(pa, perc)
                    c += 1
                    if(c >= 5):
                        print("\n")
                        break
    

# Some parameters

size_y = 256
size_x = 256


if(not(os.path.exists(save_path))):
    try:
        os.mkdir(save_path)
    except:
        pass
    

        
visualise_patch(base_path, result_path, save_path, legend, scene, patch, size_y, size_x, feature, algorithms, cloud_threshold=20, prefix=prefix, visualise_full=visualise_full)

        
print("Terminated successfully")