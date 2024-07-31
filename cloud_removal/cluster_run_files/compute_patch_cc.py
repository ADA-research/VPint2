import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import os
import pickle

# Create error overlap plots

# Some preliminaries

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

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)

x_size = 256
y_size = 256

base_path = "/scratch/arp/data/SEN2-MSI-T/"

scenes = list(legend.keys())

scene_dict = {}

for scene in scenes:
    patch_dict = {}
    
    # Check dims, iterate patches
    scene_height = -1
    scene_width = -1
    ref_product_path = base_path + legend[scene]['target'] + ".zip"
    with rasterio.open(ref_product_path) as raw_product:
        product = raw_product.subdatasets
    with rasterio.open(product[1]) as fp:
        scene_height = fp.height
        scene_width = fp.width

    max_row = int(str(scene_height / y_size).split(".")[0])
    max_col = int(str(scene_width / x_size).split(".")[0])

    # Shuffle indices to allow multiple tasks to run
    row_list = np.arange(max_row)
    col_list = np.arange(max_col)
    np.random.shuffle(row_list)
    np.random.shuffle(col_list)

    # Iterate over patches
    for y_offset in row_list:
        for x_offset in col_list:
            patch_name = "r" + str(y_offset) + "_c" + str(x_offset)

            key_mask = scene + "-mask"
            mask_path = base_path + legend[scene]['mask'] + ".zip"
            mask = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"], bands_20m={"CLD":0}).astype(float)[:,:,0]
            mask_flat = mask.flatten()
            
            num_cloud = len(mask_flat[mask_flat > 20])
            cloud_percentage = num_cloud / len(mask_flat) * 100
            patch_dict[patch_name] = cloud_percentage
            
    scene_dict[scene] = patch_dict


with open("cloud_proportions.pkl", 'wb') as fp:
    pickle.dump(scene_dict, fp)
    
print("Finished running.")