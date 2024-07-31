import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import os
import pickle


# Functions

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


def normalise_and_visualise(img, title="", rgb=[3,2,1], percentile_clip=True, save_fig=False,**kwargs):
    
    new_img = np.zeros((img.shape[0],img.shape[1],3))
    new_img[:,:,0] = img[:,:,rgb[0]]
    new_img[:,:,1] = img[:,:,rgb[1]]
    new_img[:,:,2] = img[:,:,rgb[2]]
    
    if(percentile_clip):
        min_val = np.nanpercentile(new_img,1)
        max_val = np.nanpercentile(new_img,99)

        new_img = np.clip((new_img-min_val)/(max_val-min_val),0,1)
    
    plt.imshow(new_img,interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if(save_fig):
        plt.savefig(kwargs['path'], bbox_inches='tight')
    plt.show()
    

# Parameters


base_path = "/scratch/arp/data/SEN2-MSI-T/"
result_path = "/scratch/arp/results/cloud/results/"
save_path = "/scratch/arp/results/cloud/full_reconstruction/"

algs = ["VPint", "VPint_no_EB", "replacement"]
scenes = ["europe_cropland_hungary", 
            "africa_herbaceous_southafrica", 
            "america_forest_mississippi", 
            "asia_herbaceous_kazakhstan",
        ]
#scenes = ["america_forest_mississippi"]
algs = ["VPint"]
features = "1m"

size_y = 256
size_x = 256

clip = False
clip_y_min = 0
clip_y_max = 500
clip_x_min = 0
clip_x_max = 500

save_target = False
save_mask = False
save_features = False


# Start running

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
for scene in scenes:
    for alg in algs:


        # Assume scenes of 20x20 patches
        recon = np.zeros((21*size_y, 21*size_x, 12))
        currdir = result_path + scene + "/" + alg + "/" 

        for patchname in os.listdir(currdir):
            path = currdir + patchname + "/" + features + ".npy"
            i = int(patchname.split("_")[0][1:])
            j = int(patchname.split("_")[1][1:])
            data = np.load(path)
            ip_start = i*size_y
            ip_end = i*size_y + size_y
            jp_start = j*size_x
            jp_end = j*size_x + size_x
            recon[ip_start:ip_end, jp_start:jp_end, :] = data
            
        if(clip):
            recon = recon[clip_y_min:clip_y_max, clip_x_min:clip_x_max, :]
            save_full_path = save_path + scene + "-" + features + "-" + alg + "_clipped.pdf"
        else:
            save_full_path = save_path + scene + "-" + features + "-" + alg + ".pdf"
        normalise_and_visualise(recon, save_fig=True, path=save_full_path)

    if(save_target):
        prod_path = base_path + legend[scene]['target'] + ".zip"
        prod = load_product(prod_path).astype(float)
        if(clip):
            prod = prod[clip_y_min:clip_y_max, clip_x_min:clip_x_max, :]
            save_full_path = save_path + scene + "-target_clipped.pdf"
        else:
            save_full_path = save_path + scene + "-target.pdf"
        normalise_and_visualise(prod, save_fig=True, path=save_full_path)
        
    if(save_mask):
        save_full_path = save_path + scene + "-mask.pdf"
        prod_path = base_path + legend[scene]['mask'] + ".zip"
        prod = load_product(prod_path).astype(float)
        if(clip):
            prod = prod[clip_y_min:clip_y_max, clip_x_min:clip_x_max, :]
            save_full_path = save_path + scene + "-mask_clipped.pdf"
        else:
            save_full_path = save_path + scene + "-mask.pdf"
        normalise_and_visualise(prod, save_fig=True, path=save_full_path)
        
    if(save_features):
        save_full_path = save_path + scene + "-" + features + ".pdf"
        prod_path = base_path + legend[scene][features] + ".zip"
        prod = load_product(prod_path).astype(float)
        if(clip):
            prod = prod[clip_y_min:clip_y_max, clip_x_min:clip_x_max, :]
            save_full_path = save_path + scene + "-" + features + "_clipped.pdf"
        else:
            save_full_path = save_path + scene + "-" + features + ".pdf"
        normalise_and_visualise(prod, save_fig=True, path=save_full_path)

print("Done.")