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
    
candidates = []
with open("cloud_proportions.pkl", 'rb') as fp:
    clp = pickle.load(fp)
    for scene, d in clp.items():
        for patch, prop in d.items():
            if(prop > 50 and prop < 65):
                candidates.append(scene + "_" + patch + "_" + str(prop))
print(candidates)

size_y = 256
size_x = 256


# Main code

result_path = "/scratch/arp/results/cloud/results/"
base_path = "/scratch/arp/data/SEN2-MSI-T/"
save_path = "/scratch/arp/results/cloud/"
method1 = "VPint"
method2 = "regression"
feature = "1m"

clip = False
clip_val = 2000

percentile_clip = True
percentile_clip_val = 99

lcs = {
    "cropland":("europe_cropland_hungary", "r19_c5"),
    "urban":("america_urban_atlanta", "r1_c6"),
    "shrubs":("america_shrubs_mexico", "r19_c10"),
    "forest":("africa_forest_angola", "r4_c13"),
    "herbaceous":("asia_herbaceous_kazakhstan", "r9_c14"),
}

save_path = save_path + "err_corr_plot/"
if(not(os.path.exists(save_path))):
    try:
        os.mkdir(save_path)
    except:
        pass

for lc, t in lcs.items():
    scene = t[0]
    patch = t[1]
    offset_y = int(patch.split("_")[0][1:])
    offset_x = int(patch.split("_")[1][1:])

    path1 = result_path + scene + "/" + method1 + "/" + patch + "/" + feature + ".npy"
    path2 = result_path + scene + "/" + method2 + "/" + patch + "/" + feature + ".npy"
    img1 = np.load(path1)
    img2 = np.load(path2)

    key_target = scene + "-target"
    target_path = base_path + legend[scene]['target'] + ".zip"
    target = load_product_windowed(target_path, size_y, size_x, offset_y, offset_x).astype(float)

    err1 = np.mean(np.absolute(img1 - target), axis=-1)
    err2 = np.mean(np.absolute(img2 - target), axis=-1)
    if(clip):
        err1 = np.clip(err1, 0, clip_val)
        err2 = np.clip(err2, 0, clip_val)
    if(percentile_clip):
        min_val = np.nanpercentile(err1, 1)
        max_val = np.nanpercentile(err1, 99)
        err1 = np.clip((err1-min_val)/(max_val-min_val), 0, 1)
        err2 = np.clip((err2-min_val)/(max_val-min_val), 0, 1)
    else:
        err1 = err1 - np.min(err1)
        err2 = err2 - np.min(err2)
        err1 = err1 / np.max(err1)
        err2 = err2 / np.max(err2)

    rgb = np.zeros((err1.shape[0], err1.shape[1], 3))
    rgb[:,:,0] = err1 / np.max(err1)
    rgb[:,:,2] = err2 / np.max(err2)
    
    plt.imshow(rgb, vmin=0, vmax=1.0)
    plt.axis('off')
    plt.savefig(save_path + "err_corr_" + lc + ".pdf", bbox_inches='tight')
    plt.show()

print("Finished running.")