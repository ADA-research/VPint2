# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import rasterio.features
from pyproj import Transformer

import satellite_cloud_generator as scg

from VPint.WP_MRP import WP_SMRP

import os
import sys
import random
import pickle
import time

#######################################################
# Code from Meraner et al's GitHub (minor adaptations)
#######################################################

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
from tensorflow.keras.models import Model#, Input
from tensorflow.keras.layers import Input

K.set_image_data_format('channels_first')


def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Definition of Residual Block to be repeated in body of network."""
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)

    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([input_l, tmp])


def DSen2CR_model(input_shape,
                  batch_per_gpu=2,
                  num_layers=32,
                  feature_size=256,
                  use_cloud_mask=True,
                  include_sar_input=True):
    """Definition of network structure. """

    global shape_n

    # define dimensions
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])

    if include_sar_input:
        x = Concatenate(axis=1)([input_opt, input_sar])
    else:
        x = input_opt

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)

    # main body of network as succession of resblocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])

    # One more convolution
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)

    # Add first layer (long skip connection)
    x = Add()([x, input_opt])

    if use_cloud_mask:
        # the hacky trick with global variables and with lambda functions is needed to avoid errors when
        # pickle saving the model. Tensors are not pickable.
        # This way, the Lambda function has no special arguments and is "encapsulated"

        shape_n = tf.shape(input_opt)

        def concatenate_array(x):
            global shape_n
            return K.concatenate([x, K.zeros(shape=(batch_per_gpu, 1, shape_n[2], shape_n[3]))], axis=1)

        x = Concatenate(axis=1)([x, input_opt])

        x = Lambda(concatenate_array)(x)

    model = Model(inputs=[input_opt, input_sar], outputs=x)

    return model



#######################################################
# Code from Aou et al's GitHub (minor adaptations)
#######################################################

from models import create_model, define_network, define_loss, define_metric


model = create_model(
    opt = opt,
    networks = networks,
    phase_loader = phase_loader,
    val_loader = val_loader,
    losses = losses,
    metrics = metrics,
    logger = phase_logger,
    writer = phase_writer
)

model.test()


#############################################
# Own code
#############################################


# Helper functions

def instantiate_dsen2cr(input_shape, use_sar=True):
    # If not using SAR, second part will not get used
    input_shape = ((input_shape[0], input_shape[1], input_shape[2]), (2, input_shape[1], input_shape[2]))

    # Still based on original code; just trying to get the model as accurately as possible

    # model parameters
    num_layers = 16  # B value in paper
    feature_size = 256  # F value in paper

    # include the SAR layers as input to model
    include_sar_input = use_sar

    # cloud mask parameters
    use_cloud_mask = False # my own note: changed because we are not training, only doing inference
    cloud_threshold = 0.2  # set threshold for binarisation of cloud mask

    batch_size = None

    model = DSen2CR_model(input_shape,
                                    batch_per_gpu=batch_size,
                                    num_layers=num_layers,
                                    feature_size=feature_size,
                                    use_cloud_mask=use_cloud_mask,
                                    include_sar_input=include_sar_input)
    
    return(model)


def VPint_interpolation(target_grid, feature_grid, use_IP=True, use_EB=True):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b])
        mu = np.nanmean(target_grid[:,:,b]) + 2*np.nanstd(target_grid[:,:,b])
        pred_grid[:,:,b] = MRP.run(method='exact',
                       auto_adapt=True, auto_adaptation_verbose=False,
                       auto_adaptation_epochs=25, auto_adaptation_max_iter=100,
                       auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, 
                       resistance=use_EB,prioritise_identity=use_IP)

    return(pred_grid)
    
    
def normalise_and_visualise(img, title="", rgb=[3,2,1],percentile_clip=True, save_fig=False, **kwargs):
    
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
        plt.savefig(kwargs['path'],bbox_inches='tight')
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


def load_product_windowed(path, y_size, x_size, y_offset, x_offset, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
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


def simulate_clouds(target, adjust_range=True, **kwargs):
    if(adjust_range):
        target = target / 10000
    target = np.moveaxis(target, -1, 0)
    
    target_cloudy, mask_cloud, mask_shadow = scg.add_cloud_and_shadow(target, return_cloud=True, **kwargs)
    
    target_cloudy = target_cloudy.numpy()[0,:,:,:] # We don't need a batch dim
    mask_cloud = np.moveaxis(mask_cloud.numpy()[0,:,:,:], 0, -1)
    mask_shadow = np.moveaxis(mask_shadow.numpy()[0,:,:,:], 0, -1)
    target_cloudy = np.moveaxis(target_cloudy, 0, -1)
    
    # Combine the two masks, keep highest value (probability of being either cloud or cloud shadow)
    mask = np.maximum.reduce([mask_cloud, mask_shadow])
    
    if(adjust_range):
        target_cloudy = target_cloudy * 10000
    
    return(target_cloudy, mask)


def mask_buffer(mask,passes=1):
    for p in range(0,passes):
        new_mask = mask.copy()
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if(np.isnan(mask[i,j])):
                    if(i>0):
                        new_mask[i-1,j] = np.nan
                    if(i<mask.shape[0]-1):
                        new_mask[i+1,j] = np.nan
                    if(j>0):
                        new_mask[i,j-1] = np.nan
                    if(j<mask.shape[1]-1):
                        new_mask[i,j+1] = np.nan
        mask = new_mask
    return(mask)

def run_dsen2cr(target, sar, model):
    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = np.clip(target, 0, 10000)
    target = target / 2000
    target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])
    
    # Preprocess sar data:
    # move band axis to 0, clip VV to [-25,0], clip VH to [-32.5,0], add respective values to shift to positive, rescale to 0-2 range, add batch dimension
    sar = np.moveaxis(sar, -1, 0)
    sar[1,:,:] = np.clip(sar[1,:,:], -25, 0) + 25 # VV
    sar[0,:,:] = np.clip(sar[0,:,:], -32.5, 0) + 32.5 # VH
    sar[1,:,:] = sar[1,:,:] / 25 * 2
    sar[0,:,:] = sar[0,:,:] / 32.5 * 2
    sar = sar.reshape(1, sar.shape[0], sar.shape[1], sar.shape[2])
    
    # Perform inference
    pred_grid = model((target, sar))

    # Get rid of batch dim, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0,:,:,:], 0, -1) * 2000
    
    return(pred_grid)
   
    
def run_dsen2cr_nosar(target, model):
    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = np.clip(target, 0, 10000)
    target = target / 2000
    target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])

    # Perform inference
    pred_grid = model((target, np.zeros((1, 2, target.shape[2], target.shape[3]))))

    # Get rid of batch dim, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0,:,:,:], 0, -1) * 2000
    
    return(pred_grid)


def run_uncrtaints(target, sar, mask, model):
    # UnCRtain-TS based on ResNet, so use ResNet preprocessing described in paper


    dates_S1 = dates_S2 = [(to_date(date) - to_date('2014-04-03')).days for date in ['2014-04-03']]
    dates = torch.stack((torch.tensor(dates_S1),torch.tensor(dates_S2))).float().mean(dim=0)[None]


    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = process_MS(target, method='resnet')
    #target = np.clip(target, 0, 10000)
    #target = target / 2000

    mask[mask >= 0.1] = 1
    mask[mask < 0.1] = 0
   
    # Preprocess sar data:
    # move band axis to 0, clip VV to [-25,0], clip VH to [-32.5,0], add respective values to shift to positive, rescale to 0-2 range, add batch dimension
    sar = np.moveaxis(sar, -1, 0)
    sar = process_SAR(sar, method='resnet')
    #sar[1,:,:] = np.clip(sar[1,:,:], -25, 0) + 25 # VV
    #sar[0,:,:] = np.clip(sar[0,:,:], -32.5, 0) + 32.5 # VH
    #sar[1,:,:] = sar[1,:,:] / 25 * 2
    #sar[0,:,:] = sar[0,:,:] / 32.5 * 2

    input_img = np.concatenate([target, sar], axis=0) * 10 # Concatenate on band dim, multiply by 10
    input_img = torch.from_numpy(input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1], input_img.shape[2]).astype(np.float32)) # (b, t, c, h, w)
    masks = torch.from_numpy(mask.reshape((1, 1, 1, 256, 256)).astype(np.float32)) 
    y = torch.from_numpy(target.reshape((1, 1, target.shape[0], target.shape[1], target.shape[2])).astype(np.float32))

    inputs = {'A': input_img, 'B': y, 'dates': dates, 'masks': masks}
    
    # Perform inference
    with torch.no_grad():
        # compute single-model mean and variance predictions
        model.set_input(inputs)
        model.forward()
        model.get_loss_G()
        model.rescale()
        pred_grid = model.fake_B.cpu().numpy() / 10 # divide by 10 to undo above thing


    # Perform inference, version 2
    #with torch.no_grad():
    #    fake_B = model(input_img, batch_positions=dates)
    #pred_grid = fake_B / 10

    
        
    # Get rid of batch/time dims, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0, 0, :, :, :], 0, -1) * 2000

    normalise_and_visualise(np.moveaxis(target, 0, -1), title='Target')
    #plt.imshow(sar[0,:,:])
    #plt.title("SAR VH")
    #plt.show()
    #plt.imshow(sar[1,:,:])
    #plt.title("SAR VV")
    #plt.show()
    #plt.imshow(mask)
    #plt.title("Mask")
    #plt.show()
    normalise_and_visualise(pred_grid, title='Reconstruction')
    sdlkfj
   
    return(pred_grid)


def run_patch(base_path, legend, scene, y_size, x_size, y_offset, x_offset, model, cloud_prop_list, 
                       cloud_threshold=0.1, algorithm="dsen2cr", buffer_mask=True, mask_buffer_size=5):
                       
    target_sar_path = base_path + "l1c/collocated_" + scene +".tif"    
    
    # Load target and sar unified image
    try:
        target = load_product_windowed_withSAR(target_sar_path, y_size, x_size, y_offset, x_offset).astype(float)
        target = target[:,:,:13]
        sar = load_product_windowed_withSAR(target_sar_path, y_size, x_size, y_offset, x_offset, dt=np.float32).astype(float)
        sar = sar[:,:,13:]
    except:
        print(y_size, x_size, y_offset, x_offset)
        return(False)

    # L1C, SAR and feature product coverages don't completely align; this will be bad at edges, but filter out most of those cases
    if(np.median(sar) == 0.0 or np.median(target) <= 0):
        print("Coverage problem for:", scene, y_offset, x_offset)
        return(False)
            
    # Simulate clouds, sample from real dataset cloud proportions for similar distribution
    clear_prop = 1 - (cloud_prop_list[np.random.randint(low=0, high=len(cloud_prop_list)-1)] / 100)
    target_cloudy, mask = simulate_clouds(target, channel_offset=0, clear_threshold=clear_prop)
    mask = np.max(mask, axis=2) # just to keep it like the rest without difference per band
            
    if(algorithm == "VPint"):
        # VPint needs feature img (we use 1m) and
        # clouds marked as missing values
        # Load features to filter by feature img coverage (good comparison with VPint)
        feature_path = base_path + legend[scene]["1m"] + ".zip"
        features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset, target_res=10).astype(float)
        features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))
        
        if(buffer_mask):
            # Buffer mask (pad extra masked pixels around edges), also discretises in process
            mask = mask_buffer(mask, mask_buffer_size, threshold=cloud_threshold)
        
        for i in range(0, target_cloudy.shape[0]):
            for j in range(0, target_cloudy.shape[1]):
                if(mask[i,j] > cloud_threshold): 
                    a = np.ones(target_cloudy.shape[2]) * np.nan
                    target_cloudy[i,j,:] = a
    
    if(algorithm=="dsen2cr"):
        pred_grid = run_dsen2cr(target_cloudy, sar, model)
    elif(algorithm=="dsen2cr_nosar"):
        pred_grid = run_dsen2cr_nosar(target_cloudy, model)
    elif(algorithm=="uncrtaints"):
        pred_grid = run_uncrtaints(target_cloudy, sar, mask, model)
    elif(algorithm=="VPint"):
        pred_grid = VPint_interpolation(target_cloudy, features, use_IP=True, use_EB=True)
    
    return(pred_grid)
    


# Setup

if(len(sys.argv) != 3):
    print("Usage: python SEN2-MSI-T-nn.py [method name] [scene number]")

conditions_algorithms = [
    #"dsen2cr", 
    #"dsen2cr_nosar",
    "uncrtaints",
    #"VPint",
]

save_path = "/mnt/c/Users/laure/Data/results/"
base_path = "/mnt/c/Users/laure/Data/SEN2-MSI-T/"

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
conditions_scenes = list(legend.keys())
# Filter out scenes where no SAR data was available
conditions_scenes = [s for s in conditions_scenes if not(legend[s]['sar'] == "")]
conditions_scenes = [
    ##"europe_urban_madrid",
    #"america_shrubs_mexico",
    #"america_herbaceous_peru",
    #"europe_cropland_hungary",
    #"asia_shrubs_indiapakistan",
    "america_forest_mississippi",
]

i = 1
conds = {}

#for scene in conditions_scenes:
#    for alg in conditions_algorithms:
#        conds[i] = {"algorithm":alg, "scene":scene}
#        i += 1
            
#this_run_cond = conds[int(sys.argv[3])]
#scene = conditions_scenes[int(sys.argv[2])]
scene = conditions_scenes[0] # 3 bad


# Hardcoding this because adding UnCRtain-TS messed up CLI arguments, not worth the hassle of fixing it
for scene in conditions_scenes:
    print("Scene:", scene)
    this_run_cond = {
        #"algorithm":sys.argv[1], 
        "algorithm": "uncrtaints", 
        #"algorithm": "dsen2cr", 
        "scene": scene,
    }



    # Some parameters

    replace = True # Set to True to overwrite existing runs
    size_y = 256
    size_x = 256
    size_f = 13

    checkpoint_path = "/mnt/c/Users/laure/Data/models/model_SARcarl.hdf5"
    checkpoint_path_nosar = "/mnt/c/Users/laure/Data/models/model_noSARcarl.hdf5"
    checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/monotemporalL2/"
    #checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/multitemporalL2/"
    #checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/noSAR_1/"

    # Instantiate model and setup TF/torch

    if(this_run_cond["algorithm"] in ["dsen2cr", "dsen2cr_nosar"]):
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        print(tf.test.is_built_with_cuda())

        if gpus:
        # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        input_shape = (size_f, size_y, size_x)
        if(this_run_cond['algorithm'] == "dsen2cr"):
            model = instantiate_dsen2cr(input_shape)
            model.load_weights(checkpoint_path)
        elif(this_run_cond['algorithm'] == "dsen2cr_nosar"):
            model = instantiate_dsen2cr(input_shape, use_sar=False)
            model.load_weights(checkpoint_path_nosar)
        else:
            model = None
        print("Initialised Dsen2-CR and loaded checkpoint weights.")

    elif(this_run_cond["algorithm"] == "uncrtaints"):
        dirname = checkpoint_path_uncrtaints + "conf.json"

        # Bit more of Ebel code
        parser = create_parser(mode='test')
        test_config = parser.parse_args()

        #conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json") if not test_config.load_config else test_config.load_config
        conf_path = checkpoint_path_uncrtaints + "/conf.json"
        if os.path.isfile(conf_path):
            with open(conf_path) as file:
                model_config = json.loads(file.read())
                t_args = argparse.Namespace()
                # do not overwrite the following flags by their respective values in the config file
                no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'root1', 'root2', 'root3', 
                'max_samples_count', 'batch_size', 'display_step', 'plot_every', 'export_every', 'input_t', 'region', 'min_cov', 'max_cov']
                conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
                for key, val in vars(test_config).items(): 
                    if key in no_overwrite: conf_dict[key] = val
                t_args.__dict__.update(conf_dict)
                config = parser.parse_args(namespace=t_args)
        else: config = test_config # otherwise, keep passed flags without any overwriting
        config = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
        if config.pretrain: config.batch_size = 32
        

        device = torch.device(config.device)

        model = get_model(config)
        model = model.to(device)
        print(model)
        model.eval()

    else:
        raise NotImplementedError


    # Prepare for run


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
            
    #for scene in conditions_scenes:
    #    this_run_cond["scene"] = scene

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

    # Get a list of cloud proportions to sample from
    cloud_prop_list = np.zeros(20*500)
    i = 0
    with open("cloud_proportions.pkl", 'rb') as fp:
        a = pickle.load(fp)
        for k, v in a.items():
            for patch, prop in v.items():
                cloud_prop_list[i] = prop
                i += 1
    cloud_prop_list = cloud_prop_list[:i]

    # Check dims, iterate patches
    scene_height = -1
    scene_width = -1
    ref_product_path = base_path + legend[scene]['l1c'] + ".SAFE.zip"
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
            path = path + "/reconstruction.npy"
            
            # Run reconstruction
            if(replace or not(os.path.exists(path))):
                st = time.time()
                #try:
                pred_grid = run_patch(base_path, legend, this_run_cond["scene"], size_y, size_x, y_offset, x_offset, model, cloud_prop_list, algorithm=this_run_cond["algorithm"])
                et = time.time()
                        
                if(type(pred_grid) != type(False)): # just because if(pred_grid) is ambiguous
                    np.save(path, pred_grid)
                        
                #except:
                #    print("Failed run for ", this_run_cond, y_offset, x_offset)
                    
        
print("Terminated successfully")