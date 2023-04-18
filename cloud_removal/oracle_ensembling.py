import numpy as np
import matplotlib.pyplot as plt
import rasterio

import os
import sys
import pickle
import math


# Setup

use_VPint = sys.argv[2]

results_path = "/scratch/arp/results/cloud/results/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"

if(use_VPint=="True"):
    ensemble_name = "OracleEnsembleVPint"
    methods = ["VPint","replacement","regression","ARMA"]
    
else:
    ensemble_name = "OracleEnsembleNoVPint"
    methods = ["replacement","regression","ARMA"]
    
all_scenes = ["dry_australia","flood_brazil","urban_tokyo","vegetation_ukraine","water_baikal"]
features = ["msi_1w","msi_1m","msi_6m","msi_12m"]
props = ["0.1_0.3","0.3_0.7","0.7_0.9"]
replace = False

#scenes = {
#    "1":"dry_australia",
#    "2":"flood_brazil",
#    "3":"urban_tokyo",
#    "4":"vegetation_ukraine",
#    "5":"water_baikal"
#}
params = {
    "1":"dry_australia 0.1_0.3",
    "2":"flood_brazil 0.1_0.3",
    "3":"urban_tokyo 0.1_0.3",
    "4":"vegetation_ukraine 0.1_0.3",
    "5":"water_baikal 0.1_0.3",
    "6":"dry_australia 0.3_0.7",
    "7":"flood_brazil 0.3_0.7",
    "8":"urban_tokyo 0.3_0.7",
    "9":"vegetation_ukraine 0.3_0.7",
    "10":"water_baikal 0.3_0.7",
    "11":"dry_australia 0.7_0.9",
    "12":"flood_brazil 0.7_0.9",
    "13":"urban_tokyo 0.7_0.9",
    "14":"vegetation_ukraine 0.7_0.9",
    "15":"water_baikal 0.7_0.9"
}

test_scene = params[sys.argv[1]].split(" ")[0]
test_prop = params[sys.argv[1]].split(" ")[1]
#test_scene = scenes[sys.argv[1]]


# List of invalid patches
fp = open("scene_patches.pkl","rb")
scene_patches = pickle.load(fp)
fp.close()


# Ensembling specific setup

estimated_num_patches = 6000



enc = {}
for i in range(0,len(methods)):
    enc[methods[i]] = i
    
#dec = {list(v):k for k,v in enc.items()}
    

#########################################
# Begin proper code
#########################################



scene = test_scene
method = methods[0] # Just for os.listdir, real thing added later
#for prop in props:
prop = test_prop
currdir = results_path + scene + "/" + method + "_" + prop + "/"
for patchname in os.listdir(currdir):
    currdir1 = currdir + patchname + "/"
    for feature in features:
    
        # Save path, do first to check for existence
        savedir = results_path + scene + "/"
        if(not(os.path.exists(savedir))):
            try:
                os.mkdir(savedir)
            except:
                pass
                
        savedir = savedir + ensemble_name + "_" + prop + "/"
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
    
            filenames = []
            filename = currdir1 + feature + ".npy"
            for method_real in methods: # True method iteration, bit awkward
                filenames.append(filename.replace(method,method_real))
            
            # Load input, feature and mask
            target_path = original_path + "scenes/" + scene + "/" + patchname + "_msi_target.tif"
            feature_path = original_path + "scenes/" + scene + "/" + patchname + "_" + feature + ".tif"
            mask_path = original_path + "scenes/" + scene + "/" + patchname + "_mask.tif"           
            
            fp = rasterio.open(mask_path)
            mask_grid = np.moveaxis(fp.read().astype(float),0,-1)[:,:,0]
            fp.close()
            fp = rasterio.open(target_path)
            target_grid = np.moveaxis(fp.read().astype(float),0,-1)
            fp.close()
            fp = rasterio.open(feature_path)
            feature_grid = np.moveaxis(fp.read().astype(float),0,-1)
            fp.close()
            
            # Load pred grids
            
            pred_grids = []
            for fp in filenames:
                pred_grid = np.load(fp)
                pred_grids.append(pred_grid)
            
            # Apply mask for f0, rest doesn't need it
            
            shp = target_grid.shape
            size = np.product(shp)
            
            mask = np.zeros(shp)
            for b in range(0,mask.shape[2]):
                mask[:,:,b] = mask_grid.copy()
            mask = mask.reshape(size)
            
            input_grid = target_grid.copy()
            input_grid_flat = input_grid.reshape(size)
            input_grid_flat = input_grid_flat[mask>0.3]
            
            # Pre-compute means
            f0 = np.nanmean(input_grid_flat)
            f1 = np.nanmean(feature_grid)
            
            # Initialise ensemble pred grid
            ensemble_pred_grid = np.zeros(input_grid.shape)
            
            # Iterate over pixels
            for i in range(0,input_grid.shape[0]):
                for j in range(0,input_grid.shape[1]):
                    for b in range(0,input_grid.shape[2]):
            
                        # Replace non-cloudy with input, cloudy with predicted method pred
                        if(mask_grid[i,j] > 0.3):
                            feature_vec = np.zeros(len(methods))
                            
                            for p_i in range(0,len(pred_grids)):
                                offset = p_i
                                feature_vec[offset] = pred_grids[p_i][i,j,b]
                                
                            errs = np.absolute(np.ones(len(feature_vec))*target_grid[i,j,b] - feature_vec)    
                            
                            ind = np.argmin(errs)
                            ensemble_pred_grid[i,j,b] = feature_vec[ind]
                            
                        else:
                            ensemble_pred_grid[i,j,b] = input_grid[i,j,b]
                    
            # Actually save
            np.save(fn,ensemble_pred_grid)
                
print("Finished run for " + test_scene + " " + str(prop))