import numpy as np
import matplotlib.pyplot as plt
import rasterio

import os
import sys
import pickle
import math
import random

from sklearn.svm import SVC
from sklearn.utils import shuffle

import autosklearn.classification

# Setup

use_VPint = sys.argv[2]

results_path = "/scratch/arp/results/cloud/results/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"

autosklearn_store_path = "/scratch/arp/autosklearn/cld_ens/"

if(use_VPint=="True"):
    ensemble_name = "ensembleVPint"
    methods = ["VPint","replacement","regression","ARMA"]
    
else:
    ensemble_name = "ensembleNoVPint"
    methods = ["replacement","regression","ARMA"]
    
all_scenes = ["dry_australia","flood_brazil","urban_tokyo","vegetation_ukraine","water_baikal"]
features = ["msi_1w","msi_1m","msi_6m","msi_12m"]
props = ["0.1_0.3","0.3_0.7","0.7_0.9"]
replace = True

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

num_pixels_to_sample = 250
estimated_num_patches = 6000

#enc = {}
#for i in range(0,len(methods)):
#    vec = np.zeros(len(methods))
#    vec[i] = 1
#    enc[methods[i]] = vec

enc = {}
for i in range(0,len(methods)):
    enc[methods[i]] = i
    
#dec = {list(v):k for k,v in enc.items()}
    

#########################################
# Begin proper code
#########################################

# Training


# Set training set to set of other scenes in dataset
training_scenes = all_scenes.remove(test_scene)
print(all_scenes)

# Initiate feature matrix and label matrix (one-hot encoding)
num_features = len(methods) + 3 # Pred intensity + nanmean inp, nanmean feature, band
X_train = np.zeros((num_pixels_to_sample*estimated_num_patches,num_features))
#Y_train = np.zeros((num_pixels_to_sample*estimated_num_patches,len(methods)))
Y_train = np.zeros(num_pixels_to_sample*estimated_num_patches)

instance = 0
# Iterate over patches in training set
for scene in all_scenes:
    method = methods[0] # Just for os.listdir, real thing added later
    for prop in props:
        currdir = results_path + scene + "/" + method + "_" + prop + "/"
        for patchname in os.listdir(currdir):
            currdir1 = currdir + patchname + "/"
            for feature in features:
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
                
                # Apply mask to target and pred grids
                
                shp = target_grid.shape
                size = np.product(shp)
                
                mask = np.zeros(shp)
                for b in range(0,mask.shape[2]):
                    mask[:,:,b] = mask_grid.copy()
                mask_full_grid = mask.copy()
                mask = mask.reshape(size)
                
                input_grid = target_grid.copy()
                input_grid_flat = input_grid.reshape(size)
                input_grid_flat = input_grid_flat[mask>0.3]
                
                #for pred_grid in pred_grids:
                #    pred_grid = pred_grid.reshape(size)
                #    pred_grid = pred_grid[mask>0.3]
                
                # Pre-compute means
                f0 = np.nanmean(input_grid_flat)
                f1 = np.nanmean(feature_grid)
                
                # Sample random index from target and pred grids
                for c in range(0,num_pixels_to_sample):
                    # Create feature vector
                    feature_vec = np.zeros(len(methods)+3)
                    feature_vec[0] = f0
                    feature_vec[1] = f1
                    # Ensure pixel is cloudy
                    nan_inds = np.argwhere(mask_full_grid>0.3)
                    pixel_inds = random.choice(nan_inds)
                    pixel_index_i = pixel_inds[0]
                    pixel_index_j = pixel_inds[1]
                    pixel_index_b = pixel_inds[2]
                    
                    feature_vec[2] = pixel_index_b
                    
                    for i in range(0,len(pred_grids)):
                        offset = 3+i
                        feature_vec[offset] = pred_grids[i][pixel_index_i,pixel_index_j,pixel_index_b]
                    
                    # Create label
                    
                    perfs = np.absolute(feature_vec[3:]-input_grid[pixel_index_i,pixel_index_j,pixel_index_b])
                    best_ind = np.argmin(perfs)
                    best_method = methods[best_ind]
                    label = enc[best_method]
                    
                    # Add to matrices
                    X_train[instance,:] = feature_vec
                    #Y_train[instance,:] = label
                    Y_train[instance] = label
                    instance += 1


# Remove unused rows, shuffle data

X_train = X_train[:instance,:]
#Y_train = Y_train[:instance,:]
Y_train = Y_train[:instance]
X_train, Y_train = shuffle(X_train,Y_train)

# TODO: Train auto-sklearn classifier for 12? hours on training set
# For now: just basic SVM

print("Starting training for " + test_scene)

autoskl_current_id = 1
if(not(os.path.exists("/scratch/arp/autosklearn/id.txt"))): 
    with open("/scratch/arp/autosklearn/id.txt",'w') as fp:
        fp.write("2")
else:
    with open("/scratch/arp/autosklearn/id.txt",'r') as fp:
        s = fp.read()
        autoskl_current_id = int(s)
    with open("/scratch/arp/autosklearn/id.txt",'w') as fp:
        fp.write(str(autoskl_current_id+1))
        
model = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=43200, # seconds, so 12 hours
    per_run_time_limit=3600, # one hour
    tmp_folder=autosklearn_store_path + str(autoskl_current_id) + "temp",
    output_folder=autosklearn_store_path + str(autoskl_current_id) + "out",
    delete_tmp_folder_after_terminate=True,
    delete_output_folder_after_terminate=True,
)

#model = SVC()
model.fit(X_train,Y_train)


print("Finished training for " + test_scene + ", now starting test")




# Testing

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
            mask_full_grid = mask.copy()
            mask = mask.reshape(size)
            
            input_grid = target_grid.copy()
            input_grid_flat = input_grid.reshape(size)
            input_grid_flat = input_grid_flat[mask>0.3]
            
            # Pre-compute means
            f0 = np.nanmean(input_grid_flat)
            f1 = np.nanmean(feature_grid)
            
            # Initialise ensemble pred grid
            ensemble_pred_grid = target_grid.copy() # if 0 error, this is why and I made an error elsewhere
            
            # Create matrix for efficiency
            pixel_list_matrix = np.ones((np.product(input_grid.shape),len(methods)+3))
            pixel_list_matrix[:,0] = pixel_list_matrix[:,0] * f0
            pixel_list_matrix[:,1] = pixel_list_matrix[:,1] * f1
            easy_index_matrix = np.zeros((np.product(input_grid.shape),3))
            
            instance = 0
            # Iterate over pixels
            for i in range(0,input_grid.shape[0]):
                for j in range(0,input_grid.shape[1]):
                    for b in range(0,input_grid.shape[2]):
            
                        # Replace non-cloudy with input, cloudy with predicted method pred
                        if(mask_grid[i,j] > 0.3):
                            feature_vec = np.zeros(len(methods)+1)
                            feature_vec[0] = b
                            
                            for p_i in range(0,len(pred_grids)):
                                offset = 1+p_i
                                feature_vec[offset] = pred_grids[p_i][i,j,b]
                                
                                
                            pixel_list_matrix[instance,2:] = feature_vec
                            easy_index_matrix[instance,:] = np.array([i,j,b])
                            instance += 1

                        #    f = feature_vec.reshape(1,len(feature_vec))
                        #    pred_best_method = methods[int(model.predict(f)[0])]
                        #    pred_best_index = methods.index(pred_best_method)
                        #    ensemble_pred_grid[i,j,b] = pred_grids[pred_best_index][i,j,b]
                            
                        #else:
                        #    ensemble_pred_grid[i,j,b] = input_grid[i,j,b]
                    
                    
            # Get rid of unused rows
            pixel_list_matrix = pixel_list_matrix[:instance,:]
            easy_index_matrix = easy_index_matrix[:instance,:]
            
            # Make pred
            pred_method_vec = model.predict(pixel_list_matrix).astype(int)
            
            # Iterate over rows and replace
            for r in range(0,easy_index_matrix.shape[0]):
                # Get index of pixel
                inds = easy_index_matrix[r,:].astype(int)
                # Get predicted pixel value by predicted best method
                pred_method = pred_method_vec[r]
                #pred_best_index = methods.index(pred_method)
                pred_best_index = methods[pred_method]
                pred_val = pred_grids[pred_best_index][inds]
                # Replace
                ensemble_pred_grid[inds] = pred_val
                    
            # Actually save
            np.save(fn,ensemble_pred_grid)
                
print("Finished run for " + test_scene + " " + str(prop))