import numpy as np
import matplotlib.pyplot as plt
import rasterio

import os
import sys
import pickle
import math

from skimage.measure import compare_ssim


# Setup



results_path = "/scratch/arp/results/cloud/results/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"
save_dir = "/scratch/arp/results/cloud/NDVI_performance_plot/"

methods = ["VPint","replacement","regression","ARMA"]
scenes = ["dry_australia","flood_brazil","urban_tokyo","vegetation_ukraine","water_baikal"]
features = ["msi_1w","msi_1m","msi_6m","msi_12m"]

#msi_indices = ["NDVI","RVI","PSSRa","TNDVI","NDI45M","GNDVI","MCARI","S2REP",
#                "IRECI","SAVI"]
#msi_indices = ["PSSRa","TNDVI","NDI45M","GNDVI","MCARI","S2REP",
#                "IRECI","SAVI"] # TODO: remove, just testing


method_markers = {
    "VPint":"s",
    "replacement":"P",
    "regression":"o",
    "ARMA":"X",
}

msi_indices = {
    "1":"NDVI",
    "2":"RVI",
    "3":"PSSRa",
    "4":"TNDVI",
    "5":"NDI45M",
    "6":"GNDVI",
    "7":"MCARI",
    "8":"S2REP",
    "9":"IRECI",
    "10":"SAVI",
}
msi_indices = {
    "1":"contrast",
    "2":"contrast std",
    "3":"dynamicity",
    "4":"SSIM"
}
msi_index = msi_indices[sys.argv[1]]

# List of invalid patches
fp = open("scene_patches.pkl","rb")
scene_patches = pickle.load(fp)
fp.close()



def compute_ndvi(vals):
    t1 = vals[:,:,7]-vals[:,:,3]
    t2 = vals[:,:,7]+vals[:,:,3]
    ndvi = np.nanmean(t1 / t2)
    return(ndvi)
    
def compute_rvi(vals):
    t1 = vals[:,:,7]
    t2 = vals[:,:,3]
    rvi = np.nanmean(t1 / t2)
    return(rvi)
    
def compute_pssra(vals):
    t1 = vals[:,:,6]
    t2 = vals[:,:,3]
    pssra = np.nanmean(t1 / t2)
    return(pssra)
    
def compute_tndvi(vals):
    ndvi = compute_ndvi(vals)
    tndvi = math.sqrt(ndvi + 0.5)
    return(tndvi)
    
def compute_ndi45m(vals):
    t1 = vals[:,:,4]-vals[:,:,3]
    t2 = vals[:,:,4]+vals[:,:,3]
    ndi45m = np.nanmean(t1 / t2)
    return(ndi45m)
    
def compute_gndvi(vals):
    t1 = vals[:,:,7]-vals[:,:,2]
    t2 = vals[:,:,7]+vals[:,:,2]
    gndvi = np.nanmean(t1 / t2)
    return(gndvi)
    
def compute_mcari(vals):
    t1 = vals[:,:,4]-vals[:,:,3]
    t2 = vals[:,:,4]-vals[:,:,2]
    mcari = np.nanmean((t1 - 0.2*t2) * t1)
    return(mcari)
    
def compute_s2rep(vals):
    t1 = vals[:,:,3]+vals[:,:,6]
    t2 = vals[:,:,4] / (vals[:,:,5]-vals[:,:,4])
    s2rep = np.nanmean(705+35 * (t1/2) - t2)
    return(s2rep)
    
def compute_ireci(vals): 
    t1 = vals[:,:,6]-vals[:,:,3]
    t2 = vals[:,:,4]/vals[:,:,5]
    ireci = np.nanmean(t1 / t2)
    return(ireci)
    
def compute_savi(vals,L=0.428):
    tL = np.ones((vals.shape[0],vals.shape[1]))*L
    t1 = vals[:,:,7]-vals[:,:,3]
    t2 = vals[:,:,7]+vals[:,:,3]+tL
    t3 = 1+tL
    
    savi = np.nanmean(t1 / t2 * t3)
    return(savi)

def compute_contrast(vals): 
    tot_contrast = 0
    for b in range(0,vals.shape[2]):
        height = vals.shape[0]
        width = vals.shape[1]

        grid = vals[:,:,b]
        
        # Create neighbour count grid
        neighbour_count_grid = np.ones(grid.shape) * 4

        neighbour_count_grid[:,0] = neighbour_count_grid[:,0] - np.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[:,width-1] = neighbour_count_grid[:,width-1] - np.ones(neighbour_count_grid.shape[1])

        neighbour_count_grid[0,:] = neighbour_count_grid[0,:] - np.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[height-1,:] = neighbour_count_grid[height-1,:] - np.ones(neighbour_count_grid.shape[0])

        # Create (h*w*4) value grid
        val_grid = np.zeros((height,width,4))
        
        up_grid = np.zeros((height,width))
        right_grid = np.zeros((height,width))
        down_grid = np.zeros((height,width))
        left_grid = np.zeros((height,width))

        up_grid[1:-1,:] = grid[0:-2,:]
        right_grid[:,0:-2] = grid[:,1:-1]
        down_grid[0:-2,:] = grid[1:-1,:]
        left_grid[:,1:-1] = grid[:,0:-2]
        
        val_grid[:,:,0] = up_grid
        val_grid[:,:,1] = right_grid
        val_grid[:,:,2] = down_grid
        val_grid[:,:,3] = left_grid
        
        # Compute contrast as average absolute distance
        temp_grid = np.repeat(grid[:,:,np.newaxis],4,axis=2)
        diff = np.absolute(val_grid-temp_grid)
        sum_diff = np.nansum(diff,axis=-1)
        avg_contrast = sum_diff / neighbour_count_grid
        
        min_val = np.nanmin(avg_contrast)
        max_val = np.nanmax(avg_contrast)
        avg_contrast = np.nanmean(np.clip((avg_contrast-min_val)/(max_val-min_val), 0,1))
        
        tot_contrast += avg_contrast
        
    tot_contrast = tot_contrast / vals.shape[2]
    
    return(tot_contrast)

def compute_contrast_std(vals): 
    tot_contrast = 0
    for b in range(0,vals.shape[2]):
        height = vals.shape[0]
        width = vals.shape[1]

        grid = vals[:,:,b]
        
        # Create neighbour count grid
        neighbour_count_grid = np.ones(grid.shape) * 4

        neighbour_count_grid[:,0] = neighbour_count_grid[:,0] - np.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[:,width-1] = neighbour_count_grid[:,width-1] - np.ones(neighbour_count_grid.shape[1])

        neighbour_count_grid[0,:] = neighbour_count_grid[0,:] - np.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[height-1,:] = neighbour_count_grid[height-1,:] - np.ones(neighbour_count_grid.shape[0])

        # Create (h*w*4) value grid
        val_grid = np.zeros((height,width,4))
        
        up_grid = np.zeros((height,width))
        right_grid = np.zeros((height,width))
        down_grid = np.zeros((height,width))
        left_grid = np.zeros((height,width))

        up_grid[1:-1,:] = grid[0:-2,:]
        right_grid[:,0:-2] = grid[:,1:-1]
        down_grid[0:-2,:] = grid[1:-1,:]
        left_grid[:,1:-1] = grid[:,0:-2]
        
        val_grid[:,:,0] = up_grid
        val_grid[:,:,1] = right_grid
        val_grid[:,:,2] = down_grid
        val_grid[:,:,3] = left_grid
        
        # Compute contrast as average absolute distance
        temp_grid = np.repeat(grid[:,:,np.newaxis],4,axis=2)
        diff = np.absolute(val_grid-temp_grid)
        sum_diff = np.nansum(diff,axis=-1)
        avg_contrast = sum_diff / neighbour_count_grid
        
        min_val = np.nanmin(avg_contrast)
        max_val = np.nanmax(avg_contrast)
        avg_contrast = np.nanstd(np.clip((avg_contrast-min_val)/(max_val-min_val), 0,1))
        
        tot_contrast += avg_contrast
        
    tot_contrast = tot_contrast / vals.shape[2]
    
    return(tot_contrast)

def compute_dynamicity(ground_truth_path): 
    # Compute the mean difference between target and features
    # Average over all features, use as indication of dynamicity
    msi_1w_path = ground_truth_path.replace("msi_target", "msi_1w")
    msi_1m_path = ground_truth_path.replace("msi_target", "msi_1m")
    msi_6m_path = ground_truth_path.replace("msi_target", "msi_6m")
    msi_12m_path = ground_truth_path.replace("msi_target", "msi_12m")
    paths = [msi_1w_path, msi_1m_path, msi_6m_path, msi_12m_path]

    ground_truth_vals = np.moveaxis(rasterio.open(ground_truth_path).read(),0,-1)

    dynamicity = 0.0
    for feature_path in paths:
        feature_vals = np.moveaxis(rasterio.open(feature_path).read(),0,-1)
        mean_diff = np.nanmean(np.absolute(ground_truth_vals-feature_vals))
        dynamicity += mean_diff

    dynamicity = dynamicity / len(paths)
    
    return(dynamicity)

def compute_SSIM(ground_truth_path): 
    # Compute the mean SSIM between target and features
    # Average over all features
    msi_1w_path = ground_truth_path.replace("msi_target", "msi_1w")
    msi_1m_path = ground_truth_path.replace("msi_target", "msi_1m")
    msi_6m_path = ground_truth_path.replace("msi_target", "msi_6m")
    msi_12m_path = ground_truth_path.replace("msi_target", "msi_12m")
    paths = [msi_1w_path, msi_1m_path, msi_6m_path, msi_12m_path]

    ground_truth_vals = np.moveaxis(rasterio.open(ground_truth_path).read(),0,-1)

    ssim = 0.0
    tot_count = 0
    for feature_path in paths:
        feature_vals = np.moveaxis(rasterio.open(feature_path).read(),0,-1)
        #try:
        (s,d) = compare_ssim(ground_truth_vals,feature_vals,full=True)
        #except:
        #    s = np.nan
        if(not(np.isnan(s))):
            tot_count += 1
            ssim += s

    if(tot_count == 0):
        ssim = -1
    else:
        ssim = ssim / tot_count
    print(ssim)
    return(ssim)

method_y = {}

# Result 1: NDVI --> dict of scene:avg NDVI
# Result 2: method per-scene performance dict method:scene performance
# Result 3: method scene dict list --> list of method:scene performances dict


#for msi_index in msi_indices:

# For all scenes, compute average NDVI

# For all methods, compute average MAE on scene

method_scene_perf_dict = {}
index_dict = {}



for scene in scenes: # dry_australia etc
    method_scene_perf = {}
    for method in methods: # VPint etc
        
        # Get target image invalid patches
        target_key = scene + "_msi_target"
        invalid_patches = scene_patches[target_key]
        
        resdir = results_path + scene + "/"       
        total_mae = 0 # MAE counter, to be divided by i
        
        total_index = 0 # NDVI counter, to be divided by i
        
        i = 0 # Number of patches for this method-scene combination
        
        for dirname in os.listdir(resdir): # VPint_0.1_0.3 etc
            if(dirname.split("_")[0] == method): 
                methoddir = resdir + dirname + "/"
                for patchname in os.listdir(methoddir): # r0_c0 etc
                    # Check for invalid patches in this scene's target image
                    if(patchname + "_msi_target" not in invalid_patches):
                        patchdir = methoddir + patchname + "/"
                        for feature in features: # msi_1w etc
                            # Check for invalid patches specific to feature set
                            feature_key = scene + "_" + feature
                            invalid_patches2 = scene_patches[feature_key]
                            if(patchname + "_" + feature not in invalid_patches2):
                                # Paths for both method results and ground truth
                                m_path = patchdir + feature + ".npy"
                                ground_truth_path = original_path + "scenes/" + scene + "/" + patchname + "_msi_target.tif"
                                
                                # Load results + ground truth
                                true_vals = np.moveaxis(rasterio.open(ground_truth_path).read(),0,-1)
                                
                                # Compute indices

                                compute_error = True
                                
                                if(msi_index=="NDVI"):
                                    total_index += compute_ndvi(true_vals)
                                elif(msi_index=="RVI"):
                                    total_index += compute_rvi(true_vals)
                                elif(msi_index=="PSSRa"):
                                    total_index += compute_pssra(true_vals)
                                elif(msi_index=="TNDVI"):
                                    total_index += compute_tndvi(true_vals)
                                elif(msi_index=="NDI45M"):
                                    total_index += compute_ndi45m(true_vals)
                                elif(msi_index=="GNDVI"):
                                    total_index += compute_gndvi(true_vals)
                                elif(msi_index=="MCARI"):
                                    total_index += compute_mcari(true_vals)
                                elif(msi_index=="S2REP"):
                                    total_index += compute_s2rep(true_vals)
                                elif(msi_index=="IRECI"):
                                    total_index += compute_ireci(true_vals)
                                elif(msi_index=="SAVI"):
                                    total_index += compute_savi(true_vals)
                                elif(msi_index=="contrast"):
                                    total_index += compute_contrast(true_vals)
                                elif(msi_index=="contrast std"):
                                    total_index += compute_contrast_std(true_vals)
                                elif(msi_index=="dynamicity"):
                                    total_index += compute_dynamicity(ground_truth_path)
                                elif(msi_index=="SSIM"):
                                    ssim = compute_SSIM(ground_truth_path)
                                    # For nan results
                                    if(ssim > -1):
                                        total_index += ssim
                                    else:
                                        compute_error = False

                                

                                # Compute 

                                if(compute_error):
                                    try:
                                        m_diff = None
                                        m_vals = np.load(m_path)
                                        #with open(m_path,'r') as fp:
                                            #m_vals = np.array(eval(fp.read().replace("nan","np.nan"))).astype(float)
                                        m_diff = np.nanmean(np.absolute(m_vals - true_vals))
                                        
                                        # Update values
                                        
                                        total_mae += m_diff
                                        i += 1
                                    except:
                                        print("Failure for patch: " + str(m_path))
                                    
        avg_mae = total_mae / i
        
        avg_index = total_index / i
        
        method_scene_perf[method] = avg_mae
        
        index_dict[scene] = avg_index # This will get recomputed |methods| times, but easy
        
    
    method_scene_perf_dict[scene] = method_scene_perf

print("method_scene_perf_dict:")
print(method_scene_perf_dict)
print("index_dict:")
print(index_dict)

for method in methods:
    x_index = []
    
    y_mae = []
    i = 0
    for sc,s_dict in method_scene_perf_dict.items():
        for m,avg_mae in s_dict.items():
            if(m==method):
                y_mae.append(avg_mae)
                x_index.append(index_dict[sc])

                i += 1
            
    x_index = np.array(x_index)
    y_mae = np.array(y_mae)
    
    y_mae = y_mae.ravel()[x_index.argsort(axis=None).reshape(x_index.shape)]
    x_index = x_index.ravel()[x_index.argsort(axis=None).reshape(x_index.shape)]
    
    plt.plot(x_index,y_mae,marker=method_markers[method],label=method)

plt.title("Effect of pixel properties on performance")
plt.xlabel("Average scene " + msi_index)
plt.ylabel("Average scene MAE")
plt.legend()
plt.savefig(save_dir + msi_index + "_performance_plot.pdf",bbox_inches='tight')
plt.show()
plt.clf()

