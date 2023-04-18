import numpy as np
import matplotlib.pyplot as plt
import rasterio

import os
import sys
import pickle


# Setup



results_path = "/scratch/arp/results/cloud/results/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"
save_dir = "/scratch/arp/results/cloud/wavelength_error_plot/"

methods = ["VPint","replacement","regression","ARMA","NN"]
scenes = ["dry_australia","flood_brazil","urban_tokyo","vegetation_ukraine","water_baikal"]
scene_name_extra = "_all"
#scenes = ["water_baikal"]
#scene_name_extra = "_only_water_baikal"
features = ["msi_1w","msi_1m","msi_6m","msi_12m"]

band_reflectance = {
    0:443,#"B01"
    1:490,#"B02"
    2:560,#"B03"
    3:665,#"B04"
    4:705,#"B05"
    5:740,#"B06"
    6:783,#"B07"
    7:842,#"B08"
    8:865,#"B8A"
    9:940,#"B9"
}

method_markers = {
    "VPint":"s",
    "replacement":"P",
    "regression":"o",
    "ARMA":"X",
}

# List of invalid patches
fp = open("scene_patches.pkl","rb")
scene_patches = pickle.load(fp)
fp.close()

method_y = {}

for method in methods: # VPint etc
    method_avg_error = np.zeros((256,256,10))
    i = 0
    for scene in scenes: # dry_australia etc
        # Get target image invalid patches
        target_key = scene + "_msi_target"
        invalid_patches = scene_patches[target_key]
        
        resdir = results_path + scene + "/"       
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
                                
                                try:
                                    m_diff = None
                                    m_vals = np.load(m_path)
                                    #with open(m_path,'r') as fp:
                                        #m_vals = np.array(eval(fp.read().replace("nan","np.nan"))).astype(float)
                                    m_diff = np.absolute(m_vals - true_vals)
                                    
                                    # Update values
                                    
                                    method_avg_error += m_diff
                                    i += 1
                                except:
                                    print("Failure for patch: " + str(m_path))
                        
    # Compute avg, add to dict  
    method_avg_error = method_avg_error / i
    local_y = np.zeros(10)
    for b in range(0,method_avg_error.shape[-1]):
        local_y[b] = np.nanmean(method_avg_error[:,:,b])
    method_y[method] = local_y
    
    
x = np.zeros(10)
for k,v in band_reflectance.items():
    x[k] = v

# Save raw values

s = "method,"
for i in range(0,len(x)):
    s += str(x[i]) + ","
s += "\n"

for method,ys in method_y.items():
    s += method + ","
    for i in range(0,len(ys)):
        s += str(ys[i]) + ","
    s += "\n"

with open(save_dir + "raw_values.csv","w") as fp:
    fp.write(s)
    
# Plot results and save

for method,ys in method_y.items():
    plt.plot(x,ys,label=method,marker=method_markers[method])
plt.xlabel("Reflectance value")
plt.ylabel("Mean absolute error")
plt.title("Error per reflectance wavelength")
plt.legend()
plt.savefig(save_dir + "wavelength_error_plot" + scene_name_extra + ".pdf",bbox_inches='tight')