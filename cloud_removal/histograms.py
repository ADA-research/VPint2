import numpy as np
import matplotlib.pyplot as plt
import rasterio

import os
import sys
import pickle



def set_bins(r,num_bins,add_extra=False,extra_val='auto'):
    vals_per_bin = (r[1]-r[0]) / num_bins
    if(add_extra):
        bins = np.zeros(num_bins+1)
    else:
        bins = np.zeros(num_bins)
        
    for i in range(0,num_bins):
        bins[i] = r[0] + i*vals_per_bin
        
    if(add_extra):
        if(extra_val=='auto'):
            bins[-1] = bins[-2] * 2
        else:
            bins[-1] = extra_val
            
    return(bins)


# Setup



results_path = "/scratch/arp/results/cloud/results/"
original_path = "/scratch/arp/data/SEN2-MSI-T/"
save_dir = "/scratch/arp/results/cloud/histograms/"

methods = ["VPint","replacement","regression","ARMA"]
scenes = ["dry_australia","flood_brazil","urban_tokyo","vegetation_ukraine","water_baikal"]
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





num_bins = 100
err_max = 0.1
error_range = (-err_max,err_max)
bins = set_bins(error_range,num_bins,add_extra=True,extra_val=10)



for method in methods: # VPint etc
    frq = np.zeros(num_bins)
    save_path = save_dir + method + ".pdf"
    i = 0
    edges = [] # namespace
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
                                mask_path = original_path + "scenes/" + scene + "/" + patchname + "_mask.tif"
                                
                                # Load results + ground truth
                                true_vals = np.moveaxis(rasterio.open(ground_truth_path).read(),0,-1)
                                mask_vals = rasterio.open(mask_path).read()[:,:,0]
                                mask_vec = mask_vals.reshape(np.product(mask_vals.shape))
                                
                                try:
                                    m_diff = None
                                    m_vals = np.load(m_path)
                                    #with open(m_path,'r') as fp:
                                    #    m_vals = np.array(eval(fp.read().replace("nan","np.nan"))).astype(float)
                                    m_diff = np.absolute(m_vals - true_vals).reshape(np.product(m_vals.shape))
                                    m_diff = m_diff[mask_vec>0.3]
                                    frq_local, edges = np.histogram(m_diff,bins=bins)
                                    frq += frq_local
                                    

                                except:
                                    print("Failure for patch: " + str(m_path))
                        
    plt.stairs(frq,edges)
    #plt.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
    plt.title("Absolute error frequencies " + method)
    plt.xlabel("Absolute error")
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.set_xticks((0,int(len(bins)/2),len(bins)-2))
    ax.set_yticks((len(bins)-2,int(len(bins)/2),0))
    ax.set_xticklabels((-err_max,0,err_max))
    ax.set_yticklabels((-err_max,0,err_max))
    plt.savefig(save_path,bbox_inches='tight')
    plt.show()