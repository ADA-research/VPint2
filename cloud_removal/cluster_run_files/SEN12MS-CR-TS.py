# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scisig

import rasterio
from s2cloudless import S2PixelCloudDetector
from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_BANDS, SENTINEL2_DESCRIPTORS, LANDSAT89_BANDS, LANDSAT89_DESCRIPTORS

from VPint.WP_MRP import WP_SMRP

import os
import sys
import random
import pickle
import multiprocessing



def normalise_and_visualise(img, title="", rgb=[3,2,1], percentile_clip=True, save_fig=False, save_path="", **kwargs):
    import matplotlib.pyplot as plt
    new_img = np.zeros((img.shape[0],img.shape[1],3))
    new_img[:,:,0] = img[:,:,rgb[0]]
    new_img[:,:,1] = img[:,:,rgb[1]]
    new_img[:,:,2] = img[:,:,rgb[2]]
    
    if(percentile_clip):
        min_val = np.nanpercentile(new_img, 1)
        max_val = np.nanpercentile(new_img, 99)

        new_img = np.clip((new_img-min_val) / (max_val-min_val), 0, 1)
    
    plt.imshow(new_img, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if(save_fig):
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



# Based on https://github.com/PatrickTUM/UnCRtainTS/blob/main/util/detect_cloudshadow.py for cloud shadows
def get_shadow_mask(data_image):
    data_image = np.moveaxis(data_image, -1, 0)
    data_image = data_image / 10000

    (ch, r, c) = data_image.shape
    shadowmask = np.zeros((r, c)).astype('float32')

    BB     = data_image[1]
    BNIR   = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3/4 # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return(shadowmask)



# Setup SEnSeIv2, based on https://github.com/aliFrancis/SEnSeIv2/blob/main/demo.ipynb

# Set based on your runtime (cpu-only will run but is significantly slower)
DEVICE = 'cpu' #'cpu'/'cuda'

# Pick pre-trained model from https://huggingface.co/aliFrancis/SEnSeIv2
model_name = 'SEnSeIv2-SegFormerB2-S2-ambiguous'
config, weights = get_model_files(model_name)

# Lots of options in the kwargs for different settings
model = CloudMask(config, weights, verbose=False, categorise=False, device=DEVICE)

def SEnSeI_cloud_mask(img, cloud_prob=0.1):
    mask = np.max(model(np.moveaxis(img, -1, 0), descriptors=SENTINEL2_DESCRIPTORS, stride=357)[1:, :, :], axis=0) # Highest probability for 4-class detection (thick, thin, shadow)
    mask[mask > cloud_prob] = 1
    mask[mask < cloud_prob] = 0
    return(mask)


# Main functions



def run_VPint(input_img, features):           
    input_img = np.clip(input_img, 0, 10000)
    pred_grid = input_img.copy()
    feature_grid = np.clip(features, 0, 10000)
    for b in range(0, input_img.shape[2]): 
        MRP = WP_SMRP(input_img[:,:,b], feature_grid[:,:,b])
        pred_grid[:,:,b] = MRP.run(method='exact',
                    auto_adapt=True, auto_adaptation_verbose=False,
                    auto_adaptation_epochs=10, auto_adaptation_max_iter=100,
                    auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, clip_val=10000,
                    resistance=True,prioritise_identity=True)
    return(pred_grid)
    
    
def run_VPint_multi(target_grid, feature_grid):
    # Multiprocessing VPint version code written by Dean van Laar
    
    manager = multiprocessing.Manager()
    pred_dict = manager.dict()
    pred_grid = target_grid.copy()
    grid_combos = []
    bands = []
    procs = len(range(0, target_grid.shape[2]))

    # Create lists containing the bands in order
    for b in range(0, target_grid.shape[2]):
        targetc = target_grid[:, :, b]
        feature = feature_grid[:, :, b]
        band0 = b
        grid_combos.append([targetc, feature])
        bands.append(band0)

    # Start the processes
    jobs = []
    for i in range(0, procs):
        process = multiprocessing.Process(target=VPint_multi_single,
                                          args=(pred_dict, grid_combos[i], bands[i]))
        jobs.append(process)
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

        # Ensure all of the processes have finished
        if j.is_alive():
            pass
            #print("Job is not finished!")

    #Sort the dictionary after running VPint on the bands
    sorted_dict = dict(sorted(pred_dict.items()))
    pred0 = np.array([*sorted_dict.values()])
    pred1 = pred0.swapaxes(0, 2)
    pred_grid_final = pred1.swapaxes(0, 1)
    return(pred_grid_final)
    
    
    
def VPint_multi_single(pred_dict, grids, band, use_IP=True, use_EB=True):
    MRP = WP_SMRP(grids[0], grids[1])
    pred_dict[band] = MRP.run(method='exact',
                                 auto_adapt=True, auto_adaptation_verbose=False,
                                 auto_adaptation_epochs=25, auto_adaptation_max_iter=100,
                                 auto_adaptation_strategy='random', auto_adaptation_proportion=0.8,
                                 resistance=use_EB, prioritise_identity=use_IP)

    
def mask_buffer(mask, size=1, passes=1, threshold=0.1):
    for p in range(0,passes):
        new_mask = mask.copy()
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if(mask[i, j] > threshold):
                    start_y = max(0, i-size)
                    end_y = min(mask.shape[0], i+size)
                    start_x = max(0, j-size)
                    end_x = min(mask.shape[0], j+size)
                    new_mask[start_y:end_y, start_x:end_x] = 1
                    
        mask = new_mask

    return(new_mask)
    



# Setup

conditions_algorithms = [
    #"VPint", 
    "VPint_multi",
]

save_path = sys.argv[2]
dataset_path = sys.argv[1]

overwrite = False # Set to True to overwrite existing runs
overwrite_fails = False
size_y = 256
size_x = 256

buffer_clouds = False
buffer_clouds_size = 3
buffer_clouds_passes = 1
buffer_clouds_threshold = 0.1

cloud_mask_method = "SEnSeIv2" # "s2cloudless", "SEnSeIv2"

conditions_rois = {
    "s2_africa_test": ["ROIs2017"],
    "s2_america_test": ["ROIs1158", "ROIs1970"],
    "s2_asiaEast_test": ["ROIs1868", "ROIs1970"],
    "s2_asiaWest_test": ["ROIs1868"],
    "s2_europa_test": ["ROIs1868", "ROIs1970", "ROIs2017"],
}


max_cloudy_ground_truth = 0.01
max_cloudy_feature = 0.05
max_cloudy_target = 0.90
min_cloudy_target = 0.10

max_cloudy_ground_truth_num = max_cloudy_ground_truth * (256*256)
max_cloudy_feature_num = max_cloudy_feature * (256*256)
max_cloudy_target_num = max_cloudy_target * (256*256)
min_cloudy_target_num = min_cloudy_target * (256*256)

cloud_threshold = 0.4

cloud_detector = S2PixelCloudDetector(threshold=cloud_threshold, average_over=4, dilation_size=2, all_bands=True)

       
# Assign this run to unique combination of conditions
i = 1
conds = {}

for geo, roi_list in conditions_rois.items():
    for roi in roi_list:
        for alg in conditions_algorithms:
            conds[i] = {"algorithm":alg, "roi":roi, "geo":geo}
            i += 1
            
this_run_cond = conds[int(sys.argv[3])]





# Create result dir
currdir = save_path + "results_SEN12MS_CR_TS"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
    
# Create directory per algorithm
currdir = currdir + "/" + this_run_cond["algorithm"]
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass

# Create directory per geo
currdir = currdir + "/" + this_run_cond["geo"]
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass


# Create directory per roi
currdir = currdir + "/" + this_run_cond["roi"]
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass





# Prepare patches etc for this job (this geo+roi combination)

roi_path = dataset_path + this_run_cond["geo"] + "/" + this_run_cond["roi"] + "/" 
# Not actually sure what these are, but some rois have more dirs inside than just 100, so need to include
roi_nums = os.listdir(roi_path)
for roi_num in roi_nums:

    # Create directory per roi_num
    save_path = currdir + "/" + str(roi_num)
    if(not(os.path.exists(save_path))):
        try:
            os.mkdir(save_path)
        except:
            pass

    roi_path2 = roi_path + str(roi_num) + "/S2/"
    patches = [a.split("_")[-1].split(".")[0] for a in os.listdir(roi_path2 + "0/")]
    random.shuffle(patches) # Allows for multiple tasks in parallel
    #patches = ["210"] # TODO: remove, just debugging
    
    # Setup failed runs
    roi_key = this_run_cond["geo"] + "_" + this_run_cond["roi"] + "_" + str(roi_num)
    exp_dir = "experiments_meta/"
    if(not(os.path.exists(exp_dir))):
        try:
            os.mkdir(exp_dir)
        except:
            pass
    existing_path = exp_dir + "existing_fails_" + roi_key + ".csv"
    if(overwrite_fails or not(os.path.exists(existing_path))):
        try:
            with open(existing_path, 'w') as fp:
                fp.write("patch\n")
        except:
            pass
            
    # Setup file to keep track of which image is the ground truth
    roi_key = this_run_cond["geo"] + "_" + this_run_cond["roi"] + "_" + str(roi_num)
    results_gt_path = exp_dir + "results_gt_" + roi_key + ".csv"
    if(overwrite_fails or not(os.path.exists(results_gt_path))):
        try:
            with open(results_gt_path, 'w') as fp:
                fp.write("patch,index\n")
        except:
            pass

    # Iterate over patches
    for patch in patches:
        #print("patch: " + patch, flush=True)
        save_path2 = save_path + "/" + patch + ".npy"
        
        # Don't run on existing fails. Extra I/O better than computing 30 extra cloud masks
        existing_fails = pd.read_csv(existing_path, header=0)['patch'].values
        
        #print("existing:", existing_fails)
        #print("patch:", patch)
        #print("cond:", int(patch) in existing_fails)
        #print("\n")
               
        if(overwrite or not(os.path.exists(save_path2)) and not(int(patch) in existing_fails)):
        
            # Prepare data
            time_steps = [str(i) for i in range(0, 30)]
            imgs = np.zeros((len(time_steps), 256, 256, 13))
            i = 0
            for ts in time_steps:
                ts_path = roi_path2 + ts + "/"
                img_date = os.listdir(roi_path2 + ts + "/")[0].split("_")[5]
                img_path = ts_path + "s2_" + this_run_cond["roi"] + "_" + str(roi_num) + "_ImgNo_" + ts + "_" + img_date + "_patch_" + patch + ".tif"
                with rasterio.open(img_path) as fp:
                    img = np.moveaxis(fp.read(), 0, -1)
                    imgs[i, :, :, :] = img
                    i += 1


            if(cloud_mask_method == "s2cloudless"):
                # Run s2cloudless for cloud masks
                clm = cloud_detector.get_cloud_masks(imgs/10000)
                
                # Run cloud shadow detection code from Ebel et al
                for i in range(0, clm.shape[0]):
                    clsm = get_shadow_mask(imgs[i, :, :, :]) * -1 # multiply to turn -1 to 1 to match clouds
                    clm[i, :, :] = np.maximum(clm[i, :, :], clsm)
                #plt.imshow(clm[0, :, :])
                #plt.title("Cloud+Shadow")
                #plt.colorbar()
                #plt.savefig("vis_clm2.pdf")
                #plt.show()
            
            elif(cloud_mask_method == "SEnSeIv2"):
                # Run SEnSeIv2 for cloud masks
                clm = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
                for ts in range(0, imgs.shape[0]):
                    clm[ts, :, :] = SEnSeI_cloud_mask(imgs[ts, :, :, :]/10000)
                
            else:
                raise NotImplementedError
            
            
            #ind = np.random.randint(low=0, high=clm.shape[0])
            #import matplotlib.pyplot as plt
            #plt.imshow(clm[ind, :, :])
            #plt.title("Cloud")
            #plt.colorbar()
            #plt.savefig("vis_clm1.pdf")
            #plt.show()
            
            #normalise_and_visualise(imgs[ind, :, :, :], save_fig=True, save_path="vis_clm1_img.pdf")
            #skjdflk
            
            #num_viable_gt = 0
            #num_viable_target = 0
            #num_viable_feature = 0
            #for ts in range(0, clm.shape[0]):
            #    nc = np.sum(clm[ts, :, :])
            #    if(nc < max_cloudy_ground_truth_num):
            #        num_viable_gt += 1
            #    if(num_viable_gt > 0 and nc < max_cloudy_feature_num):
            #        num_viable_feature += 1
            #    if(nc < max_cloudy_target_num and nc > min_cloudy_target_num):
            #        num_viable_target += 1
            #print("\nNew instance", flush=True)
            #print("Num viable ground truth:", num_viable_gt, flush=True)
            #print("Num viable target:", num_viable_target, flush=True)
            #print("Num viable feature:", num_viable_feature, flush=True)

            # Buffer clouds
            if(buffer_clouds):
                for i in range(0, clm.shape[0]):
                    clm[i, :, :] = mask_buffer(clm[i, :, :], size=buffer_clouds_size, passes=buffer_clouds_passes, threshold=buffer_clouds_threshold)
            #plt.imshow(clm[0, :, :])
            #plt.title("Buffered")
            #plt.colorbar()
            #plt.savefig("vis_clm3.pdf")
            #plt.show()
            
            #print(np.sum(clm, axis=(1,2)))
            #sldkfj
            
            
            viable = False # Set to True at the end if suitable stuff found      
            
            # Identify time steps
            
            ground_truth_index = -1
            for i in range(0, clm.shape[0]-2): # -2 at least because we need target and feature image
                n = np.sum(clm[i, :, :])
                if(n < max_cloudy_ground_truth_num):
                    n2 = np.sum(clm[i+1, :, :])
                    if(n2 < max_cloudy_target_num and n2 > min_cloudy_target_num):
                        ground_truth_index = i
                        ground_truth = imgs[ground_truth_index, :, :, :]
                        target_index = i+1
                        target = imgs[target_index, :, :, :]
                        break
            

            #ground_truth_index = -1 # TODO: remove (debugging) and uncomment above

            if(ground_truth_index > -1 and target_index > -1):

                #target_index = -1
                #for i in range(ground_truth_index, clm.shape[0]):
                #    n = np.sum(clm[i, :, :])
                #    if(n > min_cloudy_target_num and n < max_cloudy_target_num):
                #        target_index = i
                #        break
                #target = imgs[target_index, :, :, :]

                #if(target_index > -1):

                feature_index = -1
                for i in range(target_index, clm.shape[0]):
                    n = np.sum(clm[i, :, :])
                    if(n < max_cloudy_feature_num):
                        feature_index = i
                        break
                features = imgs[feature_index, :, :, :]

                if(feature_index > -1):  

                    viable = True
                
                    # Build input image with nan missing values based on mask
                    input_img = target.copy()
                    for y in range(0, input_img.shape[0]):
                        for x in range(0, input_img.shape[1]):
                            if(clm[target_index, y, x] >= 1):
                                input_img[y, x, :] = np.ones(input_img.shape[2]) * np.nan

                    # Run VPint
                    if(this_run_cond["algorithm"] == "VPint"):
                        pred_grid = run_VPint(input_img, features)
                    elif(this_run_cond["algorithm"] == "VPint_multi"):
                        pred_grid = run_VPint_multi(input_img, features)
                    else:
                        raise NotImplementedError
                           
                    np.save(save_path2, pred_grid)
                    with open(results_gt_path, 'a') as fp:
                        fp.write(patch + "," + str(ground_truth_index) + "\n")
                    print("Saved result to: " + save_path2, flush=True)
                    #lkjlksdj
                    
            if(not(viable)):
                with open(existing_path, 'a') as fp:
                    fp.write(patch + "\n")
                    print("Failed and saved to file", flush=True)
                            
print("Terminated successfully")