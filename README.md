# VPInt2
This repository contains code for VPint2. VPint2 extends the VPint (Value Propagation interpolation) algorithm with identity priority and elastic band resistance. These extensions were primarily intended to improve performance on remote sensing tasks.

On a personal note, please note that this project ran for a long time, from the start of my PhD, and went through many iterations. There are many coding style points I would do differently were I to start now, and the code got quite bloated and its organisation became a little awkward for the approach it grew into. I would still, ideally, like to update all of the VPint2 code Someday(tm). The updated code can then be found on https://github.com/LaurensArp/VPint. I will keep the code here as-is, to maximise reproducibility of our paper. If you are trying to reproduce our results and find that something required for that is missing, please let me know! It is possible that I forgot to include some file or another, which would be easy to fix, or messed up some versioning somewhere along the line, particularly from the start of my PhD when I was still learning about project management.

# Cloud Removal

The extensions were targeted at an application in cloud removal from Sentinel-2 optical satellite imagery. The notebooks and code for this application, which is the main focus of our paper, can be found in the cloud_removal directory.

# Instructions

I will add detailed instructions when I can; for now I am focusing on getting the code for the paper online ASAP for reproducibility. As a minimal example to run VPint2 on a Sentinel-2 image (channels last):

```
# input_img contains the cloudy input image, with cloudy pixels masked and denoted as np.nan
# feature_img contains the cloud-free reference image

pred = input_img.copy()
feature_img = np.clip(feature_img, 0, 10000) # Common preprocessing step
for b in range(0, input_img.shape[2]): 
	MRP = WP_SMRP(input_img[:,:,b], feature_img[:,:,b])
	pred[:,:,b] = MRP.run(clip_val=10000, prioritise_identity=True)
```
Here input_img is the array containing the input image (h, w, b) with cloudy pixels as np.nan, and feature_img is the cloud-free feature image. Usually the default beta (priority_intensity for prioritise_identity) of 1 is a good setting, but not always. Elastic band resistance varies more strongly, so let's disable that setting on this basic example.

To run with auto-adaptation instead (much greater computational cost, but potentially improved performance):

```
# input_img contains the cloudy input image, with cloudy pixels masked and denoted as np.nan
# feature_img contains the cloud-free reference image

pred = input_img.copy()
feature_img = np.clip(feature_img, 0, 10000) # Common preprocessing step
for b in range(0, input_img.shape[2]): 
	MRP = WP_SMRP(input_img[:,:,b], feature_img[:,:,b])
	pred[:,:,b] = MRP.run(auto_adapt=True, auto_adaptation_epochs=10, auto_adaptation_max_iter=100, auto_adaptation_strategy='random', auto_adaptation_proportion=0.8, clip_val=10000, resistance=True, prioritise_identity=True)
```