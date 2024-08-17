# VPInt2
This repository contains code for VPint2, accompanying our ISPRS Journal of Photogrammetry and Remote Sensing paper (https://doi.org/10.1016/j.isprsjprs.2024.07.030). VPint2 extends the VPint (Value Propagation interpolation) algorithm with identity priority and elastic band resistance. These extensions were primarily intended to improve performance on remote sensing tasks. The notebooks and code for this application, which is the main focus of our paper, can be found in the cloud_removal directory.

# Instructions: using latest VPint2 (recommended)

VPint2 is being maintained at https://github.com/LaurensArp/VPint. At the time of writing, the largest difference with the code here is a wrapper to make the method easier to use with EO data. For a brief introduction to using the method to remove clouds from Sentinel-2 imagery, please see this tutorial: https://adaresearch.wordpress.com/2024/08/07/removing-clouds-from-optical-earth-observation-imagery-using-vpint2-in-2-lines-of-python-code/

# Instructions: using the code in this repository (intended for reproducibility mainly)

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

The code to run the experiments in the paper in contained in the cloud_removal/cluster_run_files folder (take care to use the appropriate command line arguments).

On a personal note, please note that this project ran for a long time, from the start of my PhD, and went through many iterations. There are many coding style points I would do differently were I to start now, and the code got quite bloated and its organisation became a little awkward for the approach it grew into. I would still, ideally, like to update all of the VPint2 code Someday(tm). The updated code can then be found on https://github.com/LaurensArp/VPint. I will keep the code here as-is, to maximise reproducibility of our paper.

# Citation

If you find this work useful in your publications, please see our paper and cite using:

```
@article{ArpEtAl24,
    author = {Laurens Arp and Holger Hoos and Peter {van Bodegom} and Alistair Francis and James Wheeler and Dean {van Laar} and Mitra Baratchi},
    title = {Training-free thick cloud removal for Sentinel-2 imagery using value propagation interpolation},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {216},
    pages = {168-184},
    year = {2024},
    issn = {0924-2716},
    doi = "https://doi.org/10.1016/j.isprsjprs.2024.07.030",
    url = "https://www.sciencedirect.com/science/article/pii/S0924271624002995",
    tags = {Remote sensing, spatial interpolation, cloud removal},
}
```