# VPInt2
This repository contains code for VPint2. VPint2 extends the VPint (Value Propagation interpolation) algorithm with identity priority and elastic band resistance. These extensions were primarily intended to improve performance on remote sensing tasks.

On a personal note, please note that this project ran for a long time, from the start of my PhD, and went through many iterations. There are many coding style points I would do differently were I to start now, and the code got quite bloated and its organisation became a little awkward for the approach it grew into. I would still, ideally, like to update all of the VPint2 code Someday(tm). The updated code can then be found on https://github.com/LaurensArp/VPint. I will keep the code here as-is, to maximise reproducibility of our paper. If you are trying to reproduce our results and find that something required for that is missing, please let me know! It is possible that I forgot to include some file or another, which would be easy to fix, or messed up some versioning somewhere along the line, particularly from the start of my PhD when I was still learning about project management.

# Cloud Removal

The extensions were targeted at an application in cloud removal from Sentinel-2 optical satellite imagery. The notebooks and code for this application, which is the main focus of our paper, can be found in the cloud_removal directory.

# Instructions

I will add instructions when I can; for now I am focusing on getting the code for the paper online ASAP for reproducibility.