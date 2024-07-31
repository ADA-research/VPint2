import numpy as np
import matplotlib.pyplot as plt
import pickle

def normalise_and_visualise(img, title="", rgb=[3,2,1],percentile_clip=True, save_fig=False,**kwargs):
    
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
    
    
   
with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
scenes = list(legend.keys())
features = ["1w", "1m", "3m"]

for scene in scenes:
    for feature in features:
        #try:
        img = np.load("/scratch/arp/results/cloud/results_full/" + scene + "/VPint/" + feature + ".npy")

        normalise_and_visualise(img, save_fig=True, path="/scratch/arp/results/cloud/results_full/" + scene + "/VPint/" + feature + ".pdf")
            
        #    print("success for ", scene, feature)
        #except:
        #    pass