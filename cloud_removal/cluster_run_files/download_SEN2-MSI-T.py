import numpy as np
import pandas as pd

import os
import datetime
import pickle
import random
import matplotlib.pyplot as plt

from sentinelsat import SentinelAPI
import sentinelsat


# Parameters

overwrite = False
save_path = "/scratch/arp/data/SEN2-MSI-T/"

retrieve_longterm = True

# Evalscripts and functions



# Load pickled stuff

with open("image_ids.pkl", 'rb') as fp:
    to_download = pickle.load(fp)


# Download products SS

# TODO: just for testing, remove later
#for k,v in to_download.items():
#    to_download = {k:v}
#    break 

api = SentinelAPI(None, None)

# Because product files are stored with default names
keys = ""
if(not(os.path.exists(save_path + "legend.csv")) or overwrite):
    header = "key,file\n"
    with open(save_path + "legend.csv", "w") as fp:
        fp.write(header)
        
# Retrieve list of long-term products that need to be downloaded
long_term = []
if(os.path.exists(save_path + "long_term.pkl")):
    with open(save_path + "long_term.pkl", 'rb') as fp:
        long_term = pickle.load(fp)
        
print(long_term)
        
# Iterate over long term product list, see if they can be DLed now
long_term2 = []
if(retrieve_longterm):
    for prod in long_term:
        try:
            print("Attempting LT download")
            api.download(prod, directory_path=save_path)
        except sentinelsat.exceptions.LTATriggered as e:
            print("Failed LT download")
            print(e)
            long_term2.append(prod)
    long_term = long_term2

for location, scenes in to_download.items():
    for feature, product_id in scenes.items():
            
        if(not(os.path.exists(product_id + ".zip")) or overwrite):
            key = location + "-" + feature
            products = api.query(identifier=product_id) # SSat doesn't use names as ids
            iden = None
            
            for k,v in products.items():
                iden = k
                break # should be unnecessary since it's only one product (iterating to access first item)
            # Handle exception for long-term access (needs to be DLed later)
            if(not(iden in long_term)):
                try:
                    api.download(iden, directory_path=save_path)
                    keys += key + "," + product_id + "\n"
                    print("Successfully downloaded product " + product_id + " for scene:\n" + key + "\n\n")
                except sentinelsat.exceptions.LTATriggered:
                    long_term.append(iden)
                    keys += key + "," + product_id + "\n" # Do this here in advance, more difficult later
                    print("Appending " + product_id + " for scene:\n" + key + "\nto long term retrieval list.\n\n")
                except sentinelsat.exceptions.LTAError:
                    print("Exceeded offline retrieval quota, needs to be initialised again later.")
                    break
                
with open(save_path + "long_term.pkl", 'wb') as fp:
    pickle.dump(long_term, fp)

with open(save_path + "legend.csv", "a") as fp:
    fp.write(keys)
    
print("Finished running. ")