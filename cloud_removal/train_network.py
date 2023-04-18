import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling # https://rasterio.readthedocs.io/en/latest/topics/resampling.html
from rasterio.windows import Window # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

from VPint.utils.hide_spatial_data import hide_values_sim_cloud

import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.optimizers import Nadam

# Heavily based on Meraner's paper/implementation, but made applicable to our use case

def ResBlock(input_original,num_kernels,kernel_size,scale=0.1):
    res = layers.Conv2D(num_kernels, kernel_size, kernel_initializer='he_uniform', padding='same')(input_original)
    res = layers.Activation('relu')(res)
    res = layers.Conv2D(num_kernels, kernel_size, kernel_initializer='he_uniform', padding='same')(res)

    res = layers.Lambda(lambda x: x * scale)(res) # Res scaling layer from paper
    
    res = Add()([input_original,res])

    return(res)


def multi_input_model(input_shapes,num_kernels,num_res_blocks=1,mask=False):

    input_x = layers.Input(shape=input_shapes[0])
    input_f = layers.Input(shape=input_shapes[1])
    if(mask):
        input_m = layers.Input(shape=input_shapes[2]) # Mask

    # authors seem to concat rows
    if(mask):
        x = layers.Concatenate(axis=-1)([input_x,input_f,input_m]) # Mask
    else:
        x = layers.Concatenate(axis=-1)([input_x,input_f]) # No mask
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_kernels,(3,3),kernel_initializer="he_uniform",padding="same")(x)
    x = layers.Activation('relu')(x)
    
    for i in range(0,num_res_blocks):
        x = ResBlock(x, num_kernels, kernel_size=[3, 3])

    x = layers.Conv2D(input_shapes[0][2], (3, 3), kernel_initializer='he_uniform', padding='same')(x) # we have bands last

    # Not in the original model, but no idea how their definition got the shapes right at this point
    #if(mask):
    #    x = layers.MaxPooling2D(pool_size=(3,1))(x) # Mask
    #else:
    #    x = layers.MaxPooling2D(pool_size=(2,1))(x) # No mask
    
    # (long) skip connection from original image
    x = Add()([x, input_x])

    if(mask):
        model = Model(inputs=[input_x,input_f,input_m],outputs=x) # Mask
    else:
        model = Model(inputs=[input_x,input_f],outputs=x) # No mask
    return(model)
    
    
def cloud_only_mae(true,pred):
    # Compute MAE only on cloudy pixels. True values have to be INVERTED
    # compared to normal setup, i.e., NON-cloudy pixels are NaNs
    diff = tf.math.abs(tf.subtract(true,pred))
    #test = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
    no_nans = tf.where(tf.math.is_nan(diff), tf.zeros_like(diff), diff)
    mean = tf.math.reduce_mean(no_nans)
    return(mean)
    
    
    
# Full training set

#bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09"]
scenes = ["1","2","3","4","5"]
base_path = "/scratch/arp/data/SEN2-MSI-T/train/"

upscale_factor = {
    "B01":1/3,
    "B02":1,
    "B03":1,
    "B04":1,
    "B05":1,
    "B06":1,
    "B07":1,
    "B08":2,
    "B8A":1,
    "B09":1/3,
    "B11":1,
    "B12":1,
}

scene_height = 10000 #  Hacky
scene_width = 10000
target_res = 256

# Hyperparameters from Meraner setup

training_epochs = 8
batch_size = 16
learning_rate = 7e-5
optimiser = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004)

# Create model

num_kernels = 256
num_res_blocks = 32

input_shapes = [(target_res,target_res,len(bands)),(target_res,target_res,len(bands))]
model = multi_input_model(input_shapes,num_kernels,num_res_blocks=num_res_blocks)
#model.compile(optimizer=optimiser,loss=cloud_only_mae)
model.compile(optimizer=optimiser,loss='mae')

# Initialise batch

batch_t = np.zeros((batch_size,target_res,target_res,len(bands)))
batch_f = np.zeros((batch_size,target_res,target_res,len(bands)))
batch_i = 0

# Iterate over scenes
for scene in scenes:
    # Iterate over patches
    print(scene)
    for i in range(0,int(scene_height/target_res)): 
        
        if(i*target_res + target_res > scene_height):
            break
            
        for j in range(0,int(scene_width/target_res)):
            # Combine all bands
            for b in range(0,len(bands)):
                path_t = base_path + scene + "_t/bands/" + bands[b] + ".jp2"
                path_f = base_path + scene + "_f/bands/" + bands[b] + ".jp2"
                
                with rasterio.open(path_t) as dataset_t, rasterio.open(path_f) as dataset_f:
                    
                    # Bit hacky, but done because I don't have access to these earlier
                    scene_height = dataset_t.height
                    scene_width = dataset_t.width
                    
                    if(j*target_res + target_res > scene_width):
                        break                        
                    
                    # Windowed read + resample (if necessary)                    
                    data_t = dataset_t.read(
                        out_shape=(
                            dataset_t.count,
                            target_res,
                            target_res
                        ),
                        resampling=Resampling.bilinear,
                        masked=True,
                        window=Window(i*target_res*upscale_factor[bands[b]], j*target_res*upscale_factor[bands[b]], # offset
                                      target_res*upscale_factor[bands[b]], target_res*upscale_factor[bands[b]]) # size
                    )[0,:,:]
                    
                    data_f = dataset_f.read(
                        out_shape=(
                            dataset_t.count,
                            target_res,
                            target_res
                        ),
                        resampling=Resampling.bilinear,
                        masked=True,
                        window=Window(i*target_res*upscale_factor[bands[b]], j*target_res*upscale_factor[bands[b]], # offset
                                      target_res*upscale_factor[bands[b]], target_res*upscale_factor[bands[b]]) # size
                    )[0,:,:]
                    
                    
                    # Set tensors
                    
                    #data_t = clip_and_normalise(data_t,0,10000,-1,1)
                    #data_f = clip_and_normalise(data_f,0,10000,-1,1)
                    
                    batch_t[batch_i,:,:,b] = data_t
                    batch_f[batch_i,:,:,b] = data_f
                    
                    batch_i += 1
                    
                    if(batch_i >= batch_size-1):
                        print("New batch")
                        # Create input with simulated clouds
                        
                        radius = int((target_res+target_res)/2)
                        num_traj = int((target_res+target_res)/2 / 2)
                        
                        input_batch = batch_t.copy()
                        for img_num in range(0,batch_size):
                            # Keep adding more until 50% of the image consists of clouds
                            num_nan = 0
                            while(num_nan < (0.5*np.product(input_batch[img_num,:,:,0].shape))):
                                input_batch[img_num,:,:,0] = hide_values_sim_cloud(
                                        input_batch[img_num,:,:,0],1,radius,num_traj=10)
                                num_nan = np.count_nonzero(np.isnan(input_batch[img_num,:,:,0]))
                            for batch_band in range(0,len(bands)):
                                b1 = input_batch[img_num,:,:,0]
                                bn = input_batch[img_num,:,:,batch_band]
                                shp = b1.shape
                                b1_vec = b1.reshape(np.product(shp))
                                bn_vec = bn.reshape(np.product(shp))
                                bn_vec[np.isnan(b1_vec)] = np.nan
                                input_batch[img_num,:,:,batch_band] = bn_vec.reshape(shp)

                        # Set NON-cloudy pixels in ground truth to nan
                        # (needed for custom loss function, now commented)
                        #shp = batch_t.shape
                        #size = np.product(shp)
                        #batch_t_flat = batch_t.reshape(size)
                        #batch_in_flat = input_batch.reshape(size)
                        #batch_t_flat[~np.isnan(batch_in_flat)] = np.nan
                        #batch_t = batch_t_flat.reshape(shp)
                        
                        # Set simulated clouds to highest values
                        input_batch = np.nan_to_num(input_batch,nan=np.nanmax(input_batch))
                        #input_batch = batch_t.copy() # TODO: DEBUGGING, REMOVE LATER
                            
                        # Train on batch
                        
                        hist = model.fit([input_batch,batch_f],batch_t,epochs=training_epochs)
                        
                        # Reset counter
                        batch_i = 0
                        
                        
save_path = "/scratch/arp/results/models/meraner_adaptation_50percent"
model.save(save_path)