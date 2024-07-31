# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import rasterio.features
from pyproj import Transformer

import satellite_cloud_generator as scg

#from VPint.WP_MRP import WP_SMRP

import os
import sys
import random
import pickle
import time

#######################################################
# Code from Meraner et al's GitHub (minor adaptations)
#######################################################

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
from tensorflow.keras.models import Model#, Input
from tensorflow.keras.layers import Input

K.set_image_data_format('channels_first')


def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Definition of Residual Block to be repeated in body of network."""
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)

    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([input_l, tmp])


def DSen2CR_model(input_shape,
                  batch_per_gpu=2,
                  num_layers=32,
                  feature_size=256,
                  use_cloud_mask=True,
                  include_sar_input=True):
    """Definition of network structure. """

    global shape_n

    # define dimensions
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])

    if include_sar_input:
        x = Concatenate(axis=1)([input_opt, input_sar])
    else:
        x = input_opt

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)

    # main body of network as succession of resblocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])

    # One more convolution
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)

    # Add first layer (long skip connection)
    x = Add()([x, input_opt])

    if use_cloud_mask:
        # the hacky trick with global variables and with lambda functions is needed to avoid errors when
        # pickle saving the model. Tensors are not pickable.
        # This way, the Lambda function has no special arguments and is "encapsulated"

        shape_n = tf.shape(input_opt)

        def concatenate_array(x):
            global shape_n
            return K.concatenate([x, K.zeros(shape=(batch_per_gpu, 1, shape_n[2], shape_n[3]))], axis=1)

        x = Concatenate(axis=1)([x, input_opt])

        x = Lambda(concatenate_array)(x)

    model = Model(inputs=[input_opt, input_sar], outputs=x)

    return model


#######################################################
# Code from Ebel et al's GitHub (minor adaptations)
#######################################################

import torch
import torch.nn as nn

import copy

import argparse
import json
from datetime import datetime
import datetime as dt


def create_parser(mode='train'):
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument(
        "--model",
        default='uncrtaints', # e.g. 'unet', 'utae', 'uncrtaints',
        type=str,
        help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
    )
    parser.add_argument("--experiment_name", default='my_first_experiment', help="Name of the current experiment",)

    # fast switching between default arguments, depending on train versus test mode
    if mode=='train':
        parser.add_argument("--res_dir", default="./results", help="Path to where the results are stored, e.g. ./results for training or ./inference for testing",)
        parser.add_argument("--plot_every", default=-1, type=int, help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable")
        parser.add_argument("--export_every", default=-1, type=int, help="Interval (in items) of exporting data at validation or test time. Set -1 to disable")
        parser.add_argument("--resume_at", default=0, type=int, help="Epoch to resume training from (may re-weight --lr in the optimizer) or epoch to load checkpoint from at test time")
    elif mode=='test':
        parser.add_argument("--res_dir", default="./inference", type=str, help="Path to directory where results are written.")
        parser.add_argument("--plot_every", default=-1, type=int, help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable")
        parser.add_argument("--export_every", default=1, type=int, help="Interval (in items) of exporting data at validation or test time. Set -1 to disable")
        parser.add_argument("--resume_at", default=-1, type=int, help="Epoch to load checkpoint from and run testing with (use -1 for best on validation split)")

    parser.add_argument("--encoder_widths", default="[128]", type=str, help="e.g. [64,64,64,128] for U-TAE or [128] for UnCRtainTS")
    parser.add_argument("--decoder_widths", default="[128,128,128,128,128]", type=str, help="e.g. [64,64,64,128] for U-TAE or [128,128,128,128,128] for UnCRtainTS")
    parser.add_argument("--out_conv", default=f"[{13}]", help="output CONV, note: if inserting another layer then consider treating normalizations separately")
    parser.add_argument("--mean_nonLinearity", dest="mean_nonLinearity", action="store_false", help="whether to apply a sigmoidal output nonlinearity to the mean prediction") 
    parser.add_argument("--var_nonLinearity", default="softplus", type=str, help="how to squash the network's variance outputs [relu | softplus | elu ]")
    parser.add_argument("--agg_mode", default="att_group", type=str, help="type of temporal aggregation in L-TAE module")
    parser.add_argument("--encoder_norm", default="group", type=str, help="e.g. 'group' (when using many channels) or 'instance' (for few channels)")
    parser.add_argument("--decoder_norm", default="batch", type=str, help="e.g. 'group' (when using many channels) or 'instance' (for few channels)")
    parser.add_argument("--block_type", default="mbconv", type=str, help="type of CONV block to use [residual | mbconv]")
    parser.add_argument("--padding_mode", default="reflect", type=str)
    parser.add_argument("--pad_value", default=0, type=float)

    # attention-specific parameters
    parser.add_argument("--n_head", default=16, type=int, help="default value of 16, 4 for debugging")
    parser.add_argument("--d_model", default=256, type=int, help="layers in L-TAE, default value of 256")
    parser.add_argument("--positional_encoding", dest="positional_encoding", action="store_false", help="whether to use positional encoding or not") 
    parser.add_argument("--d_k", default=4, type=int)
    parser.add_argument("--low_res_size", default=32, type=int, help="resolution to downsample to")
    parser.add_argument("--use_v", dest="use_v", action="store_true", help="whether to use values v or not")

    # set-up parameters
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loading workers")
    parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
    parser.add_argument("--device",default="cuda",type=str,help="Name of device to use for tensor computations (cuda/cpu)",)
    parser.add_argument("--display_step", default=10, type=int, help="Interval in batches between display of training metrics",)

    # training parameters
    parser.add_argument("--loss", default="MGNLL", type=str, help="Image reconstruction loss to utilize [l1|l2|GNLL|MGNLL].")
    parser.add_argument("--resume_from", dest="resume_from", action="store_true", help="resume training acc. to JSON in --experiment_name and *.pth chckp in --trained_checkp")
    parser.add_argument("--unfreeze_after", default=0, type=int, help="When to unfreeze ALL weights for training")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--chunk_size", type=int, help="Size of vmap batches, this can be adjusted to accommodate for additional memory needs")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate, e.g. 0.01")
    parser.add_argument("--gamma", default=1.0, type=float, help="Learning rate decay parameter for scheduler")
    parser.add_argument("--val_every", default=1, type=int, help="Interval in epochs between two validation steps.")
    parser.add_argument("--val_after", default=0, type=int, help="Do validation only after that many epochs.")

    # flags specific to SEN12MS-CR and SEN12MS-CR-TS
    parser.add_argument("--use_sar", dest="use_sar", action="store_true", help="whether to use SAR or not")
    parser.add_argument("--pretrain", dest="pretrain", action="store_true", help="whether to perform pretraining on SEN12MS-CR or training on SEN12MS-CR-TS") 
    parser.add_argument("--input_t", default=3, type=int, help="number of input time points to sample, unet3d needs at least 4 time points")
    parser.add_argument("--ref_date", default="2014-04-03", type=str, help="reference date for Sentinel observations")
    parser.add_argument("--sample_type", default="cloudy_cloudfree", type=str, help="type of samples returned [cloudy_cloudfree | generic]")
    parser.add_argument("--vary_samples", dest="vary_samples", action="store_false", help="whether to sample different time points across epochs or not") 
    parser.add_argument("--min_cov", default=0.0, type=float, help="The minimum cloud coverage to accept per input sample at train time. Gets overwritten by --vary_samples")
    parser.add_argument("--max_cov", default=1.0, type=float, help="The maximum cloud coverage to accept per input sample at train time. Gets overwritten by --vary_samples")
    parser.add_argument("--root1", default='/home/data/SEN12MSCRTS', type=str, help="path to your copy of SEN12MS-CR-TS")
    parser.add_argument("--root2", default='/home/data/SEN12MSCRTS', type=str, help="path to your copy of SEN12MS-CR-TS validation & test splits")
    parser.add_argument("--root3", default='/home/data/SEN12MSCR', type=str, help="path to your copy of SEN12MS-CR for pretraining")
    parser.add_argument("--precomputed", default='/home/code/UnCRtainTS/util/precomputed', type=str, help="path to pre-computed cloud statistics")
    parser.add_argument("--region", default="all", type=str, help="region to (sub-)sample ROI from [all|africa|america|asiaEast|asiaWest|europa]")
    parser.add_argument("--max_samples_count", default=int(1e9), type=int, help="count of data (sub-)samples to take")
    parser.add_argument("--max_samples_frac", default=1.0, type=float, help="fraction of data (sub-)samples to take")
    parser.add_argument("--profile", dest="profile", action="store_true", help="whether to profile code or not") 
    parser.add_argument("--trained_checkp", default="", type=str, help="Path to loading a pre-trained network *.pth file, rather than initializing weights randomly")

    # flags specific to uncertainty modeling
    parser.add_argument("--covmode", default='diag', type=str, help="covariance matrix type [uni|iso|diag].")
    parser.add_argument("--scale_by", default=1.0, type=float, help="rescale data within model, e.g. to [0,10]")
    parser.add_argument("--separate_out", dest="separate_out", action="store_true", help="whether to separately process mean and variance predictions or in a shared layer")

    # flags specific for testing
    parser.add_argument("--weight_folder", type=str, default="./results", help="Path to the main folder containing the pre-trained weights")
    parser.add_argument("--use_custom", dest="use_custom", action="store_true", help="whether to test on individually specified patches or not")
    parser.add_argument("--load_config", default='', type=str, help="path of conf.json file to load")

    return parser

def str2list(config, list_args):
    for k, v in vars(config).items():
        if k in list_args and v is not None and isinstance(v, str):
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    return config


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4]
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16]
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in, use_dropout=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in # e.g. self.d_model in LTAE2d

        # define H x k queries, they are input-independent in LTAE
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        attn_dropout=0.1 if use_dropout else 0.0
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=attn_dropout)

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        # values v are of shapes [B*H*W, T, self.d_in=self.d_model], e.g. [2*32*32=2048 x 4 x 256] (see: sz_b * h * w, seq_len, d)
        # where self.d_in=self.d_model is the output dimension of the FC-projected features  
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4], e.g. Size([32768, 1, 4])
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16], e.g. Size([32768, 4, 16])
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttentionSmall(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head    # e.g. 16
        self.d_k    = d_k       # e.g. 4, number of keys per head
        self.d_in   = d_in      # e.g. 256, self.d_model in LTAE2d

        # define H x k queries, they are input-independent in LTAE
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        """
        # consider using deeper mappings with nonlinearities,
        # but this is somewhat against the original Transformer spirit
        self.fc1_k = nn.Linear(d_in, d_in)
        self.bn2_k = nn.BatchNorm1d(d_in)
        self.fc2_k = nn.Linear(d_in, n_head * d_k)
        self.bn2_k = nn.BatchNorm1d(n_head * d_k)
        """

        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        #nn.init.normal_(self.fc2_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.attention = ScaledDotProductAttentionSmall(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False, weight_v=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        # values v are of shapes [B*H*W, T, self.d_in=self.d_model], e.g. [2*32*32=2048 x 4 x 256] (see: sz_b * h * w, seq_len, d)
        # where self.d_in=self.d_model is the output dimension of the FC-projected features  
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4], e.g. Size([32768, 1, 4])
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16], e.g. Size([32768, 4, 16])
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if weight_v:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)
            if return_comp:
                output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)
        else:
            attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp, weight_v=weight_v)

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        if weight_v:
            output = output.view(n_head, sz_b, 1, d_in // n_head)
            output = output.squeeze(dim=2)

            # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
            #   in utae.py this is torch.Size([h, B, T, 32, 32])
            # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
            #   in utae.py this is torch.Size([B, 128, 32, 32])

            if return_comp:
                return output, attn, comp
            else:
                return output, attn

        return attn


class ScaledDotProductAttentionSmall(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout) # moved dropout after bilinear interpolation
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False, weight_v=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # attn   is of shape [B*H*W*h, 1, T], e.g. [2*32*32*16=32768 x 1 x 4]
        # v      is of shape [B*H*W*h, T, self.d_in/h], e.g. [2*32*32*16=32768 x 4 x 256/16=16]
        # output is of shape [B*H*W*h, 1, h], e.g. [2*32*32*16=32768 x 1 x 16], e.g. Size([32768, 1, 16])
        attn = self.softmax(attn)
        
        """
        # no longer using dropout on attention matrices before the upsampling
        # this is now done after bilinear interpolation only

        attn = self.dropout(attn)
        """

        if weight_v:
            # optionally using the weighted values
            output = torch.matmul(attn, v)

            if return_comp:
                return output, attn, comp
            else:
                return output, attn
        return attn


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        use_dropout=True
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout on the MLP-processed values
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
            use_dropout (bool): dropout on the attention masks.
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, use_dropout=use_dropout
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads, out is now [B*H*W x d_in/h * h], e.g. [2048 x 256]

        # out is of shape [head x b x t x h x w]
        out = self.dropout(self.mlp(out))
        # after MLP, out is of shape [B*H*W x outputLayerOfMLP], e.g. [2048 x 128]
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )

        # out  is of shape [B x outputLayerOfMLP x h x w], e.g. [2, 128, 32, 32]
        # attn is of shape [h x B x T x H x W], e.g. [16, 2, 4, 32, 32]
        if self.return_att:
            return out, attn
        else:
            return out



class LTAE2dtiny(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        d_model=256,
        T=1000,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        This is the tiny version, which stops further processing attention-weighted values v
        (no longer using an MLP) and only returns the attention matrix attn itself
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2dtiny, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttentionSmall(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )


    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp  = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # re-shaped attn to   [h x B*H*W x T], e.g. torch.Size([16, 2048, 4])
        #   in utae.py this is torch.Size([h, B, T, 32, 32])
        # re-shaped output to [h x B*H*W x d_in/h], e.g. torch.Size([16, 2048, 16])
        #   in utae.py this is torch.Size([B, 128, 32, 32])
        attn = self.attention_heads(out, pad_mask=pad_mask)


        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )

        # out  is of shape [B x outputLayerOfMLP x h x w], e.g. [2, 128, 32, 32]
        # attn is of shape [h x B x T x H x W], e.g. [16, 2, 4, 32, 32]
        return attn




class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out

class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3, s=1, p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu: # append a ReLU after the current CONV layer
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2: # only append ReLU if not last layer
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        k=3, s=1, p=1,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            k=k, s=s, p=p,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)
    

def calc_loss(criterion, config, out, y, var=None):
    
    if config.loss in ['GNLL']:
        loss, variance = criterion(out, y, var)
    elif config.loss in ['MGNLL']:
        loss, variance = criterion(out, y, var)
    else: 
        loss, variance = criterion(out, y), None
    return loss, variance

class BaseModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super(BaseModel, self).__init__()
        self.config     = config    # store config
        self.frozen     = False     # no parameters are frozen
        self.len_epoch  = 0         # steps of one epoch

        # temporarily rescale model inputs & outputs by constant factor, e.g. from [0,1] to [0,100],
        #       to deal with numerical imprecision issues closeby 0 magnitude (and their inverses)
        #       --- convert inputs, mean & variance predictions to original scale again after NLL loss is computed
        # note: this may also require adjusting the range of output nonlinearities in the generator network,
        #       i.e. out_mean, out_var and diag_var

        #   -------------- set input via set_input and call forward ---------------
        # inputs self.real_A & self.real_B  set in set_input            by * self.scale_by
        #   ------------------------------ then scale -----------------------------
        # output self.fake_B will automatically get scaled              by ''
        #   ------------------- then compute loss via get_loss_G ------------------
        # output self.netG.variance  will automatically get scaled      by * self.scale_by**2
        #   ----------------------------- then rescale ----------------------------
        # inputs self.real_A & self.real_B  set in set_input            by * 1/self.scale_by
        # output self.fake_B                set in self.forward         by * 1/self.scale_by
        # output self.netG.variance         set in get_loss_G           by * 1/self.scale_by**2
        self.scale_by  = config.scale_by                    # temporarily rescale model inputs by constant factor, e.g. from [0,1] to [0,100]

        # fetch generator
        self.netG = get_generator(self.config)

        # 1 criterion
        self.criterion = get_loss(self.config)
        self.log_vars  = None

        # 2 optimizer: for G
        paramsG = [{'params': self.netG.parameters()}]

        self.optimizer_G = torch.optim.Adam(paramsG, lr=config.lr)

        # 2 scheduler: for G, note: stepping takes place at the end of epoch
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=self.config.gamma)

        self.real_A = None
        self.fake_B = None
        self.real_B = None
        self.dates  = None
        self.masks  = None
        self.netG.variance = None


    def forward(self):
        # forward through generator, note: for val/test splits, 
        # 'with torch.no_grad():' is declared in train script
        self.fake_B = self.netG(self.real_A, batch_positions=self.dates)
        #if self.config.profile: 
        #    flopstats  = FlopCountAnalysis(self.netG, (self.real_A, self.dates))
        #    # print(flop_count_table(flopstats))
        #    # TFLOPS: flopstats.total() *1e-12
        #    # MFLOPS: flopstats.total() *1e-6
        #    # compute MFLOPS per input sample
        #    self.flops = (flopstats.total()*1e-6)/self.config.batch_size
        #    print(f"MFLOP count: {self.flops}")
        self.netG.variance = None # purge earlier variance prediction, re-compute via get_loss_G()

    def backward_G(self):
        # calculate generator loss
        self.get_loss_G()
        self.loss_G.backward()


    def get_loss_G(self):

        if hasattr(self.netG, 'vars_idx'):
            self.loss_G, self.netG.variance = calc_loss(self.criterion, self.config, self.fake_B[:, :, :self.netG.mean_idx, ...], self.real_B, var=self.fake_B[:, :, self.netG.mean_idx:self.netG.vars_idx, ...])
        else: # used with all other models
            self.loss_G, self.netG.variance = calc_loss(self.criterion, self.config, self.fake_B[:, :, :13, ...], self.real_B, var=self.fake_B[:, :, 13:, ...])

    def set_input(self, input):
        self.real_A = self.scale_by * input['A'].to(self.config.device)
        self.real_B = self.scale_by * input['B'].to(self.config.device)
        self.dates  = None if input['dates'] is None else input['dates'].to(self.config.device)
        self.masks  = input['masks'].to(self.config.device)


    def reset_input(self):
        self.real_A = None
        self.real_B = None
        self.dates  = None 
        self.masks  = None
        del self.real_A
        del self.real_B 
        del self.dates
        del self.masks


    def rescale(self):
        # rescale target and mean predictions
        if hasattr(self, 'real_A'): self.real_A = 1/self.scale_by * self.real_A
        self.real_B = 1/self.scale_by * self.real_B 
        self.fake_B = 1/self.scale_by * self.fake_B[:,:,:13,...]
        
        # rescale (co)variances
        if hasattr(self.netG, 'variance') and self.netG.variance is not None:
            self.netG.variance = 1/self.scale_by**2 * self.netG.variance

    def optimize_parameters(self):
        self.forward()
        del self.real_A

        # update G
        self.optimizer_G.zero_grad() 
        self.backward_G()
        self.optimizer_G.step()

        # re-scale inputs, predicted means, predicted variances, etc
        self.rescale()
        # resetting inputs after optimization saves memory
        self.reset_input()

        if self.netG.training: 
            self.fake_B = self.fake_B.cpu()
            if self.netG.variance is not None: self.netG.variance = self.netG.variance.cpu()


def get_norm_layer(out_channels, num_feats, n_groups=4, layer_type='batch'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, n_groups=4):
        super().__init__()
        self.norm = get_norm_layer(dim, dim, n_groups, norm)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class MBConv(TemporallySharedBlock):
    def __init__(self, inp, oup, downsample=False, expansion=4, norm='batch', n_groups=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                          padding=1, padding_mode='reflect', groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride=stride, padding=0, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, padding_mode='reflect',
                          groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm), 
            )
        
        self.conv = PreNorm(inp, self.conv, norm, n_groups=4)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Compact_Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Compact_Temporal_Aggregator, self).__init__()
        self.mode = mode
        # moved dropout from ScaledDotProductAttention to here, applied after upsampling 
        self.attn_dropout = nn.Dropout(0.1) # no dropout via: nn.Dropout(0.0)

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

def get_nonlinearity(mode, eps):
    if mode=='relu':        fct = nn.ReLU() + eps 
    elif mode=='softplus':  fct = lambda vars:nn.Softplus(beta=1, threshold=20)(vars) + eps
    elif mode=='elu':       fct = lambda vars: nn.ELU()(vars) + 1 + eps  
    else:                   fct = nn.Identity()
    return fct


class ResidualConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        n_groups=4,
        #last_relu=True,
        k=3, s=1, p=1,
        padding_mode="reflect",
    ):
        super(ResidualConvBlock, self).__init__(pad_value=pad_value)

        self.conv1 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv3 = ConvLayer(
            nkernels=nkernels,
            #norm='none',
            #last_relu=False,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )

    def forward(self, input):

        out1 = self.conv1(input)        # followed by built-in ReLU & norm
        out2 = self.conv2(out1)         # followed by built-in ReLU & norm
        out3 = input + self.conv3(out2) # omit norm & ReLU
        return out3



class UNCRTAINTS(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[128],
        decoder_widths=[128,128,128,128,128],
        out_conv=[13],
        out_nonlin_mean=False,
        out_nonlin_var='relu',
        agg_mode="att_group",
        encoder_norm="group",
        decoder_norm="batch",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        positional_encoding=True,
        covmode='diag',
        scale_by=1,
        separate_out=False,
        use_v=False,
        block_type='mbconv',
        is_mono=False
    ):
        """
        UnCRtainTS architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
                - none: apply no normalization
            decoder_norm (str): similar to encoder_norm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(UNCRTAINTS, self).__init__()
        self.n_stages       = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.out_widths     = out_conv
        self.is_mono        = is_mono
        self.use_v          = use_v
        self.block_type     = block_type

        self.enc_dim        = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim      = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value      = pad_value
        self.padding_mode   = padding_mode

        self.scale_by       = scale_by
        self.separate_out   = separate_out # define two separate layer streams for mean and variance predictions

        if decoder_widths is not None:
            assert encoder_widths[-1] == decoder_widths[-1]
        else: decoder_widths = encoder_widths


        # ENCODER
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0]],
            k=1, s=1, p=0,
            norm=encoder_norm,
        )

        if self.block_type=='mbconv':
            self.in_block = nn.ModuleList([MBConv(layer, layer, downsample=False, expansion=2, norm=encoder_norm) for layer in encoder_widths])
        elif self.block_type=='residual':
            self.in_block = nn.ModuleList([ResidualConvBlock(nkernels=[layer]+[layer], k=3, s=1, p=1, norm=encoder_norm, n_groups=4) for layer in encoder_widths])
        else: raise NotImplementedError

        if not self.is_mono:
            # LTAE
            if self.use_v:
                # same as standard LTAE, except we don't apply dropout on the low-resolution attention masks
                self.temporal_encoder = LTAE2d(
                    in_channels=encoder_widths[0], 
                    d_model=d_model,
                    n_head=n_head,
                    mlp=[d_model, encoder_widths[0]], # MLP to map v, only used if self.use_v=True
                    return_att=True,
                    d_k=d_k,
                    positional_encoding=positional_encoding,
                    use_dropout=False
                )
                # linearly combine mask-weighted
                v_dim = encoder_widths[0]
                self.include_v = nn.Conv2d(encoder_widths[0]+v_dim, encoder_widths[0], 1)
            else:
                self.temporal_encoder = LTAE2dtiny(
                    in_channels=encoder_widths[0],
                    d_model=d_model,
                    n_head=n_head,
                    d_k=d_k,
                    positional_encoding=positional_encoding,
                )
            
            self.temporal_aggregator = Compact_Temporal_Aggregator(mode=agg_mode)

        if self.block_type=='mbconv':
            self.out_block = nn.ModuleList([MBConv(layer, layer, downsample=False, expansion=2, norm=decoder_norm) for layer in decoder_widths])
        elif self.block_type=='residual':
            self.out_block = nn.ModuleList([ResidualConvBlock(nkernels=[layer]+[layer], k=3, s=1, p=1, norm=decoder_norm, n_groups=4) for layer in decoder_widths])
        else: raise NotImplementedError


        self.covmode = covmode
        if covmode=='uni':
            # batching across channel dimension
            covar_dim = 13
        elif covmode=='iso':
            covar_dim = 1
        elif covmode=='diag':
            covar_dim = 13
        else: covar_dim = 0 

        self.mean_idx = 13
        self.vars_idx = self.mean_idx + covar_dim

        # note: not including normalization layer and ReLU nonlinearity into the final ConvBlock
        #       if inserting >1 layers into out_conv then consider treating normalizations separately
        self.out_dims = out_conv[-1]

        eps = 1e-9 if self.scale_by==1.0 else 1e-3

        if self.separate_out: # define two separate layer streams for mean and variance predictions
            self.out_conv_mean_1 = ConvBlock(nkernels=[decoder_widths[0]] + [13], k=1, s=1, p=0, norm='none', last_relu=False)
            if self.out_dims - self.mean_idx > 0:
                self.out_conv_var_1 = ConvBlock(nkernels=[decoder_widths[0]] + [self.out_dims - 13], k=1, s=1, p=0, norm='none', last_relu=False)
        else: 
            self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, k=1, s=1, p=0, norm='none', last_relu=False)

        # set output nonlinearities
        if out_nonlin_mean: self.out_mean  = lambda vars: self.scale_by * nn.Sigmoid()(vars)    # this is for predicting mean values in [0, 1]
        else: self.out_mean  = nn.Identity()                                                    # just keep the mean estimates, without applying a nonlinearity

        if self.covmode in ['uni', 'iso', 'diag']:
            self.diag_var   = get_nonlinearity(out_nonlin_var, eps)


    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        # SPATIAL ENCODER
        # collect feature maps in list 'feature_maps'
        out = self.in_conv.smart_forward(input)

        for layer in self.in_block:
            out = layer.smart_forward(out)

        if not self.is_mono:
            att_down = 32
            down = nn.AdaptiveMaxPool2d((att_down, att_down))(out.view(out.shape[0] * out.shape[1], *out.shape[2:])).view(out.shape[0], out.shape[1], out.shape[2], att_down, att_down)

            # TEMPORAL ENCODER
            if self.use_v:
                v, att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)
            else:
                att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)

            out = self.temporal_aggregator(out, pad_mask=pad_mask, attn_mask=att)

            if self.use_v:
                # upsample values to input resolution, then linearly combine with attention masks
                up_v = nn.Upsample(size=(out.shape[-2:]), mode="bilinear", align_corners=False)(v)
                out  = self.include_v(torch.cat((out, up_v), dim=1)) 
        else: out = out.squeeze(dim=1)

        # SPATIAL DECODER
        for layer in self.out_block:
            out = layer.smart_forward(out)

        if self.separate_out:
            out_mean_1 = self.out_conv_mean_1(out)

            if self.out_dims - self.mean_idx > 0:
                out_var_1 = self.out_conv_var_1(out)
                out   = torch.cat((out_mean_1, out_var_1), dim=1)
            else: out = out_mean_1 #out = out_mean_2
        else:
            out = self.out_conv(out) # predict mean and var in single layer
        

        # append a singelton temporal dimension such that outputs are [B x T=1 x C x H x W]
        out = out.unsqueeze(dim=1)

        # apply output nonlinearities

        # get mean predictions
        out_loc   = self.out_mean(out[:,:,:self.mean_idx,...])                      # mean predictions in [0,1]
        if not self.covmode: return out_loc

        out_cov = self.diag_var(out[:,:,self.mean_idx:self.vars_idx,...])           # var predictions > 0
        out     = torch.cat((out_loc, out_cov), dim=2)                              # stack mean and var predictions plus cloud masks
        
        return out



def get_loss(config):
    if config.loss=="l2":
        criterion1 = nn.MSELoss()
        criterion = lambda pred, targ: criterion1(pred, targ)
    else: raise NotImplementedError

    # wrap losses
    loss_wrap = lambda *args: args
    loss = loss_wrap(criterion) 
    return loss if not isinstance(loss, tuple) else loss[0]


def get_generator(config):
    if 'uncrtaints' == config.model:
        model = UNCRTAINTS(
                input_dim=2*config.use_sar+13,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths, 
                out_conv=config.out_conv,
                out_nonlin_mean=config.mean_nonLinearity,
                out_nonlin_var=config.var_nonLinearity,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                decoder_norm=config.decoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
                positional_encoding=config.positional_encoding,
                covmode=config.covmode,
                scale_by=config.scale_by,
                separate_out=config.separate_out,
                use_v=config.use_v,
                block_type=config.block_type,
                is_mono=config.pretrain
            )
    else: raise NotImplementedError
    return model


def get_base_model(config):
    model = BaseModel(config)
    return model


def get_model(config):
    return get_base_model(config)


def load_checkpoint(config, checkp_dir, model, name):
    # Code from UnCRtain-TS repository to load model checkpoint
    chckp_path = os.path.join(checkp_dir, config.experiment_name, f"{name}.pth.tar")
    print(f'Loading checkpoint {chckp_path}')
    checkpoint = torch.load(chckp_path, map_location=config.device)["state_dict"]

    try: # try loading checkpoint strictly, all weights & their names must match
        model.load_state_dict(checkpoint, strict=True)
    except:
        # rename keys
        #   in_block1 -> in_block0, out_block1 -> out_block0
        checkpoint_renamed = dict()
        for key, val in checkpoint.items():
            if 'in_block' in key or 'out_block' in key:
                strs    = key.split('.')
                strs[1] = strs[1][:-1] + str(int(strs[1][-1])-1)
                strs[1] = '.'.join([strs[1][:-1], strs[1][-1]])
                key     = '.'.join(strs)
            checkpoint_renamed[key] = val
        model.load_state_dict(checkpoint_renamed, strict=False)


def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return img



to_date = lambda string: datetime.strptime(string, '%Y-%m-%d')



#############################################
# Own code
#############################################


# Helper functions

def instantiate_dsen2cr(input_shape, use_sar=True):
    # If not using SAR, second part will not get used
    input_shape = ((input_shape[0], input_shape[1], input_shape[2]), (2, input_shape[1], input_shape[2]))

    # Still based on original code; just trying to get the model as accurately as possible

    # model parameters
    num_layers = 16  # B value in paper
    feature_size = 256  # F value in paper

    # include the SAR layers as input to model
    include_sar_input = use_sar

    # cloud mask parameters
    use_cloud_mask = False # my own note: changed because we are not training, only doing inference
    cloud_threshold = 0.2  # set threshold for binarisation of cloud mask

    batch_size = None

    model = DSen2CR_model(input_shape,
                                    batch_per_gpu=batch_size,
                                    num_layers=num_layers,
                                    feature_size=feature_size,
                                    use_cloud_mask=use_cloud_mask,
                                    include_sar_input=include_sar_input)
    
    return(model)


def VPint_interpolation(target_grid, feature_grid, use_IP=True, use_EB=True):
    pred_grid = target_grid.copy()
    for b in range(0,target_grid.shape[2]): 
        MRP = WP_SMRP(target_grid[:,:,b],feature_grid[:,:,b])
        pred_grid[:,:,b] = MRP.run(method='exact',
                       auto_adapt=True, auto_adaptation_verbose=False,
                       auto_adaptation_epochs=25, auto_adaptation_max_iter=100,
                       auto_adaptation_strategy='random',auto_adaptation_proportion=0.8, 
                       resistance=use_EB,prioritise_identity=use_IP)

    return(pred_grid)
    
    
def normalise_and_visualise(img, title="", rgb=[3,2,1], percentile_clip=True, save_fig=False, save_path="", **kwargs):
    
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


def load_product(path, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
                        bands_10m={"B2":1, "B3":2, "B4":3, "B8":7},
                        bands_20m={"B5":4, "B6":5, "B7":6, "B8A":8, "B11":10, "B12":11, "CLD":12},
                        bands_60m={"B1":0, "B9":9}):

    grid = None
    size_y = -1
    size_x = -1

    scales = [bands_10m, bands_20m, bands_60m, {}] # For bands that have multiple resolutions

    with rasterio.open(path) as raw_product:
        product = raw_product.subdatasets

    # Initialise grid
    with rasterio.open(product[1]) as bandset:
        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        grid = np.zeros((size_y, size_x, len(keep_bands))).astype(np.uint16)


    # Iterate over band sets (resolutions)
    resolution_index = 0
    for bs in product:
        with rasterio.open(bs, dtype="uint16") as bandset:
            desc = bandset.descriptions
            size_y_local = bandset.profile['height']
            size_x_local = bandset.profile['width']

            band_index = 1
            # Iterate over bands
            for d in desc:
                band_name = d.split(",")[0]

                if(band_name in keep_bands and band_name in scales[resolution_index]):
                
                    if(band_name in bands_10m):
                        b = bands_10m[band_name]
                        upscale_factor = (1/2)
                    elif(band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 1
                    elif(band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 3

                    band_values = bandset.read(band_index, 
                                        out_shape=(
                                            int(size_y_local * upscale_factor),
                                            int(size_x_local * upscale_factor)
                                        ),
                                        resampling=Resampling.bilinear
                                    )

                    #grid[:,:,b] = np.moveaxis(band_values, 0, -1)
                    grid[:,:,b] = band_values
                    
                band_index += 1

        resolution_index += 1

    return(grid)


def load_product_windowed(path, y_size, x_size, y_offset, x_offset, keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
                        bands_10m={"B2":1, "B3":2, "B4":3, "B8":7},
                        bands_20m={"B5":4, "B6":5, "B7":6, "B8A":8, "B11":11, "B12":12, "CLD":13},
                        bands_60m={"B1":0, "B9":9, "B10":10},
                        return_bounds=False):

    grid = None
    size_y = -1
    size_x = -1

    scales = [bands_10m, bands_20m, bands_60m, {}] # For bands that have multiple resolutions

    with rasterio.open(path) as raw_product:
        product = raw_product.subdatasets

    # Initialise grid
    with rasterio.open(product[1]) as bandset:
        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(np.uint16)


    # Iterate over band sets (resolutions)
    resolution_index = 0
    for bs in product:
        with rasterio.open(bs, dtype="uint16") as bandset:
            desc = bandset.descriptions
            size_y_local = bandset.profile['height']
            size_x_local = bandset.profile['width']

            band_index = 1
            # Iterate over bands
            for d in desc:
                band_name = d.split(",")[0]

                if(band_name in keep_bands and band_name in scales[resolution_index]):
                
                    if(band_name in bands_10m):
                        b = bands_10m[band_name]
                        upscale_factor = (1/2)
                    elif(band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 1
                    elif(band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 3

                    # Output window using target resolution
                    window=Window(x_offset, y_offset, x_size, y_size)

                    # Second window for reading in local resolution
                    res_window = Window(x_offset*x_size/upscale_factor, y_offset*y_size/upscale_factor,
                                        window.width / upscale_factor, window.height / upscale_factor)
                          
                    if(return_bounds and band_name in bands_20m):
                        # Compute bounds for data fusion, needlessly computing for multiple bands but shouldn't be a big deal
                        # First indices, then points for xy, then extract coordinates from xy
                        # BL, TR --> (minx, miny), (maxx, maxy)
                        # Take special care with y axis; with xy indices, 0 should be top (coords 0/min is bottom)
                        left = x_offset*x_size/upscale_factor
                        top = y_offset*y_size/upscale_factor
                        right = left + x_size/upscale_factor
                        bottom = top + y_size/upscale_factor
                        tr = rasterio.transform.xy(bandset.transform, left, bottom)
                        bl = rasterio.transform.xy(bandset.transform, right, top)
                        
                        transformer = Transformer.from_crs(bandset.crs, 4326)
                        bl = transformer.transform(bl[0], bl[1])
                        tr = transformer.transform(tr[0], tr[1])

                        left = bl[0]
                        bottom = bl[1]
                        right = tr[0]
                        top = tr[1]
                        bounds = (left, bottom, right, top, bandset.transform, bandset.crs)
                
                    band_values = bandset.read(band_index, 
                                        out_shape=(
                                            window.height,
                                            window.width
                                        ),
                                        resampling=Resampling.bilinear,
                                        #masked=True, 
                                        window=res_window, 
                                    )
                    grid[:,:,b] = band_values
                band_index += 1
        resolution_index += 1

    if(return_bounds):
        return(grid, bounds)
    else:
        return(grid)
        
        
def load_product_windowed_withSAR(path, y_size, x_size, y_offset, x_offset, keep_bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dt=np.uint16):

    grid = None
    size_y = -1
    size_x = -1

    with rasterio.open(path) as bandset:

        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(dt)

        b = 0
        for band_index in range(1, len(keep_bands)+1):

            if(band_index in keep_bands):
                upscale_factor = 1

                # Output window using target resolution
                window=Window(x_offset, y_offset, x_size, y_size)

                # Second window for reading in local resolution
                res_window = Window(x_offset*x_size/upscale_factor, y_offset*y_size/upscale_factor,
                                    window.width / upscale_factor, window.height / upscale_factor)
            
                band_values = bandset.read(band_index, 
                                    out_shape=(
                                        window.height,
                                        window.width
                                    ),
                                    resampling=Resampling.bilinear,
                                    #masked=True, 
                                    window=res_window, 
                                )
                grid[:,:,b] = band_values
                b += 1
    return(grid)


def simulate_clouds(target, adjust_range=True, **kwargs):
    if(adjust_range):
        target = target / 10000
    target = np.moveaxis(target, -1, 0)
    
    target_cloudy, mask_cloud, mask_shadow = scg.add_cloud_and_shadow(target, return_cloud=True, **kwargs)
    
    target_cloudy = target_cloudy.numpy()[0,:,:,:] # We don't need a batch dim
    mask_cloud = np.moveaxis(mask_cloud.numpy()[0,:,:,:], 0, -1)
    mask_shadow = np.moveaxis(mask_shadow.numpy()[0,:,:,:], 0, -1)
    target_cloudy = np.moveaxis(target_cloudy, 0, -1)
    
    # Combine the two masks, keep highest value (probability of being either cloud or cloud shadow)
    mask = np.maximum.reduce([mask_cloud, mask_shadow])
    
    if(adjust_range):
        target_cloudy = target_cloudy * 10000
    
    return(target_cloudy, mask)


def mask_buffer(mask,passes=1):
    for p in range(0,passes):
        new_mask = mask.copy()
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if(np.isnan(mask[i,j])):
                    if(i>0):
                        new_mask[i-1,j] = np.nan
                    if(i<mask.shape[0]-1):
                        new_mask[i+1,j] = np.nan
                    if(j>0):
                        new_mask[i,j-1] = np.nan
                    if(j<mask.shape[1]-1):
                        new_mask[i,j+1] = np.nan
        mask = new_mask
    return(mask)

def run_dsen2cr(target, sar, model):
    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = np.clip(target, 0, 10000)
    target = target / 2000
    target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])
    
    # Preprocess sar data:
    # move band axis to 0, clip VV to [-25,0], clip VH to [-32.5,0], add respective values to shift to positive, rescale to 0-2 range, add batch dimension
    sar = np.moveaxis(sar, -1, 0)
    sar[1,:,:] = np.clip(sar[1,:,:], -25, 0) + 25 # VV
    sar[0,:,:] = np.clip(sar[0,:,:], -32.5, 0) + 32.5 # VH
    sar[1,:,:] = sar[1,:,:] / 25 * 2
    sar[0,:,:] = sar[0,:,:] / 32.5 * 2
    sar = sar.reshape(1, sar.shape[0], sar.shape[1], sar.shape[2])
    
    # Perform inference
    pred_grid = model((target, sar))

    # Get rid of batch dim, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0,:,:,:], 0, -1) * 2000
    
    return(pred_grid)
   
    
def run_dsen2cr_nosar(target, model):
    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = np.clip(target, 0, 10000)
    target = target / 2000
    target = target.reshape(1, target.shape[0], target.shape[1], target.shape[2])

    # Perform inference
    pred_grid = model((target, np.zeros((1, 2, target.shape[2], target.shape[3]))))

    # Get rid of batch dim, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0,:,:,:], 0, -1) * 2000
    
    return(pred_grid)





#def run_uncrtaints_original_data_verification(target, sar, mask, model):
def run_uncrtaints(target, sar, mask, model):
    # UnCRtain-TS based on ResNet, so use ResNet preprocessing described in paper

    
    base_path = "/mnt/c/Users/laure/Downloads/uncrtaints_data/"
    path_mask = base_path + "generic_2_test_all_s2cloudless_mask.npy"
    path_s1 = base_path + "s1_ROIs1868_100_ImgNo_0_2018-01-03_patch_100.tif"
    path_s2 = base_path + "s2_ROIs1868_100_ImgNo_0_2018-01-08_patch_100.tif"


    with rasterio.open(path_s2) as fp:
        target = fp.read().astype(np.float64)

    with rasterio.open(path_s1) as fp:
        sar = fp.read().astype(np.float64)




    # Preprocess target data:
    target = process_MS(target, method='default')

    #mask[mask >= 0.1] = 1
    #mask[mask < 0.1] = 0
   
    # Preprocess sar data:
    sar = process_SAR(sar, method='default')


    mask = np.zeros_like(mask) # Cloud-free patch, checkinf if results in identity

    num_steps = 30

    dates_S1 = dates_S2 = [(to_date('2018-01-03') + dt.timedelta(days=date) - to_date('2018-01-03')).days for date in range(0, num_steps)]
    dates = torch.stack((torch.tensor(dates_S1),torch.tensor(dates_S2))).float().mean(dim=0)[None]

    input_img2 = np.concatenate([sar, target], axis=0) * 10 # Concatenate on band dim, multiply by 10
    input_img = np.zeros((1, num_steps, input_img2.shape[0], input_img2.shape[1], input_img2.shape[2])).astype(np.float32)
    for a in range(0, num_steps):
        input_img[0, a, :, :, :] = input_img2
    input_img = torch.from_numpy(input_img)

    #input_img = np.concatenate([sar, target], axis=0) * 10 # Concatenate on band dim, multiply by 10
    #input_img = torch.from_numpy(input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1], input_img.shape[2]).astype(np.float32)) # (b, t, c, h, w)
    #masks = torch.from_numpy(mask.reshape((1, 1, 1, 256, 256)).astype(np.float32)) 
    masks = np.zeros_like(input_img)
    y = torch.from_numpy(target.reshape((1, 1, target.shape[0], target.shape[1], target.shape[2])).astype(np.float32))

    inputs = {'A': input_img, 'B': y, 'dates': dates, 'masks': masks}




    real_A = input_img*10 # resulting tensor is [Batchsize x Time x Channels x Height x Width]

    # forward propagation at inference time
    with torch.no_grad():
        fake_B = model(real_A, batch_positions=dates)
    pred_grid = fake_B[:, :, :13, ...].cpu().numpy() / 10


    
    # Perform inference
    #with torch.no_grad():
    #    # compute single-model mean and variance predictions
    #    model.set_input(inputs)
    #    model.forward()
    #    model.get_loss_G()
    #    model.rescale()
    #    pred_grid = model.fake_B.cpu().numpy() / 10 # divide by 10 to undo above thing


    # Perform inference, version 2
    #with torch.no_grad():
    #    fake_B = model(input_img, batch_positions=dates)
    #pred_grid = fake_B / 10

    print("target old:", np.mean(target))

        
    # Get rid of batch/time dims, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0, 0, :, :, :], 0, -1)
    pred_grid2 = pred_grid * 10000
    target = np.moveaxis(target, 0, -1) * 10000

    print("target new:", np.mean(target))


    normalise_and_visualise(target, title='Target', save_fig=True, save_path="target.pdf")
    normalise_and_visualise(pred_grid, title='Reconstruction', save_fig=True, save_path="reconstruction.pdf")

    print(target.shape)

    for b in range(0, pred_grid.shape[2]):
        print(str(np.median(target[:,:,b] )) + " - " + str(np.median(pred_grid[:,:,b])) + " - " + str(np.median(pred_grid2[:,:,b])))
        plt.imshow(np.absolute(pred_grid[:,:,b] - target[:,:,b]))
        plt.show()
    sdlkfj
   
    return(pred_grid)


def run_uncrtaints_wip(target, sar, mask, model):
    # UnCRtain-TS based on ResNet, so use ResNet preprocessing described in paper




    dates_S1 = dates_S2 = [(to_date(date) - to_date('2014-04-03')).days for date in ['2014-04-03']]
    dates = torch.stack((torch.tensor(dates_S1),torch.tensor(dates_S2))).float().mean(dim=0)[None]


    # Preprocess target data:
    # move band axis to 0, clip to 10000, divide by 2000, add batch dimension
    target = np.moveaxis(target, -1, 0)
    target = process_MS(target, method='default')
    #target = np.clip(target, 0, 10000)
    #target = target / 2000

    mask[mask >= 0.1] = 1
    mask[mask < 0.1] = 0
   
    # Preprocess sar data:
    # move band axis to 0, clip VV to [-25,0], clip VH to [-32.5,0], add respective values to shift to positive, rescale to 0-2 range, add batch dimension
    sar = np.moveaxis(sar, -1, 0)
    sar = process_SAR(sar, method='default')
    #sar[1,:,:] = np.clip(sar[1,:,:], -25, 0) + 25 # VV
    #sar[0,:,:] = np.clip(sar[0,:,:], -32.5, 0) + 32.5 # VH
    #sar[1,:,:] = sar[1,:,:] / 25 * 2
    #sar[0,:,:] = sar[0,:,:] / 32.5 * 2

    input_img = np.concatenate([target, sar], axis=0) * 10 # Concatenate on band dim, multiply by 10
    input_img = torch.from_numpy(input_img.reshape(1, 1, input_img.shape[0], input_img.shape[1], input_img.shape[2]).astype(np.float32)) # (b, t, c, h, w)
    masks = torch.from_numpy(mask.reshape((1, 1, 1, 256, 256)).astype(np.float32)) 
    y = torch.from_numpy(target.reshape((1, 1, target.shape[0], target.shape[1], target.shape[2])).astype(np.float32))

    inputs = {'A': input_img, 'B': y, 'dates': dates, 'masks': masks}
    
    # Perform inference
    with torch.no_grad():
        # compute single-model mean and variance predictions
        model.set_input(inputs)
        model.forward()
        model.get_loss_G()
        model.rescale()
        pred_grid = model.fake_B.cpu().numpy() / 10 # divide by 10 to undo above thing


    # Perform inference, version 2
    #with torch.no_grad():
    #    fake_B = model(input_img, batch_positions=dates)
    #pred_grid = fake_B / 10

    
        
    # Get rid of batch/time dims, return to 0-10000 scale and move channels back to end
    pred_grid = np.moveaxis(pred_grid[0, 0, :, :, :], 0, -1) * 2000

    normalise_and_visualise(np.moveaxis(target, 0, -1), title='Target')
    #plt.imshow(sar[0,:,:])
    #plt.title("SAR VH")
    #plt.show()
    #plt.imshow(sar[1,:,:])
    #plt.title("SAR VV")
    #plt.show()
    #plt.imshow(mask)
    #plt.title("Mask")
    #plt.show()
    normalise_and_visualise(pred_grid, title='Reconstruction')
    sdlkfj
   
    return(pred_grid)


def run_patch(base_path, legend, scene, y_size, x_size, y_offset, x_offset, model, cloud_prop_list, 
                       cloud_threshold=0.1, algorithm="dsen2cr", buffer_mask=True, mask_buffer_size=5):
                       
    target_sar_path = base_path + "l1c/collocated_" + scene +".tif"    
    
    # Load target and sar unified image
    try:
        target = load_product_windowed_withSAR(target_sar_path, y_size, x_size, y_offset, x_offset).astype(float)
        target = target[:,:,:13]
        sar = load_product_windowed_withSAR(target_sar_path, y_size, x_size, y_offset, x_offset, dt=np.float32).astype(float)
        sar = sar[:,:,13:]
    except:
        print(y_size, x_size, y_offset, x_offset)
        return(False)

    # L1C, SAR and feature product coverages don't completely align; this will be bad at edges, but filter out most of those cases
    if(np.median(sar) == 0.0 or np.median(target) <= 0):
        print("Coverage problem for:", scene, y_offset, x_offset)
        return(False)
            
    # Simulate clouds, sample from real dataset cloud proportions for similar distribution
    clear_prop = 1 - (cloud_prop_list[np.random.randint(low=0, high=len(cloud_prop_list)-1)] / 100)
    target_cloudy, mask = simulate_clouds(target, channel_offset=0, clear_threshold=clear_prop)
    mask = np.max(mask, axis=2) # just to keep it like the rest without difference per band
            
    if(algorithm == "VPint"):
        # VPint needs feature img (we use 1m) and
        # clouds marked as missing values
        # Load features to filter by feature img coverage (good comparison with VPint)
        feature_path = base_path + legend[scene]["1m"] + ".zip"
        features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset, target_res=10).astype(float)
        features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))
        
        if(buffer_mask):
            # Buffer mask (pad extra masked pixels around edges), also discretises in process
            mask = mask_buffer(mask, mask_buffer_size, threshold=cloud_threshold)
        
        for i in range(0, target_cloudy.shape[0]):
            for j in range(0, target_cloudy.shape[1]):
                if(mask[i,j] > cloud_threshold): 
                    a = np.ones(target_cloudy.shape[2]) * np.nan
                    target_cloudy[i,j,:] = a
    
    if(algorithm=="dsen2cr"):
        pred_grid = run_dsen2cr(target_cloudy, sar, model)
    elif(algorithm=="dsen2cr_nosar"):
        pred_grid = run_dsen2cr_nosar(target_cloudy, model)
    elif(algorithm=="uncrtaints"):
        pred_grid = run_uncrtaints(target_cloudy, sar, mask, model)
    elif(algorithm=="VPint"):
        pred_grid = VPint_interpolation(target_cloudy, features, use_IP=True, use_EB=True)
    
    return(pred_grid)
    


# Setup

if(len(sys.argv) != 3):
    print("Usage: python SEN2-MSI-T-nn.py [method name] [scene number]")

conditions_algorithms = [
    #"dsen2cr", 
    #"dsen2cr_nosar",
    "uncrtaints",
    #"VPint",
]

save_path = "/mnt/c/Users/laure/Data/results/"
base_path = "/mnt/c/Users/laure/Data/SEN2-MSI-T/"

with open("image_ids_unsorted.pkl", 'rb') as fp:
    legend = pickle.load(fp)
    
conditions_scenes = list(legend.keys())
# Filter out scenes where no SAR data was available
conditions_scenes = [s for s in conditions_scenes if not(legend[s]['sar'] == "")]
conditions_scenes = [
    ##"europe_urban_madrid",
    #"america_shrubs_mexico",
    #"america_herbaceous_peru",
    "europe_cropland_hungary",
    #"asia_shrubs_indiapakistan",
    #"america_forest_mississippi",
]

i = 1
conds = {}

#for scene in conditions_scenes:
#    for alg in conditions_algorithms:
#        conds[i] = {"algorithm":alg, "scene":scene}
#        i += 1
            
#this_run_cond = conds[int(sys.argv[3])]
#scene = conditions_scenes[int(sys.argv[2])]
scene = conditions_scenes[0] # 3 bad


# Hardcoding this because adding UnCRtain-TS messed up CLI arguments, not worth the hassle of fixing it
for scene in conditions_scenes:
    print("Scene:", scene)
    this_run_cond = {
        #"algorithm":sys.argv[1], 
        "algorithm": "uncrtaints", 
        #"algorithm": "dsen2cr", 
        "scene": scene,
    }



    # Some parameters

    replace = True # Set to True to overwrite existing runs
    size_y = 256
    size_x = 256
    size_f = 13

    checkpoint_path = "/mnt/c/Users/laure/Data/models/model_SARcarl.hdf5"
    checkpoint_path_nosar = "/mnt/c/Users/laure/Data/models/model_noSARcarl.hdf5"
    #checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/monotemporalL2/"
    checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/multitemporalL2/"
    #checkpoint_path_uncrtaints = "/mnt/c/Users/laure/Data/models/noSAR_1/"
    uncrtaints_complicated = False

    # Instantiate model and setup TF/torch

    if(this_run_cond["algorithm"] in ["dsen2cr", "dsen2cr_nosar"]):
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        print(tf.test.is_built_with_cuda())

        if gpus:
        # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        input_shape = (size_f, size_y, size_x)
        if(this_run_cond['algorithm'] == "dsen2cr"):
            model = instantiate_dsen2cr(input_shape)
            model.load_weights(checkpoint_path)
        elif(this_run_cond['algorithm'] == "dsen2cr_nosar"):
            model = instantiate_dsen2cr(input_shape, use_sar=False)
            model.load_weights(checkpoint_path_nosar)
        else:
            model = None
        print("Initialised Dsen2-CR and loaded checkpoint weights.")

    elif(this_run_cond["algorithm"] == "uncrtaints"):
        if(uncrtaints_complicated):

            dirname = checkpoint_path_uncrtaints + "conf.json"

            # Bit more of Ebel code
            parser = create_parser(mode='test')
            test_config = parser.parse_args()

            #conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json") if not test_config.load_config else test_config.load_config
            conf_path = checkpoint_path_uncrtaints + "/conf.json"
            if os.path.isfile(conf_path):
                with open(conf_path) as file:
                    model_config = json.loads(file.read())
                    t_args = argparse.Namespace()
                    # do not overwrite the following flags by their respective values in the config file
                    no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'root1', 'root2', 'root3', 
                    'max_samples_count', 'batch_size', 'display_step', 'plot_every', 'export_every', 'input_t', 'region', 'min_cov', 'max_cov']
                    conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
                    for key, val in vars(test_config).items(): 
                        if key in no_overwrite: conf_dict[key] = val
                    t_args.__dict__.update(conf_dict)
                    config = parser.parse_args(namespace=t_args)
            else: config = test_config # otherwise, keep passed flags without any overwriting
            config = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
            if config.pretrain: config.batch_size = 32
            

            device = torch.device(config.device)

            model = get_model(config)
            model = model.to(device)
            print(model)
            model.eval()

        else:

            model = UNCRTAINTS(input_dim=2+13,
                encoder_widths=[128],
                decoder_widths=[128, 128, 128, 128, 128], 
                out_conv=[13],
                out_nonlin_mean=True,
                out_nonlin_var="softplus",
                agg_mode="att_group",
                encoder_norm="group",
                decoder_norm="batch",
                n_head=16,
                d_model=256,
                d_k=4,
                pad_value=0,
                padding_mode="reflect",
                positional_encoding=True,
                covmode="diag",
                scale_by=10.0,
                separate_out=False,
                use_v=False,
                block_type='mbconv',
                is_mono=False)
            trained_checkp = checkpoint_path_uncrtaints + "/model.pth.tar"
            pretrained_dict = torch.load(trained_checkp, map_location="cpu")["state_dict_G"]
            pretrained_dict2 = {}
            for k, v in pretrained_dict.items():
                #print("k")
                #print(k)
                spl = k.split(".")
                firstthing = spl[0]
                # If last character is number, decrement by 1, add . in front, then reconstruct string
                if(firstthing[-1].isdigit()):
                    firstthing2 = firstthing[:-1] + "." + str(int(firstthing[-1]) - 1)
                    k2 = firstthing2
                else:
                    k2 = firstthing
                for s in spl:
                    if(s != firstthing):
                        k2 += "." + s
                #print("k2")
                #print(k2)
                pretrained_dict2[k2] = v


            
            model.load_state_dict(pretrained_dict2, strict=True)
            model.eval()

    else:
        raise NotImplementedError


    # Prepare for run


    currdir = save_path + "results_l1c"
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass
        

            
    np.set_printoptions(threshold=np.inf)

    log = rasterio.logging.getLogger()
    log.setLevel(rasterio.logging.FATAL)

    # Run
            
    #for scene in conditions_scenes:
    #    this_run_cond["scene"] = scene

    # Create directory per scene
    currdir1 = currdir + "/" + this_run_cond["scene"]
    if(not(os.path.exists(currdir1))):
        try:
            os.mkdir(currdir1)
        except:
            pass

    # Create directory per algorithm
    currdir1 = currdir1 + "/" + this_run_cond["algorithm"]
    if(not(os.path.exists(currdir1))):
        try:
            os.mkdir(currdir1)
        except:
            pass

    # Get a list of cloud proportions to sample from
    cloud_prop_list = np.zeros(20*500)
    i = 0
    with open("cloud_proportions.pkl", 'rb') as fp:
        a = pickle.load(fp)
        for k, v in a.items():
            for patch, prop in v.items():
                cloud_prop_list[i] = prop
                i += 1
    cloud_prop_list = cloud_prop_list[:i]

    # Check dims, iterate patches
    scene_height = -1
    scene_width = -1
    ref_product_path = base_path + "l1c/collocated_" + scene + ".tif"
    with rasterio.open(ref_product_path) as fp:
        scene_height = fp.height
        scene_width = fp.width

    max_row = int(str(scene_height / size_y).split(".")[0])
    max_col = int(str(scene_width / size_x).split(".")[0])

    # Shuffle indices to allow multiple tasks to run
    row_list = np.arange(max_row)
    col_list = np.arange(max_col)
    np.random.shuffle(row_list)
    np.random.shuffle(col_list)

    # Iterate
    for y_offset in row_list:
        for x_offset in col_list:

            # Create directory for patch
            
            patch_name = "r" + str(y_offset) + "_c" + str(x_offset)

            path = currdir1 + "/" + patch_name
            if(not(os.path.exists(path))):
                try:
                    os.mkdir(path)
                except:
                    pass
                    
            # Path for save file
            path = path + "/reconstruction.npy"
            
            # Run reconstruction
            if(replace or not(os.path.exists(path))):
                st = time.time()
                #try:
                pred_grid = run_patch(base_path, legend, this_run_cond["scene"], size_y, size_x, y_offset, x_offset, model, cloud_prop_list, algorithm=this_run_cond["algorithm"])
                et = time.time()
                        
                if(type(pred_grid) != type(False)): # just because if(pred_grid) is ambiguous
                    np.save(path, pred_grid)
                        
                #except:
                #    print("Failed run for ", this_run_cond, y_offset, x_offset)
                    
        
print("Terminated successfully")