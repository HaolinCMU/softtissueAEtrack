# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:34:53 2022

@author: hlinl
"""


import os
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from PARAM import *


class Flatten(nn.Module):
    """
    """

    def forward(self, input):
        return input.view(input.size()[0], -1)


class UnFlatten(nn.Module):
    """
    """

    def forward(self, input, size):
        return input.view(input.size()[0], size, 1, 1)


class Encoder(nn.Module):
    """
    """

    pass


class Decoder(nn.Module):
    """
    """

    pass


class AutoEncoder_Linear(nn.Module):
    """
    """

    def __init__(self):
        """
        """

        super(AutoEncoder_Linear, self).__init__()

        pass


class AutoEncoder_Conv(nn.Module):
    """
    """

    def __init__(self, encoder_layer_num, decoder_layer_num, ):
        """
        """

        super(AutoEncoder_Conv, self).__init__()

        pass


class VAE_Linear(nn.Module):
    """
    """

    def __init__(self):
        """
        """

        super(VAE_Linear, self).__init__()

        self._bottle_neck_dim = ML_VAE.BOTTLENECK_DIM
        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Linear(ML_VAE.INPUT_DIM, 256), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                # nn.Dropout(0.5), 
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Linear(256, self._bottle_neck_dim), 
                                                # ML_VAE.ACTIVATION_LAYER,
                                                # nn.Dropout(0.5), 
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        # self._enc_conv_hidden_3 = nn.Sequential(nn.Linear(1024, 512), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (128, 128, 32) -> (64, 64, 64). 
        # self._enc_conv_hidden_4 = nn.Sequential(nn.Linear(512, 256), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (64, 64, 64) -> (32, 32, 128). 
        # self._enc_conv_hidden_5 = nn.Sequential(nn.Linear(256, 128), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (32, 32, 128) -> (16, 16, 256). 
        # self._enc_conv_hidden_6 = nn.Sequential(nn.Linear(128, 64), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (16, 16, 256) -> (8, 8, 512).
        # self._enc_conv_hidden_7 = nn.Sequential(nn.Linear(64, self._bottle_neck_dim), 
        #                                         # ML_VAE.ACTIVATION_LAYER,
        #                                         ) # (16, 16, 256) -> (8, 8, 512).

        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                    #   self._enc_conv_hidden_3, 
                                    #   self._enc_conv_hidden_4, 
                                    #   self._enc_conv_hidden_5, 
                                    #   self._enc_conv_hidden_6, 
                                    #   self._enc_conv_hidden_7,
                                      )

        # Flatten(). 

        # Bottleneck
        self._enc_output_mu = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._enc_output_logvar = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._dec_fc_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, self._bottle_neck_dim),
                                              # nn.Dropout(0.5)
                                              ) # (16, ) -> (512, ). 

        # Unflatten(). 

        # Decoder. 
        self._dec_deconv_hidden_1 = nn.Sequential(nn.Linear(self._bottle_neck_dim, 256),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                #   nn.Dropout(0.5), 
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.Linear(256, ML_VAE.OUTPUT_DIM),
                                                #   ML_VAE.ACTIVATION_LAYER,
                                                #   nn.Dropout(0.5), 
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        # self._dec_deconv_hidden_3 = nn.Sequential(nn.Linear(128, 256),
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           nn.Dropout(0.5), 
        #                                           ) # (16, 16, 256) -> (32, 32, 128). 
        # self._dec_deconv_hidden_4 = nn.Sequential(nn.Linear(256, 512),
        #                                           nn.Dropout(0.5), 
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           ) # (32, 32, 128) -> (64, 64, 64). 
        # self._dec_deconv_hidden_5 = nn.Sequential(nn.Linear(512, 1024),
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           nn.Dropout(0.5), 
        #                                           ) # (64, 64, 64) -> (128, 128, 32). 
        # self._dec_deconv_hidden_6 = nn.Sequential(nn.Linear(1024, 2048),
        #                                           nn.Dropout(0.5), 
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           ) # (128, 128, 32) -> (256, 256, 16). 
        # self._dec_deconv_hidden_7 = nn.Sequential(nn.Linear(2048, ML_VAE.OUTPUT_DIM)) # (256, 256, 16) -> (512, 512, 1). 
                                                  
        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                    #   self._dec_deconv_hidden_3,
                                    #   self._dec_deconv_hidden_4,
                                    #   self._dec_deconv_hidden_5,
                                    #   self._dec_deconv_hidden_6,
                                    #   self._dec_deconv_hidden_7,
                                      )
    
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        # output = output.view(output.size(0), -1) # Flatten. 

        mu = self._enc_output_mu(output)
        logvar = self._enc_output_logvar(output)

        return mu, logvar
    

    def reparameterize(self, mu, logvar):
        """
        """

        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()

        return eps.mul(std).add_(mu)


    def decoder(self, latent):
        """
        """

        output = self._dec_fc_hidden_1(latent) # (16, ) -> (512, ). 
        # output = output.view(output.size(0), self._bottle_neck_dim, 1, 1) # Flatten. (512, ) -> (1, 1, 512). 

        output = self._decoder(output)

        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        output = self.decoder(latent)

        return output, mu, logvar, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim



class AE_Linear(nn.Module):
    """
    """

    def __init__(self):
        """
        """

        super(AE_Linear, self).__init__()

        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Linear(ML_VAE.INPUT_DIM, 256), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                # nn.Dropout(0.5), 
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Linear(256, self._latent_dim), 
                                                # ML_VAE.ACTIVATION_LAYER,
                                                # nn.Dropout(0.5), 
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        # self._enc_conv_hidden_3 = nn.Sequential(nn.Linear(1024, 512), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (128, 128, 32) -> (64, 64, 64). 
        # self._enc_conv_hidden_4 = nn.Sequential(nn.Linear(512, 256), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (64, 64, 64) -> (32, 32, 128). 
        # self._enc_conv_hidden_5 = nn.Sequential(nn.Linear(256, 128), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (32, 32, 128) -> (16, 16, 256). 
        # self._enc_conv_hidden_6 = nn.Sequential(nn.Linear(128, 64), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         nn.Dropout(0.5), 
        #                                         ) # (16, 16, 256) -> (8, 8, 512).
        # self._enc_conv_hidden_7 = nn.Sequential(nn.Linear(64, self._latent_dim), 
        #                                         ML_VAE.ACTIVATION_LAYER,
        #                                         ) # (16, 16, 256) -> (8, 8, 512).

        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                    #   self._enc_conv_hidden_3, 
                                    #   self._enc_conv_hidden_4, 
                                    #   self._enc_conv_hidden_5, 
                                    #   self._enc_conv_hidden_6, 
                                    #   self._enc_conv_hidden_7,
                                      )

        # Decoder. 
        self._dec_deconv_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, 256),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                #   nn.Dropout(0.5), 
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.Linear(256, ML_VAE.OUTPUT_DIM),
                                                #   ML_VAE.ACTIVATION_LAYER,
                                                #   nn.Dropout(0.5), 
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        # self._dec_deconv_hidden_3 = nn.Sequential(nn.Linear(128, 256),
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           nn.Dropout(0.5), 
        #                                           ) # (16, 16, 256) -> (32, 32, 128). 
        # self._dec_deconv_hidden_4 = nn.Sequential(nn.Linear(256, 512),
        #                                           nn.Dropout(0.5), 
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           ) # (32, 32, 128) -> (64, 64, 64). 
        # self._dec_deconv_hidden_5 = nn.Sequential(nn.Linear(512, 1024),
        #                                           ML_VAE.ACTIVATION_LAYER,
        #                                           nn.Dropout(0.5), 
        #                                           ) # (64, 64, 64) -> (128, 128, 32). 
        # self._dec_deconv_hidden_6 = nn.Sequential(nn.Linear(1024, 2048),
        #                                           nn.Dropout(0.5), 
        #                                         #   ML_VAE.ACTIVATION_LAYER,
        #                                           ) # (128, 128, 32) -> (256, 256, 16). 
        # self._dec_deconv_hidden_7 = nn.Sequential(nn.Linear(2048, ML_VAE.OUTPUT_DIM)) # (256, 256, 16) -> (512, 512, 1). 
                                                  
        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                    #   self._dec_deconv_hidden_3,
                                    #   self._dec_deconv_hidden_4,
                                    #   self._dec_deconv_hidden_5,
                                    #   self._dec_deconv_hidden_6,
                                    #   self._dec_deconv_hidden_7,
                                      )
    
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        # output = output.view(output.size(0), -1) # Flatten. 
        return output


    def decoder(self, latent):
        """
        """
        
        output = self._decoder(latent)
        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        latent = self.encoder(x)
        output = self.decoder(latent)

        return output, latent, latent, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim


class VAE_Conv(nn.Module):
    """
    """

    def __init__(self, input_dim, latent_dim, enc_archi_dict, dec_archi_dict, 
                 latent_distrib='Gaussian'):
        """
        """

        super(VAE_Conv, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_archi_dict = copy.deepcopy(enc_archi_dict)
        self.dec_archi_dict = copy.deepcopy(dec_archi_dict)
        self.latent_distrib = latent_distrib

        # Encoder. 
        self._enc_is_convBatchNorm = self.enc_archi_dict['is_convBatchNorm']
        self._enc_is_convPooling = self.enc_archi_dict['is_convPooling']
        self._enc_is_convDropout = self.enc_archi_dict['is_convDropout']

        self._enc_conv_layer_num = self.enc_archi_dict['conv_layer_num']
        self._enc_conv_kernel_sizes = self.enc_archi_dict['conv_kernel_sizes'] # A list or array. 
        self._enc_conv_paddings = self.enc_archi_dict['conv_paddings'] # A list or array. 
        self._enc_conv_strides = self.enc_archi_dict['conv_strides'] # A list or array. 
        self._enc_conv_dilation = self.enc_archi_dict['conv_dilation']
        self._enc_conv_dropout_ratio = self.enc_archi_dict['conv_dropout_ratio']

        self._enc_poolingLayer = self.enc_archi_dict['poolingLayer'] # Pooling layer type. 
        self._enc_pooling_kernel_size = self.enc_archi_dict['pooling_kernel_size'] # A single number. 
        self._enc_pooling_padding = self.enc_archi_dict['pooling_padding']
        self._enc_pooling_stride = self.enc_archi_dict['pooling_stride']
        
        self._enc_conv_channel_num_init = self.enc_archi_dict['conv_channel_num_init']
        self._enc_conv_channel_num_multiplier = self.enc_archi_dict['conv_channel_num_multiplier']
        self._enc_conv_predef_channel_num_list = self.enc_archi_dict['conv_predef_channel_num_list']

        self._enc_is_mlpBatchNorm = self.enc_archi_dict['is_mlpBatchNorm']
        self._enc_is_mlpDropout = self.enc_archi_dict['is_mlpDropout']

        self._enc_mlp_layer_num = self.enc_archi_dict['mlp_layer_num']



        
        # self.mlp_layer_num = mlp_layer_num
        # self._is_mlpBatchNorm = IS_MLP_BATCHNORM
        # self._is_mlpDropout = IS_MLP_DROPOUT
        self._mlp_1st_layer_num = ML_VAE.MLP_FIRST_LAYER_NUM
        self._mlp_layer_num_decay_div = ML_VAE.MLP_LAYER_NUM_DECAY_DIV
        
        self._activationLayer = ML_VAE.ACTIVATION_LAYER


        self.hidden_1 = nn.Conv2d()

        pass


class VAE_Conv_test(nn.Module):
    """
    Only for validation. 
    """

    def __init__(self):
        """
        """

        super(VAE_Conv_test, self).__init__()

        self._bottle_neck_dim = ML_VAE.BOTTLENECK_DIM
        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        self._enc_conv_hidden_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (128, 128, 32) -> (64, 64, 64). 
        self._enc_conv_hidden_4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (64, 64, 64) -> (32, 32, 128). 
        self._enc_conv_hidden_5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (32, 32, 128) -> (16, 16, 256). 
        self._enc_conv_hidden_6 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (16, 16, 256) -> (8, 8, 512).
        self._enc_conv_hidden_7 = nn.Sequential(nn.Conv2d(512, self._bottle_neck_dim, kernel_size=(8,8)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (16, 16, 256) -> (8, 8, 512).
        # self._enc_GAP = nn.AvgPool2d((8,8), 1) # GAP, (8, 8, 512) -> (1, 1, 512).

        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                      self._enc_conv_hidden_3, 
                                      self._enc_conv_hidden_4, 
                                      self._enc_conv_hidden_5, 
                                      self._enc_conv_hidden_6, 
                                      self._enc_conv_hidden_7)

        # Flatten(). 

        # Bottleneck
        self._enc_output_mu = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._enc_output_logvar = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._dec_fc_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, self._bottle_neck_dim),
                                              # nn.Dropout(0.5)
                                              ) # (16, ) -> (512, ). 

        # Unflatten(). 

        # Decoder. 
        self._dec_deconv_hidden_1 = nn.Sequential(nn.ConvTranspose2d(self._bottle_neck_dim, 512, kernel_size=(8,8)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        self._dec_deconv_hidden_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (16, 16, 256) -> (32, 32, 128). 
        self._dec_deconv_hidden_4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (32, 32, 128) -> (64, 64, 64). 
        self._dec_deconv_hidden_5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (64, 64, 64) -> (128, 128, 32). 
        self._dec_deconv_hidden_6 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (128, 128, 32) -> (256, 256, 16). 
        self._dec_deconv_hidden_7 = nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  nn.Sigmoid()) # (256, 256, 16) -> (512, 512, 1). 
                                                  
        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                      self._dec_deconv_hidden_3,
                                      self._dec_deconv_hidden_4,
                                      self._dec_deconv_hidden_5,
                                      self._dec_deconv_hidden_6,
                                      self._dec_deconv_hidden_7)
    
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        output = output.view(output.size(0), -1) # Flatten. 

        mu = self._enc_output_mu(output)
        logvar = self._enc_output_logvar(output)

        return mu, logvar
    

    def reparameterize(self, mu, logvar):
        """
        """

        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()

        return eps.mul(std).add_(mu)


    def decoder(self, latent):
        """
        """

        output = self._dec_fc_hidden_1(latent) # (16, ) -> (512, ). 
        output = output.view(output.size(0), self._bottle_neck_dim, 1, 1) # Flatten. (512, ) -> (1, 1, 512). 

        output = self._decoder(output)

        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        output = self.decoder(latent)

        return output, mu, logvar, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim


class VAE_Conv_test_128(nn.Module):
    """
    Only for validation. 
    """

    def __init__(self):
        """
        """

        super(VAE_Conv_test_128, self).__init__()

        self._bottle_neck_dim = ML_VAE.BOTTLENECK_DIM
        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        self._enc_conv_hidden_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (128, 128, 32) -> (64, 64, 64). 
        self._enc_conv_hidden_4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (64, 64, 64) -> (32, 32, 128). 
        self._enc_conv_hidden_5 = nn.Sequential(nn.Conv2d(128, self._bottle_neck_dim, kernel_size=(8,8)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (32, 32, 128) -> (16, 16, 256). 

        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                      self._enc_conv_hidden_3, 
                                      self._enc_conv_hidden_4, 
                                      self._enc_conv_hidden_5)

        # Flatten(). 

        # Bottleneck
        self._enc_output_mu = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._enc_output_logvar = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._dec_fc_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, self._bottle_neck_dim),
                                              # nn.Dropout(0.5)
                                              ) # (16, ) -> (512, ). 

        # Unflatten(). 

        # Decoder. 
        self._dec_deconv_hidden_1 = nn.Sequential(nn.ConvTranspose2d(self._bottle_neck_dim, 128, kernel_size=(8,8)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        self._dec_deconv_hidden_3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (16, 16, 256) -> (32, 32, 128). 
        self._dec_deconv_hidden_4 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (32, 32, 128) -> (64, 64, 64). 
        self._dec_deconv_hidden_5 = nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (64, 64, 64) -> (128, 128, 32). 
                                                  
        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                      self._dec_deconv_hidden_3,
                                      self._dec_deconv_hidden_4,
                                      self._dec_deconv_hidden_5)
    
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        output = output.view(output.size(0), -1) # Flatten. 

        mu = self._enc_output_mu(output)
        logvar = self._enc_output_logvar(output)

        return mu, logvar
    

    def reparameterize(self, mu, logvar):
        """
        """

        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()

        return eps.mul(std).add_(mu)


    def decoder(self, latent):
        """
        """

        output = self._dec_fc_hidden_1(latent) # (16, ) -> (512, ). 
        output = output.view(output.size(0), self._bottle_neck_dim, 1, 1) # Flatten. (512, ) -> (1, 1, 512). 

        output = self._decoder(output)

        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        output = self.decoder(latent)

        return output, mu, logvar, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim


class Autoencoder_Conv_test(nn.Module):
    """
    Only for validation. 
    """

    def __init__(self):
        """
        """

        super(Autoencoder_Conv_test, self).__init__()

        self._bottle_neck_dim = ML_VAE.BOTTLENECK_DIM
        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        self._enc_conv_hidden_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (128, 128, 32) -> (64, 64, 64). 
        self._enc_conv_hidden_4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (64, 64, 64) -> (32, 32, 128). 
        self._enc_conv_hidden_5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (32, 32, 128) -> (16, 16, 256). 
        self._enc_conv_hidden_6 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (16, 16, 256) -> (8, 8, 512).
        self._enc_conv_hidden_7 = nn.Sequential(nn.Conv2d(512, self._bottle_neck_dim, kernel_size=(8,8)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (16, 16, 256) -> (8, 8, 512).
        # self._enc_GAP = nn.AvgPool2d((8,8), 1) # GAP, (8, 8, 512) -> (1, 1, 512).

        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                      self._enc_conv_hidden_3, 
                                      self._enc_conv_hidden_4, 
                                      self._enc_conv_hidden_5, 
                                      self._enc_conv_hidden_6, 
                                      self._enc_conv_hidden_7)

        # Flatten(). 

        # Bottleneck
        self._enc_output = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._dec_fc_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, self._bottle_neck_dim),
                                              # nn.Dropout(0.5)
                                              ) # (16, ) -> (512, ). 

        # Unflatten(). 

        # Decoder. 
        self._dec_deconv_hidden_1 = nn.Sequential(nn.ConvTranspose2d(self._bottle_neck_dim, 512, kernel_size=(8,8)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        self._dec_deconv_hidden_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (16, 16, 256) -> (32, 32, 128). 
        self._dec_deconv_hidden_4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (32, 32, 128) -> (64, 64, 64). 
        self._dec_deconv_hidden_5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (64, 64, 64) -> (128, 128, 32). 
        self._dec_deconv_hidden_6 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (128, 128, 32) -> (256, 256, 16). 
        self._dec_deconv_hidden_7 = nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  nn.Sigmoid()) # (256, 256, 16) -> (512, 512, 1). 

        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                      self._dec_deconv_hidden_3,
                                      self._dec_deconv_hidden_4,
                                      self._dec_deconv_hidden_5,
                                      self._dec_deconv_hidden_6,
                                      self._dec_deconv_hidden_7)
    
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        output = output.view(output.size(0), -1) # Flatten. 

        latent = self._enc_output(output)

        return latent


    def decoder(self, latent):
        """
        """

        output = self._dec_fc_hidden_1(latent) # (16, ) -> (512, ). 
        output = output.view(output.size(0), self._bottle_neck_dim, 1, 1) # Flatten. (512, ) -> (1, 1, 512). 

        output = self._decoder(output)

        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        latent = self.encoder(x)
        output = self.decoder(latent)

        return output, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim
    

class Loss_Autoencoder(nn.Module):
    """
    """

    def __init__(self, reconstruct_mode=ML_VAE.LOSS_RECONSTRUCT_MODE):
        """
        """

        super(Loss_Autoencoder, self).__init__()
        self.reconstruct_mode = reconstruct_mode


    def forward(self, y, x):
        """
        Might change due to different types of sampled distribution. 

        y: generated image. 
        x: input/groundtruth image. 
        mu: mean vect of sampled distribution. 
        logvar: std of sampled distribution.
        """

        if self.reconstruct_mode == 'MSE': # For grayscale or binarized image. 
            reconstruct_loss = F.mse_loss(y.view(y.size(0),-1), x.view(x.size(0),-1), reduction='sum')
        elif self.reconstruct_mode == 'BCE': # For binarized image. 
            reconstruct_loss = F.binary_cross_entropy(y.view(y.size(0),-1), x.view(x.size(0),-1), reduction='sum')
        else: reconstruct_loss = 0

        return reconstruct_loss


class Autoencoder_Conv_test_128(nn.Module):
    """
    Only for validation. 
    """

    def __init__(self):
        """
        """

        super(Autoencoder_Conv_test_128, self).__init__()

        self._bottle_neck_dim = ML_VAE.BOTTLENECK_DIM
        self._latent_dim = ML_VAE.LATENT_DIM
        
        # Encoder. 
        self._enc_conv_hidden_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (512, 512, 1) -> (256, 256, 16). 
        self._enc_conv_hidden_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (256, 256, 16) -> (128, 128, 32). 
        self._enc_conv_hidden_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (128, 128, 32) -> (64, 64, 64). 
        self._enc_conv_hidden_4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (64, 64, 64) -> (32, 32, 128). 
        self._enc_conv_hidden_5 = nn.Sequential(nn.Conv2d(128, self._bottle_neck_dim, kernel_size=(8,8)), 
                                                ML_VAE.ACTIVATION_LAYER,
                                                ) # (64, 64, 64) -> (32, 32, 128). 
        
        self._encoder = nn.Sequential(self._enc_conv_hidden_1, 
                                      self._enc_conv_hidden_2, 
                                      self._enc_conv_hidden_3, 
                                      self._enc_conv_hidden_4, 
                                      self._enc_conv_hidden_5)

        # Flatten(). 

        # Bottleneck
        self._enc_output = nn.Linear(self._bottle_neck_dim, self._latent_dim)
        self._dec_fc_hidden_1 = nn.Sequential(nn.Linear(self._latent_dim, self._bottle_neck_dim),
                                              # nn.Dropout(0.5)
                                              ) # (16, ) -> (512, ). 
        
        self._dec_deconv_hidden_1 = nn.Sequential(nn.ConvTranspose2d(self._bottle_neck_dim, 128, kernel_size=(8,8)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (1, 1, 512) -> (8, 8, 512). 
        self._dec_deconv_hidden_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (8, 8, 512) -> (16, 16, 256). 
        self._dec_deconv_hidden_3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (16, 16, 256) -> (32, 32, 128). 
        self._dec_deconv_hidden_4 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (32, 32, 128) -> (64, 64, 64). 
        self._dec_deconv_hidden_5 = nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                                                  ML_VAE.ACTIVATION_LAYER,
                                                  ) # (64, 64, 64) -> (128, 128, 32). 
        
        self._decoder = nn.Sequential(self._dec_deconv_hidden_1,
                                      self._dec_deconv_hidden_2,
                                      self._dec_deconv_hidden_3,
                                      self._dec_deconv_hidden_4,
                                      self._dec_deconv_hidden_5)
    
    def encoder(self, x):
        """
        """

        output = self._encoder(x)
        output = output.view(output.size(0), -1) # Flatten. 

        latent = self._enc_output(output)

        return latent


    def decoder(self, latent):
        """
        """

        output = self._dec_fc_hidden_1(latent) # (16, ) -> (512, ). 
        output = output.view(output.size(0), self._bottle_neck_dim, 1, 1) # Flatten. (512, ) -> (1, 1, 512). 

        output = self._decoder(output)

        return output # The generated batch of image. 


    def forward(self, x):
        """
        """

        latent = self.encoder(x)
        output = self.decoder(latent)

        return output, latent


    @property
    def latent_dim(self):
        """
        """

        return self._latent_dim


class Loss_VAE(nn.Module):
    """
    """

    def __init__(self, loss_beta=1., reconstruct_mode=ML_VAE.LOSS_RECONSTRUCT_MODE):
        """
        """

        super(Loss_VAE, self).__init__()
        self.loss_beta = loss_beta # Hyperparameter for controlling weights of the KL-divergence loss function term. 
        self.reconstruct_mode = reconstruct_mode


    def forward(self, y, x, mu, logvar):
        """
        Might change due to different types of sampled distribution. 

        y: generated image. 
        x: input/groundtruth image. 
        mu: mean vect of sampled distribution. 
        logvar: std of sampled distribution.
        """

        if self.reconstruct_mode == 'MSE': # For grayscale or binarized image. 
            reconstruct_loss = F.mse_loss(y.view(y.size(0),-1), x.view(x.size(0),-1), reduction='sum')
            # reconstruct_loss = F.mse_loss(y, x, reduction='sum')
        elif self.reconstruct_mode == 'BCE': # For binarized image. 
            reconstruct_loss = F.binary_cross_entropy(y.view(y.size(0),-1), x.view(x.size(0),-1), reduction='sum')
        else: reconstruct_loss = 0

        # MSE_loss = nn.MSELoss(y, x) # For grayscale or binarized image. 
        # BCE_loss = F.binary_cross_entropy(y.view(y.size(0),-1), x.view(x.size(0),-1), reduction='sum') # For binarized image. 

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) # Mimic the unit normal distribution. 
        KLD_loss = -0.5 * self.loss_beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD_loss = torch.sum(logvar.add_(1).add_(-mu.pow(2)).add_(-logvar.exp())).mul_(-0.5).mul_(self.loss_beta)

        return reconstruct_loss + KLD_loss