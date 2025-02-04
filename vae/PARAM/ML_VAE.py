# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:35:35 2021

@author: hlinl
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils


# VAE & Autoencoder PARAMS. 
# INPUT_IMAGE_SIZE = [128, 128] # # [h, w]. The image size of the input image. [512, 512].  
# OUTPUT_IMAGE_SIZE = [128, 128] # # [h, w]. The image size of the output image. [512, 512]. 
INPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the input image. [512, 512].  
OUTPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the output image. [512, 512].  

INPUT_DIM = 3465 # 1158 * 3 - 9 (fixed points=3). `nDOF`. `
OUTPUT_DIM = 3465

# INPUT_IMAGE_DATA_DIR = "D:/image_processing/data/processed/highspeed/straighten"
# OUTPUT_IMAGE_DATA_DIR = "D:/image_processing/data/processed/highspeed/straighten"
INPUT_IMAGE_DATA_DIR = "D:/image_processing/data/vae_train_dataset"
OUTPUT_IMAGE_DATA_DIR = "D:/image_processing/data/vae_train_dataset"

INPUT_IMAGE_EVAL_DATA_DIR = "D:/image_processing/data/vae_eval_dataset"
OUTPUT_IMAGE_EVAL_DATA_DIR = "D:/image_processing/data/vae_eval_dataset"

MAX_DATASET_SIZE = None # Default: None. could be a number like 72000. 
BATCH_SIZE = 128
NUM_EPOCHS = 8000 # Default: 5. Debug: 1. Short: 5. Medium: 15 - 30. Long: 50. 
LOSS_BETA = 0 # Default: 0 or 1e-4.  # Penalty factor that controls weights of the KL-divergence loss function term. Used only for VAE. 
LOSS_RECONSTRUCT_MODE = 'MSE' # 'MSE' (general) or 'BCE' (binarized-specific). 
LAMBDA_REGLR = 1e-5 # Set regularization in optimizer. 0 for not applying regularization. 

LEARNING_RATE = 1e-4 # Default: 1e-5. 
LEARNING_RATE_SCHEDULE_PERIOD = 5 # Default: 5. 
LEARNING_RATE_DECAY_FACTOR = 1 # Default: 1. Change it to a number within [0., 1.] to define learning rate decaying rate. 

MODEL_ARXIV_DIR = "model_checkpoints_autoencoder"
TRAINING_LOG_SAVEPATH = "vae_train.log"
TRAIN_VALID_LOSS_SAVEPATH = "Train_Valid_Loss_VAE.png"

ACTIVATION_LAYER = nn.LeakyReLU() # Default: nn.ReLU(). 

LATENT_DIM = 27 # Default: 27. # 6. For soft tissue tracking cases, it should be equal to the `PC_num`. 
BOTTLENECK_DIM = 64 # Default: 512. Could be 1024. # 128. 

MODEL_CHECKPOINT_EPOCH_NUM = 1000 # The epoch number for periodically archiving the intermediate 'checkpoint' models. Default: 5. 

# ------------- Temp -------------
MLP_FIRST_LAYER_NUM = 512
MLP_LAYER_NUM_DECAY_DIV = 2
