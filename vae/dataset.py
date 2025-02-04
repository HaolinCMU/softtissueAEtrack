# -*- coding: utf-8 -*-
"""
Created on Sun May 22 04:33:47 2022

@author: hlinl
"""


import copy
import glob
import os
import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL

from collections import defaultdict
from PIL import ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from PARAM import *


class DeformationAutoencoderDataset(Dataset):
    """
    """
    
    def __init__(self, data_matrix, dtype=torch.float, transform=Compose([ToTensor()])):
        """
        Parameters:
        ----------
            data_matrix: 2D array of Float. Dim: nDOF(node_num*3-9) x SampleNum. 
        """
        
        self.data_matrix = data_matrix
        self.dtype = dtype
        self.transform = transform
        
        self._dataset_size = 0
        self._input_data_dict = {} # {`ind`->int: [`deformation vector`->1d Float array]}. 
        self._output_data_dict = {} # {`ind`->int: [`deformation vector`->1d Float array]}. 
        
        self._set_data_repo()
        
    
    def __len__(self):
        """
        """
        
        return self._dataset_size
    

    def __getitem__(self, index):
        """
        """
        
        index = copy.deepcopy(str(index).zfill(5))

        sample = {'input': self.transform(self._input_data_dict[index]).to(self.dtype), 
                  'output': self.transform(self._output_data_dict[index]).to(self.dtype)}
        return sample
    

    @property
    def input_data_repo_dict(self):
        """
        """

        return self._input_data_dict
    

    @property
    def output_data_repo_dict(self):
        """
        """

        return self._output_data_dict
    
    
    def _set_data_repo(self):
        """
        """

        # ---------- Set input data repo ----------
        self._dataset_size = 0 # Re-initialize dataset size.  
        for ind in range(self.data_matrix.shape[1]):
            self._input_data_dict[str(ind).zfill(5)] = self.data_matrix[:,ind].reshape(1,-1) # 1d Float array. Dim: (1, node_num*3). 
            self._output_data_dict[str(ind).zfill(5)] = self.data_matrix[:,ind].reshape(1,-1) # 1d Float array. Dim: (1, node_num*3). 
            self._dataset_size += 1


class FrameAutoencoderDataset(Dataset):
    """
    """

    def __init__(self, input_data_dir, output_data_dir, img_pattern='png', dtype=torch.float32, max_dataset_size=None, 
                 input_image_transform=Compose([Resize(ML_VAE.INPUT_IMAGE_SIZE), ToTensor()]), 
                 output_image_transform=Compose([Resize(ML_VAE.OUTPUT_IMAGE_SIZE), ToTensor()])):
        """
        Expected data directory structure: folder (data_dir) -> subfolders (layers) -> images (frames). 
        """

        self.input_data_dir = input_data_dir # The total directory of all input images for the autoencoder. 
        self.output_data_dir = output_data_dir # The total directory of all output images for the autoencoder. 
        self.img_pattern = img_pattern
        self.image_dtype = dtype
        self.input_image_transform = input_image_transform
        self.output_image_transform = output_image_transform
        self.max_dataset_size = max_dataset_size
        
        # Data repos (path lists) & Data label dictionaries. 
        self._dataset_size = 0 
        self._input_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 
        self._output_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 

        self._set_data_repo()
    

    def __len__(self):
        """
        """

        return self._dataset_size


    def __getitem__(self, index):
        """
        """

        index = copy.deepcopy(str(int(index)))

        # input_img = PIL.Image.open(self._input_image_label_dict[index][2])
        # output_img = PIL.Image.open(self._output_image_label_dict[index][2])

        input_img = PIL.Image.fromarray(np.uint8(mig.imread(self._input_image_label_dict[index][2])*255))
        output_img = PIL.Image.fromarray(np.uint8(mig.imread(self._output_image_label_dict[index][2])*255))
        # input_img = PIL.Image.fromarray(np.uint8(mig.imread(self._input_image_label_dict[index][2])[128:384,:]*255))
        # output_img = PIL.Image.fromarray(np.uint8(mig.imread(self._output_image_label_dict[index][2])[128:384,:]*255))
        folder_name = self._output_image_label_dict[index][0]
        infolder_ind = self._output_image_label_dict[index][1]

        if self.input_image_transform:
            input_img = copy.deepcopy(self.input_image_transform(input_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 
        if self.output_image_transform:
            output_img = copy.deepcopy(self.output_image_transform(output_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 

        # # ---------- Reshape tensors to make them compatible with NN ---------- 
        # input_img_c, input_img_h, input_img_w = input_img.size()
        # output_img_c, output_img_h, output_img_w = output_img.size()

        # input_img = input_img.view(-1, input_img_c, input_img_h, input_img_w)
        # output_img = output_img.view(-1, output_img_c, output_img_h, output_img_w)

        sample = {'input': input_img, 'output': output_img, 'folder_name': folder_name, 'file_No': infolder_ind}

        return sample
    

    @property
    def input_data_repo_dict(self):
        """
        """

        return self._input_image_label_dict
    

    @property
    def output_data_repo_dict(self):
        """
        """

        return self._output_image_label_dict
    

    def _set_data_repo(self):
        """
        """

        # ---------- Set input data repo ---------- 
        input_image_subfolder_list = os.listdir(self.input_data_dir)
        input_image_subdir_list = glob.glob(os.path.join(self.input_data_dir, "*"))
        input_image_accum_num = 0 # Used for establishing the image label dict. 
        
        if self.max_dataset_size is None: max_dataset_size_perfolder_input = None
        else: max_dataset_size_perfolder_input = self.max_dataset_size // len(input_image_subdir_list)

        for ind, input_image_subdir in enumerate(input_image_subdir_list):
            input_image_subfolder_name_temp = input_image_subfolder_list[ind]
            input_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(input_image_subdir, 
                                                                              "*.{}".format(self.img_pattern))))

            if max_dataset_size_perfolder_input is None: input_image_pick_every = 1
            else:
                if max_dataset_size_perfolder_input >= len(input_image_path_list_temp): input_image_pick_every = 1
                else: input_image_pick_every = len(input_image_path_list_temp) // max_dataset_size_perfolder_input
                
            # Establish image label dict. 
            for i, path in enumerate(input_image_path_list_temp):
                if i % input_image_pick_every != 0: continue
                self._input_image_label_dict[str(int(input_image_accum_num+(i//input_image_pick_every)))] = [input_image_subfolder_name_temp, i, path]
            
            if len(input_image_path_list_temp) % input_image_pick_every == 0: 
                input_image_accum_num += len(input_image_path_list_temp) // input_image_pick_every
            else: input_image_accum_num += (len(input_image_path_list_temp) // input_image_pick_every + 1)

        # ---------- Set output data repo ---------- 
        output_image_subfolder_list = os.listdir(self.output_data_dir)
        output_image_subdir_list = glob.glob(os.path.join(self.output_data_dir, "*"))
        output_image_accum_num = 0 # Used for establishing the image label dict. 
        
        if self.max_dataset_size is None: max_dataset_size_perfolder_output = None
        else: max_dataset_size_perfolder_output = self.max_dataset_size // len(output_image_subdir_list)

        for ind, output_image_subdir in enumerate(output_image_subdir_list):
            output_image_subfolder_name_temp = output_image_subfolder_list[ind]
            output_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(output_image_subdir, 
                                                                               "*.{}".format(self.img_pattern))))
            
            if max_dataset_size_perfolder_output is None: output_image_pick_every = 1
            else:
                if max_dataset_size_perfolder_output >= len(output_image_path_list_temp): output_image_pick_every = 1
                else: output_image_pick_every = len(output_image_path_list_temp) // max_dataset_size_perfolder_output

            # Establish image label dict. 
            for i, path in enumerate(output_image_path_list_temp):
                if i % output_image_pick_every != 0: continue
                self._output_image_label_dict[str(int(output_image_accum_num+(i//output_image_pick_every)))] = [output_image_subfolder_name_temp, i, path]

            if len(output_image_path_list_temp) % output_image_pick_every == 0:
                output_image_accum_num += len(output_image_path_list_temp) // output_image_pick_every
            else: output_image_accum_num += (len(output_image_path_list_temp) // output_image_pick_every + 1)
        
        self._dataset_size = min(input_image_accum_num, output_image_accum_num)


    def extract(self, ind_array):
        """
        Return info of input and output data that corresponds to the given `ind_array` with the exact order. 
        """

        ind_array = copy.deepcopy(ind_array.astype(str).reshape(-1))
        
        # Input. 
        input_image_subfolder_list = [self._input_image_label_dict[ind][0] for ind in ind_array]
        input_image_ind_list = [self._input_image_label_dict[ind][1] for ind in ind_array]
        input_image_path_list = [self._input_image_label_dict[ind][2] for ind in ind_array]

        # Output. 
        output_image_subfolder_list = [self._output_image_label_dict[ind][0] for ind in ind_array]
        output_image_ind_list = [self._output_image_label_dict[ind][1] for ind in ind_array]
        output_image_path_list = [self._output_image_label_dict[ind][2] for ind in ind_array]

        return (input_image_subfolder_list, input_image_ind_list, input_image_path_list,
                output_image_subfolder_list, output_image_ind_list, output_image_path_list)


def set_dataloader(dataset_obj, batch_size):
    """_summary_
    """
    
    dataset_size = len(dataset_obj)
    dataset_indices_total = list(range(dataset_size))
    # np.random.shuffle(dataset_indices_total)
    data_sampler = torch.utils.data.SequentialSampler(dataset_indices_total) # Cannot be random shuffler. 
    
    return torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, sampler=data_sampler)