# -*- coding: utf-8 -*-
"""
Created on Sat May 21 00:53:45 2022

@author: hlinl
"""


import os
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))
import time

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

from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from torch.utils.data import Dataset, DataLoader

from PARAM import *
from model import *

from dataset import *
from files import *


class Model_VAE(object):
    """
    """
    
    def __init__(self, data_matrix, batch_size, learning_rate, num_epochs, loss_beta=1., 
                 valid_ratio=.1, test_ratio=.1, model_arxiv_dir=ML_VAE.MODEL_ARXIV_DIR, 
                 log_path=ML_VAE.TRAINING_LOG_SAVEPATH):
        """
        """
        
        self.data_matrix = data_matrix # String. Directory of input image data. 
        self.batch_size = batch_size # Int. Batch size for training and validation dataset. 
        self.learning_rate = learning_rate # Float. Initial learning rate. Constant if no learning rate schedule is applied. 
        self.num_epochs = num_epochs # Int. Epoch number. 
        self.loss_beta = loss_beta # Float. Hyperparameter for controlling weights of the KL-divergence loss function term.  
        self.valid_ratio = valid_ratio # Float. Fraction of validation dataset. Used for dataset partition. 
        self.test_ratio = test_ratio # Float. Fraction of testing dataset. Used for dataset partition. 
        self.model_arxiv_dir = model_arxiv_dir # String. Directory for saving the intermediate and final trained neural network models. 
        self.log_path = log_path # String. Path for saving the training log file. 

        self._is_cuda = torch.cuda.is_available() # Bool. Indicate whether an available cuda is installed. 
        self._device = torch.device('cuda' if self._is_cuda else 'cpu') # Torch.device. Indicate 'cuda' or 'cpu'. 
        # self._device = torch.device('cpu') # Torch.device. 'cpu' only. 

        # Dataset & Dataloaders. 
        self.dataset = DeformationAutoencoderDataset(self.data_matrix) # Torch.Dataset. Dataset object for initializing image dataset. 
        
        self._train_set_indices = None # List of Int. The (shuffled) indices of training dataset. 
        self._valid_set_indices = None # List of Int. The (shuffled) indices of validation dataset. 
        self._test_set_indices = None # List of Int. The (shuffled) indices of testing dataset. 

        self.train_loader = None # Torch.Dataloader. The loader of training dataset. 
        self.valid_loader = None # Torch.Dataloader. The loader of validation dataset. 
        self.test_loader = None # Torch.Dataloader. The loader of testing dataset. 

        self.init_dataLoaders() # Initialize, partition and create train, valid and test dataloaders. 

        # Learning model. 
        # self.vae_net = vae.VAE_Linear().to(self._device) # Torch.nn.Module. Initialize a VAE neural network object. Send to `device`.
        self.vae_net = vae.AE_Linear().to(self._device) # Torch.nn.Module. Initialize a VAE neural network object. Send to `device`.
        self._weight_init(self.vae_net) # Initialize weight and bias of the neural network using 'xavier_normal' and 'zeros' methods, respectively. 
        self._loss_func = vae.Loss_VAE(loss_beta=self.loss_beta) # Torch.nn.Module. Self-defined loss function for VAE neural network. 
        self.epoch_loss_list_train = [] # List of Float. Training loss value of each epoch. 
        self.epoch_loss_list_valid = [] # List of Float. Validation loss value of each epoch. 
        self.batch_loss_list_train = [] # List of Float. Training loss value of each batch. 
        self.batch_loss_list_valid = [] # List of Float. Validation loss value of each batch. 
        self.lr_list = [] # List of Float. Learning rate of each epoch. All the same if learning rate schedule is not applied. 

        self._training_log = [] # List of String. List of lines of training log. 
        self._training_time_total = 0 # Float. Total training time. 
        
    
    @property
    def train_set_ind_array(self):
        """
        """

        if self._train_set_indices is not None: 
            return np.array(self._train_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Training dataloader partition and generation failed. ")
    

    @property
    def valid_set_ind_array(self):
        """
        """

        if self._valid_set_indices is not None: 
            return np.array(self._valid_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Validation dataloader partition and generation failed. ")
    

    @property
    def test_set_ind_array(self):
        """
        """

        if self._test_set_indices is not None: 
            return np.array(self._test_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Testing dataloader partition and generation failed. ")


    @staticmethod
    def _weight_init(net):
        """
        """

        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=1)
                nn.init.zeros_(layer.bias.data)

            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data, gain=1)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data, gain=1)

            else: pass

    
    def _set_datasets(self, train_set_ind_list=None, valid_set_ind_list=None, test_set_ind_list=None):
        """
        Initialize and partition dataset into train_set, valid_set and test_set. 
        """
        
        dataset_size = len(self.dataset)

        # Partition strategy: [ test | valid | train ]
        dataset_indices_total = list(range(dataset_size))
        np.random.shuffle(dataset_indices_total)
        # split_pt_test = int(np.floor(self.test_ratio*dataset_size))
        # split_pt_valid = int(np.floor((self.test_ratio+self.valid_ratio)*dataset_size))

        # if train_set_ind_list is not None: self._train_set_indices = copy.deepcopy(train_set_ind_list)
        # else: self._train_set_indices = dataset_indices_total[split_pt_valid:]

        # if valid_set_ind_list is not None: self._valid_set_indices = copy.deepcopy(valid_set_ind_list)
        # else: self._valid_set_indices = dataset_indices_total[split_pt_test:split_pt_valid]

        # if test_set_ind_list is not None: self._test_set_indices = copy.deepcopy(test_set_ind_list)
        # else: self._test_set_indices = dataset_indices_total[:split_pt_test]
        
        split_pt_train = int(np.floor((1.-self.test_ratio-self.valid_ratio)*dataset_size))
        split_pt_valid = int(np.floor((1.-self.test_ratio)*dataset_size))
        
        if train_set_ind_list is not None: self._train_set_indices = copy.deepcopy(train_set_ind_list)
        else: self._train_set_indices = dataset_indices_total[:split_pt_train]

        if valid_set_ind_list is not None: self._valid_set_indices = copy.deepcopy(valid_set_ind_list)
        else: self._valid_set_indices = dataset_indices_total[split_pt_train:split_pt_valid]

        if test_set_ind_list is not None: self._test_set_indices = copy.deepcopy(test_set_ind_list)
        else: self._test_set_indices = dataset_indices_total[split_pt_valid:]


    def init_dataLoaders(self, train_set_ind_list=None, valid_set_ind_list=None, test_set_ind_list=None):
        """
        """

        self._set_datasets(train_set_ind_list, valid_set_ind_list, test_set_ind_list) # Partition the dataset. 

        train_set_sampler = torch.utils.data.SubsetRandomSampler(self._train_set_indices)
        valid_set_sampler = torch.utils.data.SubsetRandomSampler(self._valid_set_indices)
        test_set_sampler = torch.utils.data.SubsetRandomSampler(self._test_set_indices)

        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_set_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=valid_set_sampler)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, sampler=test_set_sampler)


    def train(self):
        """
        """

        if not os.path.isdir(self.model_arxiv_dir): os.mkdir(self.model_arxiv_dir)
        clr_dir(self.model_arxiv_dir) # Clear the directory of pre-saved trained models before starting a new batch of training. 

        # Define criterion and optimizer
        optimizer = torch.optim.Adam(self.vae_net.parameters(), self.learning_rate, 
                                     weight_decay=ML_VAE.LAMBDA_REGLR)
        
        # Iterative training and validation
        print("##############################")
        print("Starting training....")
        self._training_log.append("({}) Starting training....".format(str(datetime.today())))
        self._training_log.append("--------------------")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            # ---------- Apply learning rate decaying schedule ---------- 
            if epoch != 0 and epoch % ML_VAE.LEARNING_RATE_SCHEDULE_PERIOD == 0: 
                for p in optimizer.param_groups: p['lr'] *= ML_VAE.LEARNING_RATE_DECAY_FACTOR
            self.lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            
            # ---------- Training ----------
            loss_train_perEpoch = 0
            for iter, batch in enumerate(self.train_loader):
                # Forward. 
                inputs_train, groundtruths_train = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
                # print(inputs_train.cpu().data.numpy().astype(float)[0,:])
                batch_size_temp = groundtruths_train.size(0) # Change to 1 if not intended for getting batch-averaged loss. 

                inputs_train = inputs_train.to(self._device)
                groundtruths_train = groundtruths_train.to(self._device)
                
                self.vae_net.train()
                y, mu, logvar, _ = self.vae_net(inputs_train)
                loss_train_thisBatch = self._loss_func(y, groundtruths_train, mu, logvar) # Total loss of one batch. 
                
                # Back propagation. 
                optimizer.zero_grad()
                loss_train_thisBatch.backward()
                optimizer.step()

                loss_train_thisBatch_perSample = copy.deepcopy(float(loss_train_thisBatch.item()/batch_size_temp))
                self.batch_loss_list_train.append(loss_train_thisBatch_perSample) # Average loss on batch size to obtain loss per sample. 
                loss_train_perEpoch += loss_train_thisBatch_perSample # Adding up averaged loss per sample. 

                # Check training loss every batch. 
                if iter == 0 or (iter+1) % 5 == 0 or iter + 1 == len(self.train_loader):
                    batch_line_temp = "Epoch: [{}/{}]\t| Batch: [{}/{}]\t| Loss: {:.4f}\t| Time: {:.4f} s". \
                                       format(epoch+1, self.num_epochs, iter+1, len(self.train_loader), 
                                              loss_train_thisBatch_perSample, time.time()-start_time)
                    self._training_log.append(batch_line_temp)
                    print(batch_line_temp)
            
            loss_train_perEpoch /= len(self.train_loader)
            self.epoch_loss_list_train.append(loss_train_perEpoch) # Average loss on batch number to obtain loss per epoch per batch. 
            
            # ---------- Validation ----------
            loss_valid_perEpoch = 0
            for _, batch in enumerate(self.valid_loader):
                inputs_valid, groundtruths_valid = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
                batch_size_temp = groundtruths_valid.size(0) # Change to 1 if not intended for getting batch-averaged loss. 

                inputs_valid = inputs_valid.to(self._device)
                groundtruths_valid = groundtruths_valid.to(self._device)
                
                self.vae_net.eval()
                with torch.no_grad():
                    y, mu, logvar, _ = self.vae_net(inputs_valid)
                    loss_valid_perBatch_perEpoch = self._loss_func(y, groundtruths_valid, mu, logvar)

                    loss_valid_perEpoch_perSample = copy.deepcopy(float(loss_valid_perBatch_perEpoch.item()/batch_size_temp))
                    self.batch_loss_list_valid.append(loss_valid_perEpoch_perSample) # Average loss on batch size to obtain loss per sample. 
                    loss_valid_perEpoch += loss_valid_perEpoch_perSample # Adding up averaged loss per sample. 
            
            loss_valid_perEpoch /= len(self.valid_loader)
            self.epoch_loss_list_valid.append(loss_valid_perEpoch) # Average loss on batch number to obtain loss per epoch per batch. 

            # ---------- Wrap-up this epoch ----------
            epoch_line_temp = "Epoch Number: {}\t| Train Loss: {:.4f}\t| Valid Loss: {:.4f}\t| Time Elapsed: {:.4f} s". \
                               format(int(epoch+1), loss_train_perEpoch, loss_valid_perEpoch, time.time()-start_time)
            self._training_log.append(epoch_line_temp)
            self._training_log.append("--------------------")
            print(epoch_line_temp)
            print("--------------------")

            # Save trained intermediate models every certain epochs. 
            if (epoch+1) % ML_VAE.MODEL_CHECKPOINT_EPOCH_NUM == 0:
                model_savePath_temp = os.path.join(self.model_arxiv_dir, "model_epoch_{}.pth".format(epoch+1))
                torch.save(self.vae_net, model_savePath_temp)
            
            torch.cuda.empty_cache() # Clear unnecessary memory allocations. 
        
        # ---------- Summary & Save ----------
        self._training_time_total = time.time() - start_time
        print("Training completed. ")
        print("##############################")
        # torch.save(self.vae_net.state_dict(), os.path.join(self.model_arxiv_dir, "model_final.pkl")) # Save the final trained model.
        torch.save(self.vae_net, os.path.join(self.model_arxiv_dir, "model_final.pth")) # Save the final trained model.
        self._training_log.append("({}) Training completed. ".format(str(datetime.today())))
        self._save_training_log() # Save a training log. 


    def evaluate(self, model, test_loader):
        """
        """

        loss_list, groundtruths_list, generations_list = [], [], []
        mu_list, logvar_list, latent_list = [], [], []
        
        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(test_loader): 
                inputs_test, groundtruths_test = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
                batch_size_temp, c, h, w = groundtruths_test.size()
                
                inputs_test = inputs_test.to(self._device)
                groundtruths_test = groundtruths_test.to(self._device)
                
                y, mu, logvar, latent = model(inputs_test)
                loss_test_perBatch = self._loss_func(y, groundtruths_test, mu, logvar)

                # print(groundtruths_test.cpu().numpy()[0,:].reshape(-1)[:5]/100.)
                
                loss_list.append(float(loss_test_perBatch.item()/batch_size_temp)) # Loss per test sample. 
                groundtruths_list += [groundtruths_test.cpu().numpy()[i,:].reshape(-1) for i in range(batch_size_temp)]
                generations_list += [y.cpu().numpy()[i,:].reshape(-1) for i in range(batch_size_temp)]
                mu_list += [mu.cpu().numpy()[i,:].reshape(-1) for i in range(batch_size_temp)]
                logvar_list += [logvar.cpu().numpy()[i,:].reshape(-1) for i in range(batch_size_temp)]
                latent_list += [latent.cpu().numpy()[i,:].reshape(-1) for i in range(batch_size_temp)]
        
        return loss_list, groundtruths_list, generations_list, mu_list, logvar_list, latent_list

    
    def loss_plot(self):
        """
        """

        np.save("epoch_loss_list_train.npy", self.epoch_loss_list_train)
        np.save("epoch_loss_list_valid.npy", self.epoch_loss_list_valid)
        np.save("batch_loss_list_train.npy", self.batch_loss_list_train)
        np.save("batch_loss_list_valid.npy", self.batch_loss_list_valid)

        epoch_loss_train_array = np.log(np.array(self.epoch_loss_list_train).astype(float).reshape(-1))
        epoch_loss_valid_array = np.log(np.array(self.epoch_loss_list_valid).astype(float).reshape(-1))
        train_epochs_ind_array = np.linspace(0, self.num_epochs, num=len(epoch_loss_train_array))
        valid_epochs_ind_array = np.linspace(0, self.num_epochs, num=len(epoch_loss_valid_array))

        batch_loss_train_array = np.log(np.array(self.batch_loss_list_train).astype(float).reshape(-1))
        batch_loss_valid_array = np.log(np.array(self.batch_loss_list_valid).astype(float).reshape(-1))
        train_batch_ind_array = np.linspace(0, self.num_epochs, num=len(batch_loss_train_array))
        valid_batch_ind_array = np.linspace(0, self.num_epochs, num=len(batch_loss_valid_array))

        plt.figure(figsize=(20,20))
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        line1, = plt.plot(train_batch_ind_array, batch_loss_train_array, 
                          color='blue', linewidth=10.0, alpha=0.3, label="Train Loss (log) - Batch")
        line2, = plt.plot(valid_batch_ind_array, batch_loss_valid_array, 
                          color='orange', linewidth=10.0, alpha=0.3, label="Valid Loss (log) - Batch")
        line3, = plt.plot(train_epochs_ind_array, epoch_loss_train_array, 
                          color='blue', linewidth=3.0, label="Train Loss (log) - Epoch")
        line4, = plt.plot(valid_epochs_ind_array, epoch_loss_valid_array, 
                          color='orange', linewidth=3.0, label="Valid Loss (log) - Epoch")
        plt.xlabel("Epochs", fontsize=40)
        plt.ylabel("log(Loss)", fontsize=40)
        plt.legend([line1,line2,line3,line4], ["Train Loss (log) - Batch", 
                                               "Valid Loss (log) - Batch",
                                               "Train Loss (log) - Epoch", 
                                               "Valid Loss (log) - Epoch",], prop={"size": 40})
        plt.title("Train & Valid Loss v/s Epochs")
        plt.savefig("Train_Valid_Loss_VAE.png")


    @property
    def training_time(self):
        """
        """

        return self._training_time_total


    def showcase_generation(self, ind=None):
        """
        """

        pass


    def sample_from_latent(self):
        """
        """

        pass


    def generate(self, latent):
        """
        """

        pass


    def _save_training_log(self):
        """
        Save framework setup and training info. as a '.log' file. 
        """

        heading = ["Date and Time: {}".format(str(datetime.today())), 
                   "##############################",
                   "Batch size: {}".format(self.batch_size),
                   "Learning rate: {}".format(self.learning_rate),
                   "Epoch number: {}".format(self.num_epochs),
                   "Loss penalty factor: {}".format(self.loss_beta),
                   "Reconstruction loss mode: {}".format(ML_VAE.LOSS_RECONSTRUCT_MODE),  
                   "Dataset size: {}".format(len(self.dataset)),
                   "Train/Valid/Test: {:.2f}/{:.2f}/{:.2f}".format(1.-self.test_ratio-self.valid_ratio, 
                                                                   self.valid_ratio, self.test_ratio), 
                   "Latent dimension: {}".format(int(self.vae_net.latent_dim)), 
                   "##############################"
                   ]

        self._training_log.append("##############################"),
        self._training_log.append("Total training time: {:.4f}. ".format(self._training_time_total))

        content = copy.deepcopy(heading + self._training_log)

        with open(self.log_path, 'w') as f:
            content = '\n'.join(content)
            f.write(content)
            f.close()


class Model_AE(object):
    """
    Autoencoder training and evaluation framework. 
    """

    def __init__(self, input_data_dir, output_data_dir, max_dataset_size, batch_size, learning_rate, num_epochs, 
                 valid_ratio=.05, test_ratio=.15, model_arxiv_dir=ML_VAE.MODEL_ARXIV_DIR, 
                 log_path=ML_VAE.TRAINING_LOG_SAVEPATH):
        """
        Initialize an Autoencoder training framework with the given arguemnts. 

        Parameters:
        ----------
            input_data_dir: String. 
                Directory of input image data. 
            output_data_dir: String. 
                Directory of output image data. 
            batch_size: Int. 
                Batch size for training and validation dataset. 
            learning_rate: Float. 
                Initial learning rate. Constant if no learning rate schedule is applied. 
            num_epochs: Int. 
                Epoch number. 
            valid_ratio: Float. 
                Fraction of validation dataset. Used for dataset partition.
                Default: .05. 
            test_ratio: Float. 
                Fraction of testing dataset. Used for dataset partition. 
                Default: .15. 
            model_arxiv_dir: String. 
                Directory for saving the intermediate and final trained neural network models.
                Default: `ML_VAE.MODEL_ARXIV_DIR`. 
            log_path: String. 
                Path for saving the training log file. 
                Default: `ML_VAE.TRAINING_LOG_SAVEPATH`. 

        Return:
        ----------
            None.
        """

        self.input_data_dir = input_data_dir # String. Directory of input image data. 
        self.output_data_dir = output_data_dir # String. Directory of output image data. 
        self.max_dataset_size = max_dataset_size
        self.batch_size = batch_size # Int. Batch size for training and validation dataset. 
        self.learning_rate = learning_rate # Float. Initial learning rate. Constant if no learning rate schedule is applied. 
        self.num_epochs = num_epochs # Int. Epoch number. 
        self.valid_ratio = valid_ratio # Float. Fraction of validation dataset. Used for dataset partition. 
        self.test_ratio = test_ratio # Float. Fraction of testing dataset. Used for dataset partition. 
        self.model_arxiv_dir = model_arxiv_dir # String. Directory for saving the intermediate and final trained neural network models. 
        self.log_path = log_path # String. Path for saving the training log file. 

        self._is_cuda = torch.cuda.is_available() # Bool. Indicate whether an available cuda is installed. 
        self._device = torch.device('cuda' if self._is_cuda else 'cpu') # Torch.device. Indicate 'cuda' or 'cpu'. 
        # self._device = torch.device('cpu') # Torch.device. 'cpu' only. 

        # Dataset & Dataloaders. 
        self.dataset = FrameAutoencoderDataset(self.input_data_dir, self.output_data_dir, max_dataset_size=self.max_dataset_size) # Torch.Dataset. Dataset object for initializing image dataset. 
        
        self._train_set_indices = None # List of Int. The (shuffled) indices of training dataset. 
        self._valid_set_indices = None # List of Int. The (shuffled) indices of validation dataset. 
        self._test_set_indices = None # List of Int. The (shuffled) indices of testing dataset. 

        self.train_loader = None # Torch.Dataloader. The loader of training dataset. 
        self.valid_loader = None # Torch.Dataloader. The loader of validation dataset. 
        self.test_loader = None # Torch.Dataloader. The loader of testing dataset. 

        self.init_dataLoaders() # Initialize, partition and create train, valid and test dataloaders. 

        # Learning model. 
        self.autoencoder = vae.Autoencoder_Conv_test().to(self._device) # Torch.nn.Module. Initialize a VAE neural network object. Send to `device`.
        self._weight_init(self.autoencoder) # Initialize weight and bias of the neural network using 'xavier_normal' and 'zeros' methods, respectively. 
        self._loss_func = vae.Loss_Autoencoder() # Torch.nn.Module. Self-defined loss function for VAE neural network. 
        self.epoch_loss_list_train = [] # List of Float. Training loss value of each epoch. 
        self.epoch_loss_list_valid = [] # List of Float. Validation loss value of each epoch. 
        self.batch_loss_list_train = [] # List of Float. Training loss value of each batch. 
        self.batch_loss_list_valid = [] # List of Float. Validation loss value of each batch. 
        self.lr_list = [] # List of Float. Learning rate of each epoch. All the same if learning rate schedule is not applied. 

        self._training_log = [] # List of String. List of lines of training log. 
        self._training_time_total = 0 # Float. Total training time. 


    @property
    def train_set_ind_array(self):
        """
        """

        if self._train_set_indices is not None: 
            return np.array(self._train_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Training dataloader partition and generation failed. ")
    

    @property
    def valid_set_ind_array(self):
        """
        """

        if self._valid_set_indices is not None: 
            return np.array(self._valid_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Validation dataloader partition and generation failed. ")
    

    @property
    def test_set_ind_array(self):
        """
        """

        if self._test_set_indices is not None: 
            return np.array(self._test_set_indices).astype(str).reshape(-1)
        else: raise ValueError("Testing dataloader partition and generation failed. ")


    @staticmethod
    def _weight_init(net):
        """
        """

        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=1)
                nn.init.zeros_(layer.bias.data)

            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data, gain=1)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data, gain=1)

            else: pass

    
    def _set_datasets(self, train_set_ind_list=None, valid_set_ind_list=None, test_set_ind_list=None):
        """
        Initialize and partition dataset into train_set, valid_set and test_set. 
        """
        
        dataset_size = len(self.dataset)

        # Partition strategy: [ test | valid | train ]
        dataset_indices_total = list(range(dataset_size))
        np.random.shuffle(dataset_indices_total)
        split_pt_test = int(np.floor(self.test_ratio*dataset_size))
        split_pt_valid = int(np.floor((self.test_ratio+self.valid_ratio)*dataset_size))

        if train_set_ind_list is not None: self._train_set_indices = copy.deepcopy(train_set_ind_list)
        else: self._train_set_indices = dataset_indices_total[split_pt_valid:]

        if valid_set_ind_list is not None: self._valid_set_indices = copy.deepcopy(valid_set_ind_list)
        else: self._valid_set_indices = dataset_indices_total[split_pt_test:split_pt_valid]

        if test_set_ind_list is not None: self._test_set_indices = copy.deepcopy(test_set_ind_list)
        else: self._test_set_indices = dataset_indices_total[:split_pt_test]


    def init_dataLoaders(self, train_set_ind_list=None, valid_set_ind_list=None, test_set_ind_list=None):
        """
        """

        self._set_datasets(train_set_ind_list, valid_set_ind_list, test_set_ind_list) # Partition the dataset. 

        train_set_sampler = torch.utils.data.SubsetRandomSampler(self._train_set_indices)
        valid_set_sampler = torch.utils.data.SubsetRandomSampler(self._valid_set_indices)
        test_set_sampler = torch.utils.data.SubsetRandomSampler(self._test_set_indices)

        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_set_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=valid_set_sampler)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, sampler=test_set_sampler)


    def train(self):
        """
        """

        if not os.path.isdir(self.model_arxiv_dir): os.mkdir(self.model_arxiv_dir)
        clr_dir(self.model_arxiv_dir) # Clear the directory of pre-saved trained models before starting a new batch of training. 

        # Define criterion and optimizer
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), self.learning_rate, 
                                     weight_decay=ML_VAE.LAMBDA_REGLR)
        
        # Iterative training and validation
        print("##############################")
        print("Starting training....")
        self._training_log.append("({}) Starting training....".format(str(datetime.today())))
        self._training_log.append("--------------------")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            # ---------- Apply learning rate decaying schedule ---------- 
            if epoch != 0 and epoch % ML_VAE.LEARNING_RATE_SCHEDULE_PERIOD == 0: 
                for p in optimizer.param_groups: p['lr'] *= ML_VAE.LEARNING_RATE_DECAY_FACTOR
            self.lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            
            # ---------- Training ----------
            loss_train_perEpoch = 0
            for iter, batch in enumerate(self.train_loader):
                # Forward. 
                inputs_train, groundtruths_train = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
                batch_size_temp = groundtruths_train.size(0) # Change to 1 if not intended for getting batch-averaged loss. 

                inputs_train = inputs_train.to(self._device)
                groundtruths_train = groundtruths_train.to(self._device)
                
                self.autoencoder.train()
                y, _ = self.autoencoder(inputs_train)
                loss_train_perBatch_perEpoch = self._loss_func(y, groundtruths_train) # Total loss of one batch. 
                
                # Back propagation. 
                optimizer.zero_grad()
                loss_train_perBatch_perEpoch.backward()
                optimizer.step()

                loss_train_perEpoch_perSample = copy.deepcopy(float(loss_train_perBatch_perEpoch.item()/batch_size_temp))
                self.batch_loss_list_train.append(loss_train_perEpoch_perSample) # Average loss on batch size to obtain loss per sample. 
                loss_train_perEpoch += loss_train_perEpoch_perSample # Adding up averaged loss per sample. 

                # Check training loss every batch. 
                if iter == 0 or (iter+1) % 5 == 0 or iter + 1 == len(self.train_loader):
                    batch_line_temp = "Epoch: [{}/{}]\t| Batch: [{}/{}]\t| Loss: {:.4f}\t| Time: {:.4f} s". \
                                       format(epoch+1, self.num_epochs, iter+1, len(self.train_loader), 
                                              loss_train_perEpoch_perSample, time.time()-start_time)
                    self._training_log.append(batch_line_temp)
                    print(batch_line_temp)
            
            loss_train_perEpoch /= len(self.train_loader)
            self.epoch_loss_list_train.append(loss_train_perEpoch) # Average loss on batch number to obtain loss per epoch per batch. 
            
            # ---------- Validation ----------
            loss_valid_perEpoch = 0
            for _, batch in enumerate(self.valid_loader):
                inputs_valid, groundtruths_valid = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
                batch_size_temp = groundtruths_valid.size(0) # Change to 1 if not intended for getting batch-averaged loss. 

                inputs_valid = inputs_valid.to(self._device)
                groundtruths_valid = groundtruths_valid.to(self._device)
                
                self.autoencoder.eval()
                with torch.no_grad():
                    y, _ = self.autoencoder(inputs_valid)
                    loss_valid_perBatch_perEpoch = self._loss_func(y, groundtruths_valid)

                    loss_valid_perEpoch_perSample = copy.deepcopy(float(loss_valid_perBatch_perEpoch.item()/batch_size_temp))
                    self.batch_loss_list_valid.append(loss_valid_perEpoch_perSample) # Average loss on batch size to obtain loss per sample. 
                    loss_valid_perEpoch += loss_valid_perEpoch_perSample # Adding up averaged loss per sample. 
            
            loss_valid_perEpoch /= len(self.valid_loader)
            self.epoch_loss_list_valid.append(loss_valid_perEpoch) # Average loss on batch number to obtain loss per epoch per batch. 

            # ---------- Wrap-up this epoch ----------
            epoch_line_temp = "Epoch Number: {}\t| Train Loss: {:.4f}\t| Valid Loss: {:.4f}\t| Time Elapsed: {:.4f} s". \
                               format(int(epoch+1), loss_train_perEpoch, loss_valid_perEpoch, time.time()-start_time)
            self._training_log.append(epoch_line_temp)
            self._training_log.append("--------------------")
            print(epoch_line_temp)
            print("--------------------")

            # Save trained intermediate models every certain epochs. 
            if (epoch+1) % ML_VAE.MODEL_CHECKPOINT_EPOCH_NUM == 0:
                # model_savePath_temp = os.path.join(self.model_arxiv_dir, "model_epoch_{}.pkl".format(epoch+1))
                # torch.save(self.autoencoder.state_dict(), model_savePath_temp)
                model_savePath_temp = os.path.join(self.model_arxiv_dir, "model_epoch_{}.pth".format(epoch+1))
                torch.save(self.autoencoder, model_savePath_temp)
            
            torch.cuda.empty_cache() # Clear unnecessary memory allocations. 
        
        # ---------- Summary & Save ----------
        self._training_time_total = time.time() - start_time
        print("Training completed. ")
        print("##############################")
        # torch.save(self.autoencoder.state_dict(), os.path.join(self.model_arxiv_dir, "model_final.pkl")) # Save the final trained model.
        torch.save(self.autoencoder, os.path.join(self.model_arxiv_dir, "model_final.pth")) # Save the final trained model.
        self._training_log.append("({}) Training completed. ".format(str(datetime.today())))
        self._save_training_log() # Save a training log. 


    def evaluate(self, model, test_loader):
        """
        """

        loss_list, groundtruths_list, generations_list, latent_list = [], [], [], []
        folder_name_list, infolder_id_list = [], []
        
        for _, batch in enumerate(test_loader): 
            inputs_test, groundtruths_test = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
            batch_size_temp, c, h, w = groundtruths_test.size()
            
            inputs_test = inputs_test.to(self._device)
            groundtruths_test = groundtruths_test.to(self._device)
            
            model.eval()
            with torch.no_grad():
                y, latent = model(inputs_test)
                loss_test_perBatch = self._loss_func(y, groundtruths_test)

                loss_list.append(float(loss_test_perBatch.item()/batch_size_temp)) # Loss per test sample.
                if len(groundtruths_list) <= 50: # Prescribe size limit to save space. 
                    groundtruths_list += [groundtruths_test.cpu().numpy()[i,:].reshape(c, h, w) for i in range(batch_size_temp)]
                if len(generations_list) <= 50: # Prescribe size limit to save space. 
                    generations_list += [y.cpu().numpy()[i,:].reshape(c, h, w) for i in range(batch_size_temp)]
                latent_list += [latent.cpu().numpy()[i,:] for i in range(batch_size_temp)]
                folder_name_list += [batch['folder_name'][i] for i in range(batch_size_temp)]
                infolder_id_list += [batch['file_No'][i] for i in range(batch_size_temp)]
        
        return loss_list, groundtruths_list, generations_list, latent_list, folder_name_list, infolder_id_list

    
    def loss_plot(self):
        """
        """

        epochs_ind_array = np.arange(self.num_epochs, step=1)
        train_batch_ind_array = np.arange(self.num_epochs, step=1./len(self.train_loader))
        valid_batch_ind_array = np.arange(self.num_epochs, step=1./len(self.valid_loader))

        epoch_loss_train_array = np.log(np.array(self.epoch_loss_list_train).astype(float).reshape(-1))
        epoch_loss_valid_array = np.log(np.array(self.epoch_loss_list_valid).astype(float).reshape(-1))
        batch_loss_train_array = np.log(np.array(self.batch_loss_list_train).astype(float).reshape(-1))
        batch_loss_valid_array = np.log(np.array(self.batch_loss_list_valid).astype(float).reshape(-1))

        plt.figure(figsize=(20,20))
        plt.rcParams.update({"font.size": 35})
        plt.tick_params(labelsize=35)
        line1, = plt.plot(train_batch_ind_array, batch_loss_train_array, 
                          color='blue', linewidth=10.0, alpha=0.3, label="Train Loss (log) - Batch")
        line2, = plt.plot(valid_batch_ind_array, batch_loss_valid_array, 
                          color='orange', linewidth=10.0, alpha=0.3, label="Valid Loss (log) - Batch")
        line3, = plt.plot(epochs_ind_array, epoch_loss_train_array, 
                          color='blue', linewidth=3.0, label="Train Loss (log) - Epoch")
        line4, = plt.plot(epochs_ind_array, epoch_loss_valid_array, 
                          color='orange', linewidth=3.0, label="Valid Loss (log) - Epoch")
        plt.xlabel("Epochs", fontsize=40)
        plt.ylabel("log(Loss)", fontsize=40)
        plt.legend([line1,line2,line3,line4], ["Train Loss (log) - Batch", 
                                               "Valid Loss (log) - Batch",
                                               "Train Loss (log) - Epoch", 
                                               "Valid Loss (log) - Epoch",], prop={"size": 40})
        plt.title("Train & Valid Loss v/s Epochs")
        plt.savefig("Train_Valid_Loss_VAE.png")


    @property
    def training_time(self):
        """
        """

        return self._training_time_total


    def showcase_generation(self, ind=None):
        """
        """

        pass


    def sample_from_latent(self):
        """
        """

        pass


    def generate(self, latent):
        """
        """

        pass


    def _save_training_log(self):
        """
        Save framework setup and training info. as a '.log' file. 
        """

        heading = ["Date and Time: {}".format(str(datetime.today())), 
                   "##############################",
                   "Batch size: {}".format(self.batch_size),
                   "Learning rate: {}".format(self.learning_rate),
                   "Epoch number: {}".format(self.num_epochs),
                   "Reconstruction loss mode: {}".format(ML_VAE.LOSS_RECONSTRUCT_MODE),  
                   "Dataset size: {}".format(len(self.dataset)),
                   "Train/Valid/Test: {:.2f}/{:.2f}/{:.2f}".format(1.-self.test_ratio-self.valid_ratio, 
                                                                   self.valid_ratio, self.test_ratio), 
                   "Latent dimension: {}".format(int(self.autoencoder.latent_dim)), 
                   "##############################"
                   ]

        self._training_log.append("##############################"),
        self._training_log.append("Total training time: {:.4f}. ".format(self._training_time_total))

        content = copy.deepcopy(heading + self._training_log)

        with open(self.log_path, 'w') as f:
            content = '\n'.join(content)
            f.write(content)
            f.close()


if __name__ == "__main__":
    pass
