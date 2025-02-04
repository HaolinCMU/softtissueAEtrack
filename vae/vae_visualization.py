# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:19:56 2022

@author: hlinl
"""


import copy
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import shutil

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from files import *


result_directory = "E:/research/soft_tissue_tracking/code/ANN/nonlinear/head_and_neck/result/final/3"
is_VAE = True # True if and only if model is VAE. 

encoder_input_size = 3465
latent_dim = 27

plot_directory = os.path.join(result_directory, "plots")
if not os.path.isdir(plot_directory): os.mkdir(plot_directory)

latent_list_all = np.load(os.path.join(result_directory, "latent_list.npy")).reshape(-1,latent_dim) # Float array. sample_num * latent_dim. 

if is_VAE:
    mu_list_all = np.load(os.path.join(result_directory, "mu_list.npy")).reshape(-1,latent_dim) # Float array. sample_num * latent_dim. 
    logvar_list_all = np.load(os.path.join(result_directory, "logvar_list.npy")).reshape(-1,latent_dim) # Float array. sample_num * latent_dim. 

generations_list_all = np.load(os.path.join(result_directory, "generations_list.npy")).reshape(-1,encoder_input_size) # Float array. sample_num * latent_dim. 
groundtruths_list_all = np.load(os.path.join(result_directory, "groundtruths_list.npy")).reshape(-1,encoder_input_size) # Float array. sample_num * latent_dim. 

latent_all_tsne_embedded = TSNE(n_components=2).fit_transform(latent_list_all)
# latent_all_tsne_embedded = latent_list_all

plt.figure()
plt.rcParams.update({"font.size":35})
plt.tick_params(labelsize=10)
plt.scatter(latent_all_tsne_embedded[:,0], latent_all_tsne_embedded[:,1], s=4.0)
plt.axis('equal')
plt.savefig(os.path.join(plot_directory, "latent_all_tsne_2D.png"))


kmeans_clusters = 7
kmeans = KMeans(n_clusters=kmeans_clusters, random_state=0).fit(latent_list_all)

fig, ax = plt.subplots(figsize=(20, 12.8))
plt.rcParams.update({"font.size":35})
plt.tick_params(labelsize=25)
scatter = ax.scatter(latent_all_tsne_embedded[:,0], latent_all_tsne_embedded[:,1], 
                     cmap='rainbow', c=kmeans.labels_, s=30.0)
legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend)
plt.axis('equal')
plt.savefig(os.path.join(plot_directory, 
                         "latent_all_tsne_2D_kmeans_{}.png".format(kmeans_clusters)))

image_clusters_directory = os.path.join(result_directory, 'image_clusters')
if not os.path.isdir(image_clusters_directory): os.mkdir(image_clusters_directory)


# PCA encoder. 

# Box plot of extracted latent feature components. 
latent_list_all
fig, axes = plt.subplots(1,1,figsize = (20, 12.8))
plt.rcParams.update({"font.size":35})
plt.tick_params(labelsize=25)
df = pd.DataFrame(latent_list_all, # The columns of latent embeddings.
                  columns=[str(i) for i in range(latent_list_all.shape[1])])
f = df.boxplot(sym = 'o',            
           vert = True,
           whis=1.5, 
           patch_artist = True,
           meanline = False,showmeans = True,
           showbox = True,
           showfliers = True,
           notch = False,
           return_type='dict')
plt.xlabel('Autoencoder latent components', fontsize=40)
plt.ylabel('Euclidean distance (mm)', fontsize=40)
plt.savefig(os.path.join(plot_directory, "boxPlot_latent_features.png"))


