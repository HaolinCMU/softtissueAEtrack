## Overview
This repository contains code to predict soft tissue deformations using by tracking the displacement of several fiducial markers. This is accomplished by first simulating various deformations of the surgical region of interest in an offline setting with finite element methods. Next, we train and autoencoder to reduce the dimensionality of the overall deformation field, and a feed forward neural network to map from the fiducial marker diaplacements to the low dimensional latent space. Finally the latent space is decoded to obtain the models prediction of the entire soft tissue deformation.

## Code
There are three examples provided: head and neck tumor, kidney, and aorta. Within each models' folder, there are .mat files containing the reference model, as well as all of the ground truth deformations. *More details on the .py and .m files in the respective folders*

## Data availability
The head and neck tumor model was generously provided by the Photodynamic Therapy Center, Department of Cell Stress Biology, Roswell Park Comprehensive Cancer Center. That said, we cannot redistribute this model and it has therefore been omitted from this repository. The kidney and aorta models are publicly available from the NIH 3D Exchange and can be accessed here: [kidney](https://3d.nih.gov/entries/3DPX-000906), [aorta](https://3d.nih.gov/entries/3DPX-003283). Note: the kidney model was significantly post-processed to obtain a mesh of a single kidney.
