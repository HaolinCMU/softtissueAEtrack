# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:51:16 2020

@author: haolinl
"""


import copy
import os

import numpy as np
import scipy.io


def forceFieldInterpolation(force_field_1, force_field_2, alpha):
    """
    Level set linear interpolation of two force fields. 
    Two force fields must have the same matrix dimension. 

    Formulation:
    ----------
        force_field_new = alpha * force_field_1 + (1 - alpha) * force_field_2.

    Parameters:
    ----------
        force_field_1: 2D Array of floats. 
            The matrix of the first force field.
            Size: nSurfI*3 x sampleNum.    
        force_field_2: 2D Array of floats. 
            The matrix of the second force field.
            Size: nSurfI*3 x sampleNum. 
        alpha: List of float. 
            The coefficents determining the weight of linear interpolation.  
    
    Returns:
    ----------
        force_field_new: 2D Array of floats. 
            The newly interpolated force field. 
            Size: nSurfI*3 x sampleNum.
        alpha_indexing_vector: 1D Array of floats. 
            Indicating the alpha value the force field corresponds to. 
            Size: sampleNum x 1. 
    """

    sample_num = force_field_1.shape[1]
    force_field_new, alpha_indexing_vector = np.zeros(force_field_1.shape), np.zeros(shape=(sample_num, 1))

    for i in range(sample_num):
        force_field_new[:,i] = force_field_1[:,i] * alpha[i%len(alpha)] + force_field_2[:,i] * (1.0 - alpha[i%len(alpha)])
        alpha_indexing_vector[i,0] = alpha[i%len(alpha)] # The weight(portion) of Laplacian force field. 

    return force_field_new, alpha_indexing_vector


def generateForceFields(laplacian_matrix, sample_num, weight, eigen_num=20, scalar=10.0):
    """
    Generate random force fields based on the prescribed laplacian matrix. 
    Geometry-dependent. 

    Parameters:
    ----------
        laplacian_matrix: 2D Array of floats.
            Divergence of gradient (curvature) of the outer surface of the geometry. Intrinsic. 
            Size: nSurfI x nSurfI. 
        sample_Num:Int. 
            The number of samples/force fields to be generated. 
        weight: 2D Array of floats. 
            The randomly generated matrix for force fields' reconstruction. 
            Size: eigen_num x (3*sample_num)
        eigen_num (optional): Int. 
            The number of eigenvalues & eigenvectors to be used for force field reconstruction. 
            Feasible range: [1, nSurfI]. 
            Default: 20. 
        scalar (optional): Float. 
            The scaling factor of the force field. 
            Default: 10.0.
    
    Returns:
    ----------
        force_field_matrix: 2D Array of floats. 
            The generated force fields. 
            Size: nSurfI*3 x sampleNum. 
    """

    eigVal_full, eigVect_full = np.linalg.eig(laplacian_matrix)

    eigVal, eigVect = np.zeros(shape=(eigen_num, 1), dtype=complex), np.zeros(shape=(eigVect_full.shape[0], eigen_num), dtype=complex)
    eigVal_sorted_indices = np.argsort(np.real(eigVal_full))
    eigVal_PC_indices = eigVal_sorted_indices[-1:-(eigen_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
    
    for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
        eigVal[i,0] = eigVal_full[index] # Pick PC_num principal eigenvalues. Sorted. 
        eigVect[:,i] = eigVect_full[:,index] # Pick PC_num principal eigenvectors. Sorted. 

    # weight = (2.0 * np.random.rand(eigen_num, 3*sample_num) - 1.0) * scalar
    raw_random_matrix = np.real(eigVect @ weight) * scalar # Axis 1: concatenated as xyzxyz...; Size: nSurfI x (sample_num*3). 
    force_field_matrix = np.zeros(shape=(laplacian_matrix.shape[0]*3, sample_num))

    for i in range(sample_num): force_field_matrix[:,i] = raw_random_matrix[:,3*i:3*(i+1)].reshape(-1,)

    return force_field_matrix


def main():
    """
    Implement force interpolation and save interpolated force field matrix to .mat file. 

    Pipeline:
    ----------
        1. Run "nonlinearCasesCreation.py" with 'isPrescribedForceOn = False' firstly. 
        2. Run "forceInterpolation.py" in the same directory. 
        3. Set 'isPrescribedForceOn = True', set 'force_type = "interpolated", then run "nonlinearCasesCreation.py" again. 
            Get input files with "*_interpolated.inp" in the folder 'force_interpolation_folder'. 
        4. Set 'isPrescribedForceOn = True', set 'force_type = "random", then run "nonlinearCasesCreation.py" again. 
            Get input files with "*_random.inp" in the folder 'force_interpolation_folder'. 
    """

    # Result from nonlinear dataset (force field).
    force_file_name = "training_parameters_transfer.mat"
    force_field_matrix_1 = scipy.io.loadmat(force_file_name)["force_field_matrix"] # Laplacian-smoothed force field. Size: nSurfI*3 x sampleNum. 
    sample_num = force_field_matrix_1.shape[1] # Number of samples/force fields. 

    # Result from linear dataset (laplacian).
    laplacian_file_name = "data_head_and_neck.mat"
    laplacian_matrix = scipy.io.loadmat(laplacian_file_name)["laplacianMatrixI"] # Size: nSurfI x nSurfI. 
    eigenNum, force_scalar = scipy.io.loadmat(force_file_name)["eigen_number_force"][0,0], scipy.io.loadmat(force_file_name)["force_scalar_coeff"][0,0] # Float. The scalar of force fields controlling the force magnitude -> deformation magnitude of the tumor in nonlinear solver. Unit of force field scalar: N. 

    weight_matrix = scipy.io.loadmat(force_file_name)["weight_matrix"] # Fix weight matrix. Random: weight_matrix = (2.0 * np.random.rand(eigenNum, 3*sample_num) - 1.0) * force_scalar
    force_field_matrix_2 = generateForceFields(laplacian_matrix, sample_num, weight_matrix, eigen_num=eigenNum, scalar=force_scalar) # Reconstructed force field. Size: nSurfI*3 x sampleNum.

    alpha_increment = 0.1 # The increment of alpha (between 0.0 and 1.0). 
    alpha = list(np.linspace(0.0, 1.0, int(1.0/alpha_increment)+1)) # List of floats. [0.0, 1.0]. Closed. 
    force_field_new, alpha_indexing_vector = forceFieldInterpolation(force_field_matrix_1, force_field_matrix_2, alpha)

    mdict = {"force_field_interpolated": force_field_new, # The interpolated force fields. 
             "force_field_matrix_1": force_field_matrix_1, # Laplacian-smoothed force field. Size: nSurfI*3 x sampleNum. 
             "force_field_matrix_2": force_field_matrix_2, # Reconstructed force field. Size: nSurfI*3 x sampleNum.
             "force_field_random": generateForceFields(laplacian_matrix, sample_num, 
                                                       weight=(2.0*np.random.rand(eigenNum,3*sample_num)-1.0), 
                                                       eigen_num=eigenNum, scalar=force_scalar), # A new batch of reconstructed force fields. Should be independent of original weight-summed force field matrix. 
             "weight_matrix": weight_matrix,
             "alpha_indexing_vector": alpha_indexing_vector}

    scipy.io.savemat("force_field_data.mat", mdict)
    
    
    
    


if __name__ == "__main__":
    main()
