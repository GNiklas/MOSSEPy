#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional routines needed in MOSSE tracking.

Created on Thu Nov 19 11:39:05 2020

@author: niklas
"""


import numpy as np
import scipy.linalg as la
import scipy.ndimage as nd

    
def gauss2D(valRange, size, mu, sigma):
    """
    calculate 2D Gaussian on array.

    Parameters
    ----------
    valRange : int
        value range to which the Gaussian will be scaled.
    size : list of ints
        x and y size of array.
    mu : list of floats
        x and y center position of Gaussian.
    sigma : list of floats
        x and y standard deviation of 2D Gaussian.

    Returns
    -------
    g : numpy array
        2D Gaussian array.

    """
    # create grid
    x = np.linspace(0, size[0]-1, size[0])
    y = np.linspace(0, size[0]-1, size[0])
    X, Y = np.meshgrid(x, y)
    
    # exponents for 1D Gaussians
    xExponent = -(X - mu[0])**2 / (2.0 * sigma[0]**2)
    yExponent = -(Y - mu[1])**2 / (2.0 * sigma[1]**2)
    
    # 2D Gaussian is product of two 1D Gaussians
    g = (valRange - 1) * np.exp(xExponent + yExponent)

    return g

    
def randWarp(Iin, size, angMax = 10.0, scaleExt = [0.9, 1.1], tRel = 40):
    """
    randomly warp image

    Parameters
    ----------
    Iin : numpy array
        input image.
    size : list of ints
        x and y size of image.
    angMax : float. optional.
        max rotation angle in deg. default is 10.
    scaleExt : list. optional.
        min and max scaling factor. default is [0.9, 1.1].
    tRel : int. optional.
        max relative translation. default is 40.
        
    Returns
    -------
    Iout : numpy array
        output image.

    """
    # rotation angle
    angDeg = np.random.uniform(-angMax, angMax)
    angRad = np.radians(angDeg)
    c = np.cos(angRad)
    s = np.sin(angRad)
    
    # scaling factor
    scale = np.random.uniform(scaleExt[0], scaleExt[1])
    
    # translation vector
    tMax = [int(size[0]/tRel), int(size[1]/tRel)]
    t = [np.random.uniform(-tMax[0], tMax[0]), np.random.uniform(-tMax[1], tMax[1])]
    
    # rotation and scaling matrix
    R = np.zeros((2, 2))
    R[0, 0] = c
    R[0, 1] = s
    R[1, 0] = -s
    R[1, 1] = c
    R  = scale * R

    # rotation axis in center of image
    p = np.array([size[0]/2, size[1]/2])
    p = p.reshape(2,1)
    # get offset for transform
    o = p - np.dot(la.inv(R), p)
    o = o.flatten()
    
    # apply random rotation and scaling
    Iout = nd.affine_transform(Iin, la.inv(R), o, mode='nearest')
    
    # apply random translation
    Iout = nd.affine_transform(Iout, np.eye(2), t, mode='nearest')
            
    return Iout        


def preProcess(Iin, size, eps=0.1):
    """
    pre-process image according to MOSSE pre-processing steps

    Parameters
    ----------
    Iin : numpy array
        input image.
    size : list of ints
        x and y size of image.
    eps : float. optional.
        regularization parameter. default is 0.1.

    Returns
    -------
    Iout : numpy array
        output image.

    """
    # log transform
    Iout = np.log(Iin + 1.)
    
    # normalize
    Iout = (Iout - np.mean(Iout)) / (np.std(Iout) + eps)
    
    # multiply with 2D Hanning window to reduce edge effects
    win0 = np.hanning(size[0])
    win1 = np.hanning(size[1])
    win2D = np.sqrt(np.outer(win0, win1))
    Iout = Iout * win2D
    
    return Iout