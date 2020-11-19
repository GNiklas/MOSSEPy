#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional routines needed in MOSSE tracking.

Created on Thu Nov 19 11:39:05 2020

@author: niklas
"""


import os
import numpy as np
import scipy.linalg as la
import scipy.ndimage as nd


def genFilenames(objName, i):
    """
    generate names of in- and output files

    Parameters
    ----------
    objName : string
        name of the object in image.
    i : int
        iterator.

    Returns
    -------
    imgFile : string
        input image file name.
    temFile : string
        template file name.
    resFile : string
        response file name.
    filFile : string
        filter file name.

    """
    path = os.getcwd()
    inPath = path + "/data/"
    outPath = path + "/results/"
    
    if (i < 10):
        imgFile = inPath + objName + '000' + str(i) + '_noise.png'
        temFile = outPath + objName + '000' + str(i) + '_template.png'         
        resFile = outPath + objName + '000' + str(i) + '_response.png'
        filFile = outPath + objName + '000' + str(i) + '_filter.png'
    elif (i < 100):
        imgFile = inPath + objName + '00' + str(i) + '_noise.png'
        temFile = outPath + objName + '00' + str(i) + '_template.png' 
        resFile = outPath + objName + '00' + str(i) + '_response.png'
        filFile = outPath + objName + '00' + str(i) + '_filter.png'
    elif (i < 1000):
        imgFile = inPath + objName + '0' + str(i) + '_noise.png'
        temFile = outPath + objName + '0' + str(i) + '_template.png'
        resFile = outPath + objName + '0' + str(i) + '_response.png'
        filFile = outPath + objName + '0' + str(i) + '_filter.png'
    else:
        imgFile = inPath + objName + str(i) + '_noise.png'
        temFile = outPath + objName + str(i) + '_template.png' 
        resFile = outPath + objName + str(i) + '_response.png'
        filFile = outPath + objName + str(i) + '_filter.png'    
        
    return imgFile, temFile, resFile, filFile

    
def gauss2D(valRange, size, mu, sigma):
    """
    calculate 2D Gaussian on array.

    Parameters
    ----------
    valRange : int
        value range to which the Gaussian will be scaled.
    size : numpy array
        x and y size of array.
    mu : numpy array
        x and y center position of Gaussian.
    sigma : numpy array
        x and y standard deviation of Gaussian.

    Returns
    -------
    g : numpy array
        2D Gaussian array.

    """
    x = np.linspace(0, size[0]-1, size[0])
    y = np.linspace(0, size[0]-1, size[0])
    X, Y = np.meshgrid(x, y)
    
    xExponent = -(X - mu[0])**2 / (2.0 * sigma[0]**2)
    yExponent = -(Y - mu[1])**2 / (2.0 * sigma[1]**2)
    
    g = (valRange - 1) * np.exp(xExponent + yExponent)

    return g

    
def randWarp(Iin, Isize):
    """
    randomly warp image

    Parameters
    ----------
    Iin : numpy array
        input image.
    Isize : numpy array
        x and y size of image.

    Returns
    -------
    Iout : numpy array
        output image.

    """
    # rotation angle
    angMax = 10.0
    angDeg = np.random.uniform(-angMax, angMax)
    angRad = np.radians(angDeg)
    c = np.cos(angRad)
    s = np.sin(angRad)
    # scaling factor
    scaleMin = 0.9
    scaleMax  = 1.1
    scale = np.random.uniform(scaleMin, scaleMax)
    # translation vector
    tMax = [int(Isize[0]/40), int(Isize[1]/40)]
    t = [np.random.uniform(-tMax[0], tMax[0]), np.random.uniform(-tMax[1], tMax[1])]
    
    # rotation and scaling matrix
    R = np.zeros((2, 2))
    R[0, 0] = c
    R[0, 1] = s
    R[1, 0] = -s
    R[1, 1] = c
    R  = scale * R

    # rotation axis in center of image
    p = np.array([Isize[0]/2, Isize[1]/2])
    p = p.reshape(2,1)
    # get offset for transform
    o = p - np.dot(la.inv(R), p)
    o = o.flatten()
    
    # apply random transform
    Iin = np.flipud(Iin)
    Iout = nd.affine_transform(Iin, la.inv(R), o, mode='nearest')
    
    # apply random translation
    Iout = nd.affine_transform(Iout, np.eye(2), t, mode='nearest')
    Iout = np.flipud(Iout)
            
    return Iout        


def preProcess(Iin, Isize, eps):
    """
    pre-process image according to MOSSE pre-processing steps

    Parameters
    ----------
    Iin : numpy array
        input image.
    Isize : numpy array
        x and y size of image.
    eps : float
        regularization parameter.

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
    win0 = np.hanning(Isize[0])
    win1 = np.hanning(Isize[1])
    win2D = np.sqrt(np.outer(win0, win1))
    Iout = Iout * win2D
    
    return Iout