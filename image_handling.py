#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert images to numpy arrays and back again

Created on Thu Nov 19 11:29:31 2020

@author: niklas
"""


from PIL import Image
import numpy as np


def read(filename):
    """
    Read image file and convert it to numpy array.

    Parameters
    ----------
    filename : str
        name of image file to be read.

    Returns
    -------
    npFrame : numpy array
        converted image frame.

    """
    
    # check, if filename can be found
    try:
        _PILFrame = Image.open(filename)
        npFrame = np.array(_PILFrame)
        
        print("read image from", filename)
        
    except FileNotFoundError as _fnfError:
        print(_fnfError)

    return npFrame


def write(filename, npFrame):
    """
    Write numpy array to image file.

    Parameters
    ----------
    filename : str
        name of image file to be written to.
    npFrame : numpy array
        image frame to be written from.

    Returns
    -------
    None.

    """
    
    # if necessary, change data type of numpy array to uint8
    if (npFrame.dtype != 'uint8'):
        npFrame = npFrame.astype(dtype='uint8')
        
    _PILFrame = Image.fromarray(npFrame)
    _PILFrame.save(filename)
    
    print("wrote array to", filename)
    
    