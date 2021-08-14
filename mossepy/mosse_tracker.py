#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 12:00:13 2021

@author: niklas
"""


import numpy as np

import mossepy.utils as utils
from mossepy.adaptive_correlation_tracker import AdpCorrelation


class MOSSE(AdpCorrelation):
    """
    MOSSE tracker class. Inherits from adaptive correlation
    tracker class.
    """
    
    def __init__(self, 
                 relInDir='/data', 
                 relOutDir='/results', 
                 valRange=256, 
                 tempSize=[128, 128], 
                 sigma=[2., 2.], 
                 eps=0.1,
                 trainSteps=256,
                 rate=0.125):
        """
        constructor of MOSSE tracker class

        Parameters
        ----------
        relInDir : string. optional.
            relative input directory. default is '/data'.
        relOutDir : string. optional.
            relative output directory. default is '/results'.
        valRange : int. optional.
            image value range. default is 256.
        tempSize : list of ints. optional.
            vertical and horizontal size of template to be cropped.
            default is [128, 128]
        sigma : list of floats. optional.
            standard deviations of optimal filter response.
            default is [2., 2.].
        eps : float. optional.
            regularization parameter. default is 0.1.
        trainSteps : int. optional.
            number of initial training steps. default is 256.
        rate : float. optional.
            filter learning rate used in running average.
            default is 0.125.
            
        Returns
        -------
        None.

        """
        # call inherited constructor
        AdpCorrelation.__init__(self, 
                                relInDir, 
                                relOutDir, 
                                valRange, 
                                tempSize, 
                                sigma, 
                                eps,
                                trainSteps,
                                rate)
        
    def initFilter(self):
        """
        initialize MOSSE filter by training on multiple perturbations of
        initial template

        Returns
        -------
        None.

        """
        # get initial template (ground truth)
        self.cropTemplate()
        # and optimal response to it        
        self.calOptimalResponse()
        
        # template is varied with affine transformations here
        # to get a training set. Train filter.
        for i in range(0, self.trainSteps):
            fi = utils.randWarp(self.f, self.tempSize)
            fi = utils.preProcess(fi, self.tempSize, self.eps)
            Fi = np.fft.fft2(fi)
            conjFi = np.conj(Fi)
            G = np.fft.fft2(self.g)
            
            if (i == 0):
                self.A = G * conjFi
                self.B = Fi * conjFi
                
            else:
                self.A += G * conjFi
                self.B += Fi * conjFi + self.eps
                
        conjH = self.A / self.B
        H = np.conj(conjH)
        self.h = np.fft.ifft2(H)
        
    def updateFilter(self):
        """
        update MOSSE filter using a running average on the previous filter

        Returns
        -------
        None.

        """
        # get template centered in new object position
        self.cropTemplate()
        # set it as new ground truth        
        self.calOptimalResponse()
        
        fi = utils.preProcess(self.f, self.tempSize, self.eps)
        Fi = np.fft.fft2(fi)
        conjFi = np.conj(Fi)
        G = np.fft.fft2(self.g)
        
        # use running average
        self.A = (1. - self.rate) * self.A + self.rate * (G * conjFi)
        self.B = (1. - self.rate) * self.B + + self.rate * (Fi * conjFi + self.eps)
        
        conjH =  self.A / self.B
        H = np.conj(conjH)
        self.h = np.fft.ifft2(H)