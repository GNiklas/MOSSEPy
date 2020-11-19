#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Minimum Output Sum of Squared Error (MOSSE) tracker
after 

Bolme et al. 2010. Visual object tracking using adaptive correlation filters.
Proceedings / CVPR. DOI: 10.1109/CVPR.2010.5539960

Created on Thu Nov 19 10:53:58 2020

@author: niklas
"""

import numpy as np
import routines

class Correlation(object):
    """
    Basic correlation tracker class.
    Filter initialization and update have to be done in inherited class.
    """
    
    def __init__(self, valRange, tempSize, sigma, eps):
        """
        Constructor of Correlation class.
        Initalize basic tracker parameters.

        Parameters
        ----------
        valRange : int
            image value range.
        tempSize : list of ints
            vertical and horizontal size of template to be cropped.
        trainSteps : int
            number of initial training steps.
        rate : float
            filter learning rate used in running average.
        sigma : list of floats
            standard deviations of optimal filter response.
        eps : float
            regularization parameter

        Returns
        -------
        None.

        """
        # range of image values
        self.valRange = valRange
        # size of the template f
        self.tempSize = tempSize
        # standard deviation used for optimal response distribution
        self.sigma = sigma
        # regularization parameter to avoid zero division
        self.eps = eps

    def cropTemplate(self):
        """
        Crop a template with defined size and center from image
    
        Returns
        -------
        None.
    
        """
        dx = int(self.tempSize[0]/2)
        dy = int(self.tempSize[1]/2)
        
        # crop template around object position from image        
        self.f =  self.I[self.objPos[0]-dx:self.objPos[0]+dx,
                         self.objPos[1]-dy:self.objPos[1]+dy]
        
    def calOptimalResponse(self):
        """
        Calculate optimal response for initializing correlation filter.
        Response curve has the form of a 2D Gaussian.

        Returns
        -------
        None.

        """
        # optimal position of target is in center of template window
        optPos = [int(self.tempSize[0]/2), int(self.tempSize[1]/2)]
        
        # optimal response is Gaussian centered in object position
        self.g = routines.gauss2D(self.valRange, self.tempSize, optPos, self.sigma)

    def calFilterResponse(self):
        """
        Calculate response of template to filter.

        Returns
        -------
        None.

        """
        F = np.fft.fft2(self.f)        
        H = np.fft.fft2(self.h)
        conjH = np.conj(H)
        
        # calculate correlation between template and filter
        G = F * conjH
        
        self.g = np.fft.ifft2(G)
        
    def calObjPos(self):
        """
        Calculate object position from maximum in response.

        Returns
        -------
        None.

        """
        # maximum position in g
        gPos = np.unravel_index(np.argmax(self.g, axis=None), self.g.shape)
        
        # maximum position in full image from position of g
        # (old object position) and size of g
        self.objPos = [gPos[0] + self.objPos[0]- int(self.tempSize[0]/2), 
                      gPos[1] + self.objPos[1]- int(self.tempSize[1]/2)]
        
    def setImage(self, I):
        """
        Set current image frame.

        ----------
        I : numpy array
            current image frame.

        Returns
        -------
        None.

        """
        self.I = I
        
    def setObjPos(self, objPos):
        """
        Set current object position.

        Parameters
        ----------
        objPos : list of ints
            current object position.

        Returns
        -------
        None.

        """
        self.objPos = objPos

    def getResults(self):
        """
        Get template, filter and response
        
        Returns
        -------
        f : numpy array
            template
        h : numpy array
            filter mask
        g : numpy array
            response of f to h
        objPos : list of ints
            current object position
        """
        return self.f, self.h, self.g, self.objPos
    
            
#%%
class MOSSE(Correlation):
    """
    MOSSE tracker class. Inherits from basic correlation
    tracker class.
    """
    
    def __init__(self, valRange, tempSize, sigma, eps, trainSteps, rate):
        """
        constructor of MOSSE tracker class

        Parameters
        ----------
        valRange : int
            image value range.
        tempSize : list of ints
            vertical and horizontal size of template to be cropped.
        sigma : list of floats
            standard deviations of optimal filter response.
        eps : float
            regularization parameter
        trainSteps : int
            number of initial training steps.
        rate : float
            filter learning rate used in running average.
            
        Returns
        -------
        None.

        """
        # call inherited constructor
        Correlation.__init__(self, valRange, tempSize, sigma, eps)
        
        # number of training steps
        self.trainSteps = trainSteps
        # learning rate
        self.rate = rate
        
    def initFilter(self):
        """
        initialize MOSSE filter by training on multiple perturbations of
        initial template

        Returns
        -------
        None.

        """
        # template is varied with affine transformations here
        # to get a training set.
        for i in range(0, self.trainSteps):
            fi = routines.randWarp(self.f, self.tempSize)
            fi = routines.preProcess(fi, self.tempSize, self.eps)
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
        fi = routines.preProcess(self.f, self.tempSize, self.eps)
        Fi = np.fft.fft2(fi)
        conjFi = np.conj(Fi)
        G = np.fft.fft2(self.g)
        
        # use running average
        self.A = (1. - self.rate) * self.A + self.rate * (G * conjFi)
        self.B = (1. - self.rate) * self.B + + self.rate * (Fi * conjFi + self.eps)
        
        conjH =  self.A / self.B
        H = np.conj(conjH)
        self.h = np.fft.ifft2(H)
