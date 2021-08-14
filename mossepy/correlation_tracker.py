#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:56:40 2021

@author: niklas
"""


import os
import numpy as np

import mossepy.utils as utils
import mossepy.image as img
import mossepy.visualization as vis


class Correlation(object):
    """
    Basic correlation tracker class.
    
    Tracking routine, Filter initialization 
    and update have to be done in inherited class.
    """
    
    def __init__(self,  
                 relInDir, 
                 relOutDir, 
                 valRange, 
                 tempSize, 
                 sigma, 
                 eps):
        """
        Constructor of correlation tracker class.
        Initalize basic tracker parameters.

        Parameters
        ----------
        relInDir : string
            relative input directory
        relOutDir : string.
            relative output directory
        valRange : int
            image value range.
        tempSize : list of ints
            vertical and horizontal size of template to be cropped.
        sigma : list of floats
            standard deviations of optimal filter response.
        eps : float
            regularization parameter

        Returns
        -------
        None.

        """
        # absolute in- and output directories
        path = os.getcwd()
        self.inDir = path + relInDir
        self.outDir = path + relOutDir
        
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
        
        # convert to grayscale
        self.f = utils.rgb2Gray(self.f)
            
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
        self.g = utils.gauss2D(self.valRange, self.tempSize, optPos, self.sigma)

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
        # crop template from current image
        self.cropTemplate()
        # correlate filter with template
        self.calFilterResponse()
        
        # maximum position in g
        gPos = np.unravel_index(np.argmax(self.g, axis=None), self.g.shape)
        
        # maximum position in full image from position of g
        # (old object position) and size of g
        self.objPos = [gPos[0] + self.objPos[0]- int(self.tempSize[0]/2), 
                      gPos[1] + self.objPos[1]- int(self.tempSize[1]/2)]
        
    def setObjPos(self, objPos):
        """
        set object position.

        Parameters
        ----------
        objPos : list of ints
            current object position.

        Returns
        -------
        None.

        """
        self.objPos = objPos

    def showResults(self):
        """
        Show tracking results in console.

        Returns
        -------
        None.

        """
        # fit images to valRange for output
        f = (self.valRange-1)/self.f.max() * self.f
        g = (self.valRange-1)/self.g.max() * self.g
        h = (self.valRange-1)/self.h.max() * self.h
            
        print('iteration: ', self.i)
        print('objPos: ', self.objPos)
        print('---------------------')
        
        # show heat plots of template, filter and response
        # in common figure
        vis.comp3Heat(f, abs(np.fft.ifftshift(h)), abs(g))
        
    def saveResults(self):
        """
        Save tracking results to files.

        Returns
        -------
        None.

        """
        # fit images to valRange for output
        f = (self.valRange-1)/self.f.max() * self.f
        g = (self.valRange-1)/self.g.max() * self.g
        h = (self.valRange-1)/self.h.max() * self.h
        
        # define output paths
        temPath = self.outDir + '/' + self.imgFile[:-4] + '_tem.jpg'
        filPath = self.outDir + '/' + self.imgFile[:-4] + '_fil.jpg'
        resPath = self.outDir + '/' + self.imgFile[:-4] + '_res.jpg'
                
        # save results to files
        img.write(temPath, f)
        img.write(filPath, abs(np.fft.ifftshift(h)))        
        img.write(resPath, abs(g))