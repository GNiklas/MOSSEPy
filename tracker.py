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


import os
import numpy as np

import utils
import image as img
import visualization as vis


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
        # correlate old filter with new template
        self.cropTemplate()
        self.calFilterResponse()
        
        # maximum position in g
        gPos = np.unravel_index(np.argmax(self.g, axis=None), self.g.shape)
        
        # maximum position in full image from position of g
        # (old object position) and size of g
        self.objPos = [gPos[0] + self.objPos[0]- int(self.tempSize[0]/2), 
                      gPos[1] + self.objPos[1]- int(self.tempSize[1]/2)]
        
    def initObjPos(self, objPos):
        """
        initialize object position.

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
            
        print('---------------------')
        print('iteration: ', self.i)
        print('objPos: ', self.objPos)
        
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
        
        # define output filenames
        name = os.path.basename(self.imgFile.path)
        temFile = self.outDir + '/' + name[:-4] + '_tem.png'
        filFile = self.outDir + '/' + name[:-4] + '_fil.png'
        resFile = self.outDir + '/' + name[:-4] + '_res.png'
                
        # save results to files
        img.write(temFile, f)
        img.write(filFile, abs(np.fft.ifftshift(h)))        
        img.write(resFile, abs(g))


#%%
class AdpCorrelation(Correlation):
    """
    Class of adaptive correlation trackers. Inherits
    from basic correlation tracker class.
    
    Filter initialization and update have to be done in inherited class.
    """
    
    def __init__(self, 
                 relInDir, 
                 relOutDir, 
                 valRange, 
                 tempSize, 
                 sigma, 
                 eps,
                 trainSteps,
                 rate):
        """
        constructor of adaptive correlation tracker class

        Parameters
        ----------
        relInDir : string
            relative input directory
        relOutDir : string.
            relative output directory
        valRange : int
            image value range
        tempSize : list of ints
            vertical and horizontal size of template to be cropped
        sigma : list of floats
            standard deviations of optimal filter response
        eps : float
            regularization parameter
        trainSteps : int
            number of initial training steps
        rate : float
            filter learning rate used in running average.

        Returns
        -------
        None.

        """
        
        # call inherited constructor
        Correlation.__init__(self, 
                             relInDir, 
                             relOutDir, 
                             valRange, 
                             tempSize, 
                             sigma, 
                             eps)
        
        # number of training steps
        self.trainSteps = trainSteps
        # learning rate
        self.rate = rate
        
    def trackImg(self):
        """
        Track object over all images in given directory.

        Returns
        -------
        None.

        """
        self.i = 0
        
        # iterate over image files in input directory
        for self.imgFile in os.scandir(self.inDir):
            if self.imgFile.path.endswith('.png'):
                self.i += 1
                
                self.I = img.read(self.imgFile.path)
                
                if self.i == 1:
                    # this should be done somewhere else !!!
                    self.objPos = [256, 256]
                                
                    self.initFilter()
                    
                    self.saveResults()                 
                    self.showResults()
                    
                else:
                    self.calObjPos()
                    
                    self.saveResults()                 
                    self.showResults()

                    self.updateFilter()
                    
                    
#%%
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
        # and optimal response to it
        self.cropTemplate()
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