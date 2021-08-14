#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:57:40 2021

@author: niklas
"""


import os

import mossepy.image as img
from mossepy.correlation_tracker import Correlation


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
        for self.imgFile in sorted(os.listdir(self.inDir)):
            if self.imgFile.endswith('.jpg'):
                self.i += 1
                
                imgPath = self.inDir + '/' + self.imgFile
                self.I = img.read(imgPath)
                
                if self.i == 1:  
                    # initialize filter on first object position
                    self.initFilter()
                    
                    self.saveResults()                 
                    self.showResults()
                    
                else:
                    # find object position in new image
                    self.calObjPos()
                    
                    self.saveResults()                 
                    self.showResults()

                    # update filter on new object position
                    self.updateFilter()