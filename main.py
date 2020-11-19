#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main for implementation of the MOSSE tracker

Created on Thu Nov 19 11:15:49 2020

@author: niklas
"""


import numpy as np
import tracker
import routines
import image_handling as img
import visualization as vis


def main():
    # data
    objName = "Heli"
    nImg = 2
    valRange = 256
    
    # MOSSE parameters
    tempSize = [128, 128]   # template size
    trainSteps = 256          # no. of training steps for initialization
    rate = 0.125             # learning rate
    sigma = [2.0, 2.0]      # standard deviation of optimal response
    eps = 0.1            # regularization parameter
    
    # initialize tracker
    track = tracker.MOSSE(valRange,
                          tempSize,
                          sigma,
                          eps,
                          trainSteps,
                          rate)

    for i in range(1, nImg+1):
        # generate filenames for output images
        imgFile, temFile, resFile, filFile = routines.genFilenames(objName, i)
        
        # set new image
        I = img.read(imgFile)
        track.setImage(I)
        
        if i == 1:
            # get initial object position centered on target
            ##track.initObjPos()
            objPos = [256, 256]
            track.setObjPos(objPos)
            
            # initialize MOSSE filter and output            
            track.cropTemplate()
            track.calOptimalResponse()
            track.initFilter()
            
            # get initialized templates
            f, h, g, objPos = track.getResults()
            #f = track.getTemplate()
            #g = track.getResponse()
            #h = track.getFilter()
            
        else:
            # correlate old filter with new template
            track.cropTemplate()
            track.calFilterResponse()
            track.calObjPos()
            
            # get results
            f, h, g, objPos = track.getResults()
            #f = track.getTemplate()
            #g = track.getResponse()
            #h = track.getFilter()
            #objPos = track.getObjPos()
            
            # update MOSSE filter and output on new object position
            track.cropTemplate()
            track.calOptimalResponse()
            track.updateFilter()

        # fit images to valRange for output
        f = (valRange-1)/f.max() * f
        g = (valRange-1)/g.max() * g
        h = (valRange-1)/h.max() * h
        
        # save results
        img.write(temFile, f)
        img.write(resFile, abs(g))
        img.write(filFile, abs(np.fft.ifftshift(h)))
        
        # show results in console
        print('------------------------')
        print('iteration: ', i)
        print('objPos: ', objPos)
        vis.comp3Heat(f, abs(np.fft.ifftshift(h)), abs(g))


#%%
main()
