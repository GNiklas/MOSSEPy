#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routines for visualization of numpy arrays

Created on Thu Nov 19 11:24:28 2020

@author: niklas
"""


import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
def comp3Heat(M1, M2, M3, title1='template', title2='filter', title3='response'):
    '''
    Compare three heat plots in common figure.
    
    Parameters
    ----------
    extent: tuple 
        boundaries of the region to be plotted
    M1 : numpy array
        template
    M2 : numpy array
        MOSSE filter
    M3 : numpy array
        response of template to filter
    title1 : string. optional.
        name of first array. default is 'template'.
    title2 : string. optional.
        name of second string. default is 'filter'.
    title3 : string. optional.
        name of second string. default is 'response'.

    Returns
    -------
    None

    '''
    
    # define style of colorbar
    cMap = 'jet'
    
    # define figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3)

    # plot initial state with colorbar
    ax1.set_title(title1)     
    img1 = ax1.imshow(M1, vmin=M1.min(), vmax=M1.max(), cmap=cMap)
    colorBar(img1)

    # plot final state with colorbar
    ax2.set_title(title2)       
    img2 = ax2.imshow(M2, vmin=M2.min(), vmax=M2.max(), cmap=cMap)
    colorBar(img2)

    # plot final state with colorbar
    ax3.set_title(title3)       
    img3 = ax3.imshow(M3, vmin=M3.min(), vmax=M3.max(), cmap=cMap)
    colorBar(img3)
    
    # adjust axes placing
    fig.tight_layout(h_pad=1)
    
    plt.show()

       
def colorBar(mappable):
    '''
    Create colorbar for an ax.

    Parameters
    ----------
    mappable : img = ax. ...
        creates colorbar while keeping the axes

    Returns
    -------
    cBar : ax.colorbar

    '''
    
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cAx = divider.append_axes("right", size="5%", pad=0.05)
    cBar = fig.colorbar(mappable, cax=cAx)
    plt.sca(last_axes)
    
    return cBar
