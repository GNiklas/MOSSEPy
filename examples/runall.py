#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:42:39 2020

@author: niklas
"""

from mossepy.mosse_tracker import MOSSE

# choose position of object in first frame
# that should be done by mouse click
objPos = [256, 256]

# choose tracker type
tracker = MOSSE()
# initialize object position in first frame
tracker.setObjPos(objPos)
# start tracking
tracker.trackImg()