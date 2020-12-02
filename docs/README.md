# MOSSEPY

Implementation of the Minimum Output Sum of Squared Error (MOSSE) tracker after Bolme et al. (2010).

## Goals

The purpose for setting up the given project was getting deeper insights into the working principles of an adaptive correlation tracker in general and particularly the MOSSE tracker. 

As a side effect, general correlation and adaptive correlation tracker classes are given. These could be used in constructing more
variants of tracker classes by inheritance (As is effectively done in case of the MOSSE class).

The focus of the presented work was put onto pure algorithm treatment. To that end, the underlying functionality was implemented using numpy arrays. For effective application of MOSSE you might be interested in OpenCV's implementation.

## Prerequisites

Python 3.7.4 was used. The following libraries are needed:

* os
* numpy
* scipy
* matplotlib
* mpl_toolkits
* PIL

## Running the Tests

No tests have been implemented, yet.

## Usage

The MOSSE tracker can be used as follows:

* import the tracker module

'''
import tracker
'''

* construct a MOSSE object.

'''
track = tracker.MOSSE()
'''

* initialize the position of the object to be tracked in first frame

'''
track.setObjPos(objPos)
'''

* track the marked object

'''
track.trackImg()
'''

### In- and Output

By default, input files are sought in data/. All .jpg files within the input directory are read in alphanumerical order. Output files are written to results/ by default. Output to each input file are 

* the template used for tracking (*_tem.jpg), 
* the correlation filter (*_fil.jpg),
* and the correlation response (*_res.jpg).

The estimated object position and plots of the three outputs are printed to console for each time step.

### Examples

An example run file is given in runall.py Here, the object position
is set to 

'''
objPos = [256, 256]
'''

The default tracker parameters are used.

## Acknowledgements

The presented implementation follows the MOSSE algorithm as lined out in:

* D.S. Bolme et al. Visual object tracking using adaptive correlation filters. Conference on Computer Vision and Pattern Recognition (CVPR), (2010). DOI: 10.1109/CVPR.2010.5539960

Coding style of the tracker classes was inspired by:

* S. Arabas et al. Formula translation in Blitz++, NumPy and modern Fortran: A case study of the language choice tradeoffs. Scientific Programming 22 (2014) 201–222. DOI: 10.3233/SPR-140379

For further reading on best practices and project set up see also:

* G. Wilson, D.A. Aruliah, C. Titus Brown, N.P. Chue Hong, M. Davis, R.T. Guy, S.H.D. Haddock, K. Huff, I.M. Mitchell, M. Plumbley, B. Waugh, E.P. White and P. Wilson, Best Practices for Scientific Computing, PLoS Biol. 12(1) (2014),
e1001745. DOI: 10.1371/journal.pbio.1001745
* R.C. Jiménez, M. Kuzak, M. Alhamdoosh et al. Four Simple Recommendations to Encourage Best Practices in
Research Software. F1000Research (2017), 6:876. DOI: 10.12688/f1000research.11407.1
* W.S. Noble. A Quick Guide to Organizing Computational Biology Projects. PLoS Comput Biol 5(7) (2009), e1000424. DOI: 10.1371/journal.pcbi.1000424
