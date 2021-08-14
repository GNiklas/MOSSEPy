# MOSSEPy

Implementation of the Minimum Output Sum of Squared Error (MOSSE) tracker after Bolme et al. (2010), using Python.

## Goals

The purpose for setting up the given project was getting deeper insights into the working principles of an adaptive correlation tracker in general and particularly the MOSSE tracker. 

As a side effect, general correlation and adaptive correlation tracker classes are given. These could be used in constructing more
variants of tracker classes by inheritance (As is effectively done in case of the MOSSE class).

The focus of the presented work was put onto pure algorithm treatment. To that end, the underlying functionality was implemented using numpy arrays. For effective application of MOSSE you might be interested in OpenCV's implementation.

## Requirements

Python 3.9 was used. The following libraries are imported:

* os
* numpy
* scipy
* matplotlib
* mpl_toolkits
* PIL

## Installation

* if not done, install [numpy](https://numpy.org/install/), [scipy](https://www.scipy.org/install.html), [matplotlib](https://matplotlib.org/stable/users/installing.html), [PIL](https://pypi.org/project/Pillow/)

* clone the repository to your local workspace

```
$ git clone https://github.com/GNiklas/MOSSEPy.git
```

* go to your local MOSSEPy repository

```
$ cd your/local/MOSSEPy
```

* install mossepy using setup

```
$ python setup.py install
```

* uninstall using pip

```
$ pip uninstall mossepy
```

## Running the Tests

No tests have been implemented, yet.

## Usage

The MOSSE tracker can be used as follows:

* import the tracker class

```
from mossepy.mosse_tracker import MOSSE
```

* construct a MOSSE object

```
tracker = MOSSE()
```

* initialize the position of the object to be tracked in first frame

```
tracker.setObjPos(objPos)
```

* track the marked object

```
tracker.trackImg()
```

### In- and Output

By default, input files are sought in data/. All .jpg files within the input directory are read in alphanumerical order. Output files are written to results/ by default. Output to each input file are 

* the template used for tracking (*_tem.jpg), 
* the correlation filter (*_fil.jpg),
* and the correlation response (*_res.jpg).

The estimated object position and plots of the three outputs are printed to console for each time step.

### Examples

An example run file and data are given in examples/. Here, the object position is set to 

```
objPos = [256, 256]
```

The default tracker parameters are used.

## References

<a id="1">[1]</a> 
Bolme, D.S. et al. (2010).
Visual object tracking using adaptive correlation filters.
Conference on Computer Vision and Pattern Recognition (CVPR).
DOI: 10.1109/CVPR.2010.5539960

<a id="2">[2]</a>
Clark, A. (2015).
Pillow (PIL Fork) Documentation, readthedocs.
Retrieved from https://buildmedia.readthedocs.org/media/pdf/pillow/latest/pillow.pdf

<a id="3">[3]</a>
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020).
Array programming with NumPy.
Nature 585, 357–362.
DOI: 0.1038/s41586-020-2649-2

<a id="4">[4]</a>
Hunter, J.D. (2007).
Matplotlib: A 2D graphics environment.
Computing in Science & Engineering 9, 90-95.
DOI: 10.1109/MCSE.2007.55

<a id="5">[5]</a>
Jones, E., Oliphant, T., Peterson, P. et al (2001).
SciPy: Open Source Scientific Tools for Python.
Retrieved from https://www.scipy.org
