# Some general math utilities (all work on numpy arrays)

import numpy as np
""" Define an internal EPS for precision and floor protection against log's of 0 
    
    With flt32 arrays, a safe setting is EPS = 1.e-39 which is just within flt32 precision
        thus    LOGEPS = -89.801 .  LOG10EPS = -39.000
        
    With input arrays are flt64, it is sage to set EPS to 1.e-300, corresponding to 
    a LOGEPS of -690.7755
    """

__EPS__ = 1.e-39
__LOGEPS__  = np.log(__EPS__)
__LOG10EPS__ = np.log10(__EPS__)
__LOG10__ = np.log(10.0)

# SOME UTILITIES
def normalize(x, axis=0):
    """Normalizes a multidimensional input array so that the values sums to 1 along the specified axis
    Typically applied to some multinomal distribution

    x       numpy array
            of not normalized data
    axis    int
            dimension along which the normalization should be done

    """

    xs = x.sum(axis,keepdims=True)
    xs[xs<__EPS__]=1.
    shape = list(x.shape)
    shape[axis] = 1
    xs.shape = shape
    return(x / xs)

def floor(x,FLOOR):
    """ array floor:  returns  max(x,FLOOR)  """
    return(np.maximum(x,FLOOR))

def logf(x,eps=__EPS__):
    """ array log with flooring """
    return(np.log(np.maximum(x,eps)))
    
def log10f(x,eps=__EPS__):
    """ array log10 with flooring """
    return(np.log10(np.maximum(x,eps)))
    
def convertf(x,iscale="lin",oscale="log",eps=__EPS__):
    """ array conversions between lin, log and log10 with flooring protection """
    if iscale == oscale: 
        return x
    
    if iscale == "lin":
        if oscale == "log":
            return logf(x,eps)
        elif oscale == "log10":
            return log10f(x,eps)
    elif iscale == "log":
        if oscale == "lin":
            return np.exp(x)
        elif oscale == "log10":
            return x/__LOG10__
    elif iscale == "log10":
        if oscale == "lin":
            return np.power(10.0,x)
        elif oscale == "log":
            return x*__LOG10__
        