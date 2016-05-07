from numpy import *
from pylab import *
import util

def pca(X, K):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})

    should return a tuple (P, Z, evals)
    
    where P is the projected data (N*K) where
    the first dimension is the higest variance,
    the second dimension is the second higest variance, etc.

    Z is the projection matrix (D*K) that projects the data into
    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''
    
    N,D = X.shape

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    # first, we need to center the data
    X = X - np.mean(X, axis=0)

    # next, compute eigenvalues of the data variance
    #    hint 1: look at 'help(pylab.eig)'
    #    hint 2: you'll want to get rid of the imaginary portion of the eigenvalues; use: real(evals), real(evecs)
    #    hint 3: be sure to sort the eigen(vectors,values) by the eigenvalues: see 'argsort', and be sure to sort in the right direction!
            
    Dcov =  dot(X.T, X) / N
    evals, evecs = eig(Dcov)
    eorder = argsort(real(evals))[::-1][:K]

    evals = real(evals[eorder])
    Z = real(evecs[:, eorder])
    P = dot(X, Z)

    return (P, Z, evals)