#!/usr/bin/python

import sys
import numpy
#import scipy
import scipy.spatial as scip
import time

def dtwdis(temp,inp):
    dmat=scip.distance.cdist(temp,inp,'euclidean')
    m,n=numpy.shape(dmat)
    
    dcost=numpy.ones((m+2,n+1))
    dcost=dcost+numpy.inf

    dcost[2,1]=dmat[0,0]
    k=3
    for j in range(2,n+1):
       for i in range(2,min(2+k,m+2)):
           dcost[i,j]=min(dcost[i,j-1],dcost[i-1,j-1],dcost[i-2,j-1])+dmat[i-2,j-1]
       k=k+2

    return(dcost[m+1,n])
st=time.clock()
inpdata=numpy.loadtxt(sys.argv[1])
temdata=numpy.loadtxt(sys.argv[2])
cost=dtwdis(temdata,inpdata)
et=time.clock()
print et-st
print cost
