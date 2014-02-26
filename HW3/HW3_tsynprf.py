#!/usr/bin/python

import sys
#import os
import numpy
#import scipy
import scipy.spatial as scip
#import re
import time

#def dtwdis(temp,inp):
#    dmat=scip.distance.cdist(temp,inp,'euclidean')
#    m,n=numpy.shape(dmat)
    
#    dcost=numpy.ones((m+2,n+1))
#    dcost=dcost+numpy.inf

#    dcost[2,1]=dmat[0,0]
#    k=3
#    for j in range(2,n+1):
#       for i in range(2,min(2+k,m+2)):
#           dcost[i,j]=min(dcost[i,j-1],dcost[i-1,j-1],dcost[i-2,j-1])+dmat[i-2,j-1]
#       k=k+2

#    return(dcost[m+1,n])
#
def dtwtsyn(temp,inp,llist,th):
    
    domat=scip.distance.cdist(temp,inp,'euclidean')
    m,n=numpy.shape(domat)
    dcost=numpy.ones((m,n))+numpy.inf
    fcost=numpy.zeros((len(llist)-1,1))+numpy.inf
    k=3
    #print domat.shape
    for j in range(2,n+1):
           
        sz=0
        
        for l in range(0,len(llist)-1):
            
            #print l
            sz=sz+llist[l]
            #print sz
            dmat=domat[sz:sz+llist[l+1],:]
            
            #print dmat.shape
           
            dcostc=dcost[sz:sz+llist[l+1],:]
            #print min(dcostc[:,j])
            dcostc[0,0]=dmat[0,0]
            
            
            if min(dcostc[:,j-2])==numpy.inf:
                continue
            else:            
                                       #may need some changes
                dudc=numpy.ones((2,n))+numpy.inf
                dcostc=numpy.concatenate((dudc,dcostc),axis=0)
                dudr=numpy.ones((llist[l+1]+2,1))+numpy.inf
                dcostc=numpy.concatenate((dudr,dcostc),axis=1)
                mc,nc=dcostc.shape
                dcostc[2,1]=dmat[0,0]
            
                
                #print dcostc[0:8,0:3]
                #print sz
                #print min(2+k,mc)
                #print dmat.shape
                #print dcostc.shape
                for i in range(2,min(2+k,mc)):
                
                    nodmin=min(dcostc[i,j-1],dcostc[i-1,j-1],dcostc[i-2,j-1])
                    if nodmin!=numpy.inf:
                        dcostc[i,j]=nodmin+dmat[i-2,j-1]
                    else:
                        dcostc[i,j]=numpy.inf
                
            
                dcost[sz:sz+llist[l+1],:]=dcostc[2:mc,1:nc]
                #print pp
                #print dcostc[0:8,0:3]
                #print '\n'
                #time.sleep(5)
                fcost[l]=dcostc[mc-1,nc-1]
                #print "done"
        prrow=dcost[:,j-1]
        prmin=min(prrow)
        prid=numpy.nonzero(prrow > prmin+th)
        prrow[prid]=numpy.inf
        dcost[:,j-1]=prrow
        k=k+2
        

    return fcost
            

def main(argv):
    
     if argv[1]=='-d':
        st=time.clock()
        inpdat=numpy.loadtxt(argv[2])
        #cost=[]
        trlist=range(5,5+int(argv[3]))                                                  
        #trl=range(5,10)
        #random.shuffle(trl)
        #trlist=trl[0:int(argv[3])]
        llist=[0]
        temdat=numpy.zeros((0,inpdat.shape[1]))
        lab=[]
        for i in range(0,10):
            #cos=[]
            for m in trlist:
                temc=numpy.loadtxt('all_recs/'+str(i)+'_'+str(m)+'.mfcc')
                
                temdat=numpy.concatenate((temdat,temc),axis=0)
                llist[len(llist):]=[temc.shape[0]]
                lab[len(lab):]=[i]
                #fcost=dtwtsyn(temdat,inpdat)
                #print '{0}_{1}---{2}'.format(i,m,fcost)
                #cos[len(cos):]=[fcost]
                #    print '\n'
            #cost[len(cost):]=[min(cos)]

        #print '\n'
        costs=dtwtsyn(temdat,inpdat,llist,float(argv[4]))
        #print costs
        labo=lab[numpy.argmin(costs)]
        print 'Input recognised as {0}'.format(labo)
        et=time.clock()
        print et-st
     
     elif argv[1]=='-r':
         st=time.clock()
         tlist=[0,1,2,3,4]
         #tlist=[5,6,7,8,9]                                                               

         trlist=range(5,5+int(argv[2]))                                                  
         #trlist=range(0,0+int(argv[2]))                                                  

         #trl=range(5,10)
         #random.shuffle(trl)
         #trlist=trl[0:int(argv[2])]
         th=float(argv[3])
         ac=0.0
         for i in range(0,10):
            for l in tlist:
                inpdat=numpy.loadtxt('all_recs/'+str(i)+'_'+str(l)+'.mfcc')              
                llist=[0]
                temdat=numpy.zeros((0,inpdat.shape[1]))
                lab=[]
                for j in range(0,10):
                    for m in trlist:
                        temc=numpy.loadtxt('all_recs/'+str(j)+'_'+str(m)+'.mfcc')
                        temdat=numpy.concatenate((temdat,temc),axis=0)
                        llist[len(llist):]=[temc.shape[0]]
                        lab[len(lab):]=[j]
                       
                costs=dtwtsyn(temdat,inpdat,llist,th)
                labo=lab[numpy.argmin(costs)]
                print '{0}_{1} recognised as {2}'.format(i,l,labo)
                if labo==i:
                    ac=ac+1.0
            print '\n'

         print 'Recognition Accuracy {0}'.format(ac/50)
         et=time.clock()
         print et-st

                                                
if __name__ == '__main__':
    main(sys.argv)

