#!/usr/bin/python
import train_hmm.py
tlist=[0,1,2,3,4]
#tlist=[5,6,7,8,9]
trlist=range(5,5+1)
        #trlist=range(0,0+int(argv[2]))

        #trl=range(5,10)
        #random.shuffle(trl)
        #trlist=trl[0:int(argv[2])]
ac=0.0
for i in range(0,10):
      for l in tlist:
          inpdat=numpy.loadtxt('all_recs/'+str(i)+'_'+str(l)+'.mfcc')
          
          cost=[]
          for j in range(0,10):
              hmm=numpy.loadtxt('hmm/'+str(j)+'.hmm')
              tram=numpy.loadtxt('trans/'+str(j)+'.trans')
              r,t,score=train_hmm.do_DTW(hmm,tram,inpdat)
              cost[len(cost):]=[score]
                
          print '{0}_{1} recognised as {2}'.format(i,l,cost.index(min(cost)))
          if i==cost.index(min(cost)):
              ac=ac+1.0
          print '\n'
print 'Recognition Accuracy {0}'.format(ac/50)
et=time.clock()
print et-st
