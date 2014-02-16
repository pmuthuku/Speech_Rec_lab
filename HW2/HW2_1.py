#!/usr/bin/python

import sys
import os
import numpy
import re
import time

def lvdis(temp,inp):
	m=len(temp)
	n=len(inp)
	#print m
	lvdst=numpy.zeros((m+1,n+1))
	#print lvdst
	back=numpy.ones((m+1,n+1))
	back=-1*back
	
	dummyc=numpy.arange(n+1)
	dummyr=numpy.arange(m,-1,-1)
	lvdst[:,0]=dummyr
	lvdst[m,:]=dummyc
				
	for i,sin in enumerate(inp):
		for j,ste in enumerate(temp):
			ccost=[]
			if sin==ste:
				cost=0
			else:
				cost=1
			ccost[len(ccost):]=[1+lvdst[m-1-j,i]]
			ccost[len(ccost):]=[1+lvdst[m-1-(j-1),i+1]]
			ccost[len(ccost):]=[cost+lvdst[m-1-(j-1),i]]
			
			lvdst[m-1-j,i+1]=min(ccost)
			id=ccost.index(min(ccost))
			back[m-1-j,i+1]=id
			
	#print lvdst[0,n]
	return(lvdst,back,lvdst[0,n])

def lvdisprdis(temp,inp,th):
	m=len(temp)
	n=len(inp)
	th=int(th)
	lvdst=numpy.zeros((m+1,n+1))
	#lvdst=-1*lvdst
	back=numpy.ones((m+1,n+1))
	back=-1*back
	#back[:,0]=numpy.zeros(m+1)
	#back[m,:]=numpy.zeros(n+1)
	dummyc=numpy.arange(n+1)
	dummyr=numpy.arange(m,-1,-1)
	lvdst[:,0]=dummyr
	lvdst[m,:]=dummyc
				
	for i,sin in enumerate(inp):
		for j,ste in enumerate(temp):

			if lvdst[m-1-j,i] < th  or lvdst[m-1-(j-1),i+1] < th  or lvdst[m-1-(j-1),i] < th:
 
				#print "here"
				ccost=[]
				if sin==ste:
					cost=0
				else:
					cost=1
				ccost[len(ccost):]=[1+lvdst[m-1-j,i]]
				ccost[len(ccost):]=[1+lvdst[m-1-(j-1),i+1]]
				ccost[len(ccost):]=[cost+lvdst[m-1-(j-1),i]]
				#if min(ccost) >= th:
				#	print ""
				lvdst[m-1-j,i+1]=min(ccost)
				id=ccost.index(min(ccost))
				back[m-1-j,i+1]=id
			else:
				lvdst[m-1-j,i+1]=99
			
	#print lvdst[0,n]
	return(lvdst,back,lvdst[0,n])

def lvdisprbm(temp,inp,th):
	m=len(temp)
	n=len(inp)
	th=int(th)
	lvdst=numpy.zeros((m+1,n+1))
	#lvdst=-1*lvdst
	back=numpy.ones((m+1,n+1))
	back=-1*back
	#back[:,0]=numpy.zeros(m+1)
	#back[m,:]=numpy.zeros(n+1)
	dummyc=numpy.arange(n+1)
	dummyr=numpy.arange(m,-1,-1)
	lvdst[:,0]=dummyr
	lvdst[m,:]=dummyc
				
	for i,sin in enumerate(inp):
		for j,ste in enumerate(temp):

			if i==0:
 
				ccost=[]
				if sin==ste:
					cost=0
				else:
					cost=1
				ccost[len(ccost):]=[1+lvdst[m-1-j,i]]
				ccost[len(ccost):]=[1+lvdst[m-1-(j-1),i+1]]
				ccost[len(ccost):]=[cost+lvdst[m-1-(j-1),i]]
				#if min(ccost) >= th:
				#	print ""
				lvdst[m-1-j,i+1]=min(ccost)
				id=ccost.index(min(ccost))
				back[m-1-j,i+1]=id
			else:
				if lvdst[m-1-j,i]!=99  or lvdst[m-1-(j-1),i+1]!=99  or lvdst[m-1-(j-1),i]!=99:
					ccost=[]
					if sin==ste:
						cost=0
					else:
						cost=1
					ccost[len(ccost):]=[1+lvdst[m-1-j,i]]
					ccost[len(ccost):]=[1+lvdst[m-1-(j-1),i+1]]
					ccost[len(ccost):]=[cost+lvdst[m-1-(j-1),i]]
				#if min(ccost) >= th:
				#	print ""
					lvdst[m-1-j,i+1]=min(ccost)
					id=ccost.index(min(ccost))
					back[m-1-j,i+1]=id
				else:
					lvdst[m-1-j,i+1]=99#float('inf')
		x=lvdst[:,i+1]
		p=numpy.nonzero(x > min(x)+th)
		x[p]=99
		
		
	#print lvdst[0,n]
	return(lvdst,back,lvdst[0,n])
			
			
	
def main(argv):
	
	
	if len(argv) < 6:
	      print "Insuffucient Arguments"
	      print "-w/-f input single word/file -t/-d template list/file -p n/d/b"
	      sys.exit()
	stime=time.clock()
	if argv[1]=='-w':
		wlist=[]
		wlist[len(wlist):]=[argv[2]]
	elif argv[1]=='-f':
		fl=open(argv[2],'r')
		wlist=[]
		for line in fl.readlines():
			line.rstrip()
			line=re.sub('[!@#\"?,.;:]','',line)
			for word in line.split():
				if not word=="":
					wlist.append(word.lower())
		fl.close()
	else:
		print "Input Word"
		print "-w/-f inputsingleword/file -t/-d templatelist/file -p n/d/b th "
		sys.exit()

	if '-p' in argv:
		pid=argv.index('-p')
		prune=argv[pid+1]
		if prune=='d' or prune=='b':
			if len(argv)-1 < pid+2:
				print "Need threshold size"
				print "-w/-f inputsingleword/file -t/-d templatelist/file -p n/d/b th"
				sys.exit()
			else:
				th=argv[pid+2]
		elif prune=='n':
			th=0;
		else:
			print "wrong pruning method"
			print "-w/-f inputsingleword/file -t/-d templatelist/file -p n/d/b th"
			sys.exit()
	else:
		print "Pruning arguments"
		print "-w/-f inputsingleword/file -t/-d templatelist/file -p n/d/b th"
		sys.exit()

	if argv[3]=='-t':
		tlist=argv[4:pid]
	elif argv[3]=='-d':
		fl=open(argv[4],'r')
		tlist=[]
		for line in fl.readlines():
			for word in line.split():
				tlist.append(word)
		fl.close()
	else:
		print "Unreasonable Templates"
		print "-w/-f input single word/file -t/-d template list/file -p n/d/b th"
		sys.exit()

	#mindis=[]
	cltemp=[]
	cldis=[]
	#print tlist
	if argv[1]=='-w' and argv[3]=='-t':
		show=1
	else:
		show=0
	for inp in wlist:
		mindis=[]
		#inp=inp.lower()
		print inp
		for temp in tlist:
			#temp=temp.lower()
			#print temp
			if prune=='n':
				#print "Calculating Unpruned LV"
				lvdst,back,dist=lvdis(temp,inp)
				if show==1:
					print '-------------DIST MATRIX FOR {0} ---------'.format(temp)
					print lvdst[0:-1,1::]
					print '-------------BACK MATRIX FOR {0}---------'.format(temp)
					print back[0:-1,1::]
			elif prune=='d':
				#print "Calculating Pruning Min Dist."
				#print th
				lvdst,back,dist=lvdisprdis(temp,inp,th)
				if show==1:
					print '-------------DIST MATRIX FOR {0} ---------'.format(temp)
					print lvdst[0:-1,1::]
					print '-------------BACK MATRIX FOR {0}---------'.format(temp)
					print back[0:-1,1::]
				
			elif prune=='b':
				#print "Calculating Pruning Beam Search."
				lvdst,back,dist=lvdisprbm(temp,inp,th)
				if show==1:
					print '-------------DIST MATRIX FOR {0} ---------'.format(temp)
					print lvdst[0:-1,1::]
					print '-------------BACK MATRIX FOR {0}---------'.format(temp)
					print back[0:-1,1::]
			else:
				print "-w/-f input single word/file -t/-d template list/file -p n/d/b th"
				sys.exit()
			mindis[len(mindis):]=[dist]
			if dist==0:
				break;
			
		#print mindis
		cldis[len(cldis):]=[min(mindis)]
		#print  min(mindis)
		#clid=mindis.index(min(mindis))
		#print clid
		cltemp[len(cltemp):]=[tlist[mindis.index(min(mindis))]]
		#print tlist[clid]
	#print cltemp
	#print wlist[0]
	#print cltemp[0]
	
	fl=open('Checked.txt','w')
	for i,inp in enumerate(wlist):
		fl.write(cltemp[i]+' ')
		print '{0} ---> {1} ---> {2}'.format(inp,cltemp[i],cldis[i])
		if (i%20)==0:
			fl.write('\n')
	print time.clock()-stime
	print len(wlist)
	#fl.close()
#print temp
        #print inp
	#lvdst,back,dist=lvdisprdis(temp,inp,3)
	#lvdstu,backu,distu=lvdis(temp,inp)
	#print "xxxxxxxxxxxxxxxxxxxxx PRINTING DISTANCE MATRIX xxxxxxxxxxxxxxxxxxxxxx"
	#print lvdst 
	#print lvdst#[0:-1,1::]

	#print "xxxxxxxxxxxxxxxxxxxxx PRINTING back MATRIX xxxxxxxxxxxxxxxxxxxxxx"
	#print back[0:-1,1::]
	
	#print "xxxxxxxxxxxxxxxxxxxxx PRINTING back MATRIX xxxxxxxxxxxxxxxxxxxxxx"
	#print lvdstu
	#print "xxxxxxxxxxxxxxxxxxxxx PRINTING MINIMUM DISTANCE xxxxxxxxxxxxxxxxxxxxxx"
	#print backu



if __name__ == '__main__':
	main(sys.argv)
