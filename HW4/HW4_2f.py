import time
import re
import numpy
DICTIONARY_FILE_NAME='dict_new.txt'

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.end_of_word = False
        self.pno=-1 #parent node number
        self.ono=-1 #own node number

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_child(self, child_node):
        self.children.append(child_node)

def findall(list, test_function):
    i=0
    indices = []
    while(True):
        try:
            # next value in list passing the test
            nextvalue = filter(test_function, list[i:])[0]

            # add index of this value in the index list,
            # by searching the value in L[i:]
            indices.append(list.index(nextvalue, i))

            # iterate i, that is the next index from where to search
            i=indices[-1]+1
        # when there is no further "good value", filter returns [],
        # hence there is an out of range exeption
        except IndexError:
            return indices

def lvdis(temp,inp,rlist):

    temp=temp[1:]
    m=len(temp)
    n=len(inp)
    #print m
    lvdst=numpy.zeros((m+1,n+1))#*added in list
	#print lvdst
    back=numpy.ones((m+1,n+1))
    back=-1*back
	
    dummyc=numpy.arange(n+1)
    dummyr=numpy.arange(m,-1,-1)
    lvdst[:,0]=dummyr
    lvdst[m,:]=dummyc
    print "herer"
    for i,sin in enumerate(inp):
        #print i,
	for j,ste in enumerate(temp):
            ccost=[0,0,0]
            if sin==ste:
                cost=0
            else:
		cost=1
			#ccost[len(ccost):]=[1+lvdst[m-1-j,i]]
			#ccost[len(ccost):]=[1+lvdst[m-1-(j-1),i+1]]
			#ccost[len(ccost):]=[cost+lvdst[m-1-(j-1),i]]
            ccost[2]=1+lvdst[m-1-j,i]#horizontal cost remains same
            ccost[1]=1+lvdst[m-rlist[j+1],i+1]#vertical cost changes to cost of parent at that input
            ccost[0]=cost+lvdst[m-rlist[j+1],i]#diagonal cost changes to cost of parent at previous input
		
            mcost=min(ccost)
            lvdst[m-1-j,i+1]=mcost
			#id=ccost.index(mcost)
            back[m-1-j,i+1]=ccost.index(mcost)
			
	#print lvdst[0,n]
    return(lvdst,back)


def main():
    k=0 #overall counter-- at the end it will give total number nodes
    root = Node('*')
    root.pno=-1 # root's parent node number -1
    root.ono=k  # root's own number 0
    rlist=[-1]  # list for row number of parent
    elist=[-1]  # list for whether row is end of word or not
    nlist=['*'] # list of nodes in linear fashion
    
    for line in open(DICTIONARY_FILE_NAME):
        tree_ptr = root
        line = line.rstrip('\n')
        for idx, char in enumerate(map(str, line)):
            is_last_char = (idx == len(line) -1)

             # By Default Assume The Child Node With Value `char` is not present
            required_child_present = False

            if Node(char) in tree_ptr.children:
                required_child_present = True
            # if required_child_present and not tree_ptr.children[tree_ptr.children.index(Node(char))].end_of_word:
            if required_child_present and not tree_ptr.children[findall(tree_ptr.children, lambda x: x == Node(char))[-1]].end_of_word:
                tree_ptr = tree_ptr.children[findall(tree_ptr.children, lambda x: x == Node(char))[-1]]
            else:
                k=k+1
                new_node = Node(char)
                new_node.pno=tree_ptr.ono
                new_node.ono=k
                if is_last_char : new_node.end_of_word = True
                tree_ptr.add_child(new_node)
                rlist[len(rlist):]=[tree_ptr.ono]
                nlist[len(nlist):]=[char]
                if is_last_char:
                    elist[len(elist):]=[1]
                else:
                    elist[len(elist):]=[0]

                tree_ptr = new_node
    
    
    #print nlist
    #print map(str,range(0, len(nlist)))
    #print rlist
    #print map(str,elist)

    f=open('unsegmented0.txt','r')
    ip=f.readlines()
    inp=''
    for line in ip:
        ln=line.strip()
        inp=inp+ln

    print inp
    print len(inp)
    lvdist,backp=lvdis(nlist,inp,rlist)
    #print lvdist
    print '###########################################################'
    #print backp
    r,j=numpy.shape(lvdist)
    ids=[r-1-i for i,v in enumerate(elist) if v==0]
    j=j-1
    slist=[]
    while j>1:
        print j
        #time.sleep(2)
        st=lvdist[:,j]
        st[ids]=numpy.inf 
        st=st[0:r-1]# may have to do this for star        
        sti=numpy.argmin(st)
        stj=j
        while sti < r-1:
            #print 'sti=={0}'.format(r-1-sti)
            #print 'stj=={0}'.format(stj)
            #time.sleep(2)
            if backp[sti,stj]==0:
                sti=r-1-rlist[r-1-sti]
                stj=stj-1
                #j=stj-1
            elif backp[sti,stj]==1:
                #j=stj
                sti=r-1-rlist[r-1-sti]
            elif backp[sti,stj]==2:
                #j=stj
                stj=stj-1
            j=stj
        slist[len(slist):]=[j]
            
    
    #print [c for c in inp]
    print '\n'
    #print map(str,range(0,len(inp)))
    slist.insert(0,len(inp))
    slist.reverse()
    print slist
    outp=''
    for i in range(0,len(slist)-1):
        outp=outp+' '+inp[slist[i]:slist[i+1]]
    print outp
if __name__ =='__main__':
    main()
