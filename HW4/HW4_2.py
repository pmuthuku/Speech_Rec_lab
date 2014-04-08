import time
import re
import sys
from colorama import Back

DICTIONARY_FILE_NAME='dict.txt'

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
    
    
    print nlist
    print map(str,range(0, len(nlist)))
    print rlist
    print map(str,elist)
    



    # Code below does spell error counting useless for problem 2


    #################################################################################################################################

    fl=open('typos.txt')
    wlist=[]
    for line in fl.readlines():
       ls=line.strip()
       ls=re.sub('[!@#\"?,.;:]','',ls)
       for word in ls.split():
           if not word=="":
               wlist.append(word.lower())
    
    print wlist
    # time.sleep(60)
    # wlist=[l.strip() for l in fl]
    # wlist=['oncd','uoon','a','time','0909']
    cptr=root
    print wlist
    print cptr.children.index(Node('c'))
    nerr=0
    chk=root.children[19]
    print chk.value
    print len(chk.children)
    print chk.end_of_word
    errwords=[]
    for wd in wlist:
       wlen=len(wd)
       cptr=root
       print 'processing {0}'.format(wd)
       for m,c in enumerate(wd):
           ids=[]
           ids=[i for i,nd in enumerate(cptr.children) if nd==Node(c)]
            #print '{0}---{1}--{2}'.format(c,ids,cptr.value)
           if len(ids)==0:
               nerr=nerr+1
               errwords[len(errwords):]=[wd]
                #print '{0} error in {1} with {2}'.format(c,wd,nerr)
               break
           else:
                
               if m!=wlen-1:
                   if len(ids)==1:
                       cptr=cptr.children[ids[0]]
                   elif len(ids)==2:
                       if not cptr.children[ids[0]].end_of_word:
                           cptr=cptr.children[ids[0]]
                            #print "here"
                            #for gh in cptr.children:
                            #    print gh.value
                       elif not cptr.children[ids[1]].end_of_word:
                           cptr=cptr.children[ids[1]]
                            #print cptr.value
                            #print len(cptr.children)
                            #print "hereb"
                            #print
                            #for gh in cptr.children:
                            #    print gh.value
                       else:  # A few extra checks to make sure that tree is proper
                           print "Something Wrong --- Two Children with same value but flags for both are ON "
                           sys.exit()
                   else:
                       print "Three Children with same value-- NOT POSSIBLE"
                       sys.exit()

               elif m==wlen-1:
                   if len(ids)==1: #if teo nodes one flag is bound to be true
                       if not cptr.children[ids[0]].end_of_word:
                           nerr=nerr+1
                           errwords[len(errwords):]=[wd]
                    
                               
    print Back.RED + str(nerr) + '  ERRORS' + Back.WHITE
    print len(wlist)
    for w in errwords:
       print w

if __name__ == '__main__':
    main()

            
            
# `root` object is the root node loaded with all children.
