import numpy as np
import time
import re
import sys
#from colorama import Back, Fore

np.set_printoptions(threshold='nan', linewidth = 200)

DICTIONARY_FILE_NAME='dict_new.txt'
# RESULT_FILE_NAME = 'result_file_name_correct1.txt'
# INPUT_FILE_NAME = 'unsegmented0.txt'
RESULT_FILE_NAME = sys.argv[2]
INPUT_FILE_NAME = sys.argv[1]

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


def custom_print(bkptr_matrix_row,bkptr_matrix_col, dist_matrix, rlist, elist, nlist, word):
    ptr = np.shape(dist_matrix)[0] -1
    word_end_indices = np.where(np.array(elist) == 1)[0];
    (next_i, next_j) = (dist_matrix.shape[0]-1, (word_end_indices[np.where(dist_matrix[ptr, word_end_indices] == np.amin(dist_matrix[ptr, :]))[0]][0]))
    reset = False
    i = dist_matrix.shape[0] - 1

    result = []

    while i >= 0:
        for j in range(dist_matrix.shape[1])[::-1]:
            # if(reset):
            # i = i + 1;
            #
            # reset = False;
            # (next_i, next_j) = (dist_matrix.shape[0]-1, (word_end_indices[np.where(dist_matrix[ptr, word_end_indices] == np.amin(dist_matrix[ptr, :]))[0]]))
            # ptr = ptr + 1;
            # continue;
            format_str = ''
            if(i == next_i and j == next_j):
                #format_str = Back.RED
                next_i = bkptr_matrix_row[i][j]
                next_j = bkptr_matrix_col[i][j]

                if j == 0: result.append(i)

            #if(j == 0 and format_str == Back.RED):
            #    format_str = format_str + '*'
            #    reset = True
            #data_str = '%-4.0f' % (dist_matrix[i][j]) + ' ' + Back.WHITE
            #if j == 0: data_str = data_str + '%-5s'%word[i - 1]
            #sys.stdout.write(format_str + data_str)

        i = i -1
        #print('\n')
        ptr = ptr -1

    #for chr in nlist[::-1]:
    #    sys.stdout.write('%-5s'%chr)
    #print '\n'

    #while 
    result = [k - 1 for k in result]
    print result
    fp = open(RESULT_FILE_NAME, 'w')
    for idx, chr in enumerate(word):
        
        if idx in result:
            fp.write(' ')
            fp.write(chr)
            sys.stdout.write(' '),
            sys.stdout.write(chr)
        else: 
            sys.stdout.write(chr),
            fp.write(chr)
    fp.close()



def main():
    k=0 #overall counter-- at the end it will give total number nodes
    root = Node('*')
    root.pno=-1 # root's parent node number -1
    root.ono=k # root's own number 0
    rlist=[-1] # list for row number of parent
    elist=[-1] # list for whether row is end of word or not
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

    fl=open(INPUT_FILE_NAME)
    word=''
    for line in fl.readlines():
       ls=line.strip()
       ls=re.sub('[!@#\"?,.;:]','',ls)
       word=word+ls
       #for word in ls.split():
       # if not word=="":
       # wlist.append(word.lower())

    print word
    #print wlist

    dist_matrix = np.empty([len(word) + 1, k+1])
    dist_matrix[:,:] = float('inf')
    bkptr_matrix_row = np.empty_like(dist_matrix)
    bkptr_matrix_col = np.empty_like(dist_matrix)
    bkptr_matrix_row[:,:] = -1
    bkptr_matrix_col[:,:] = -1

    # Back Pointer Config

    # --- : 0
    # / : 1
    # /
    # | : 2
    # |

    for i in range(len(word) + 1):
        if i != 0 :input_char = word[i - 1]
        else : input_char = '*'

        for j, template_char in enumerate(nlist):
            if(i == 0 and j == 0):
                dist_matrix[i][j] = 0
                bkptr_matrix_row[i][j] = -1
                bkptr_matrix_col[i][j] = -1
                continue
            if(i == 0):
                dist_matrix[i][j] = dist_matrix[i][rlist[j]] + 1
                bkptr_matrix_row[i][j] = i
                bkptr_matrix_col[i][j] = rlist[j]
                continue

            if(j == 0):
                word_end_indices = np.where(np.array(elist) == 1)[0];
                dist_matrix[i][j] = np.amin(dist_matrix[i-1,word_end_indices]) 
                temp = np.empty_like(dist_matrix[i-1,:])
                #temp[:] = -1
		temp[:]=np.inf
                temp[word_end_indices] = dist_matrix[i-1,word_end_indices]
		xc=np.argmin(temp)
                bkptr_matrix_col[i][j] = xc#np.where(dist_matrix[i][j] == temp)[0][0]
                bkptr_matrix_row[i][j] = i - 1
                continue


	    if rlist[j]==0:
                c1=dist_matrix[i-1][j]+1
                factor = 1
                if template_char == input_char: factor = 0
                c2 = dist_matrix[i][rlist[j]] + factor
                this_cost=min(c1,c2)
                if this_cost == c2:
                    bkptr_matrix_row[i][j] = i
                    bkptr_matrix_col[i][j] = rlist[j]
                elif this_cost == c1:
                    bkptr_matrix_row[i][j] = i -1
                    bkptr_matrix_col[i][j] = j
                dist_matrix[i][j] = this_cost
                continue

            c1 = dist_matrix[i-1][j] + 1
            factor = 1
            if template_char == input_char: factor = 0

            c2 = dist_matrix[i][rlist[j]] + factor
            c3 = dist_matrix[i-1][rlist[j]] + factor

            this_cost = min(c1, c2, c3)

            ## HACK HACK. MUST CLEAN THIS CODE

            if this_cost == c2:
                bkptr_matrix_row[i][j] = i
                bkptr_matrix_col[i][j] = rlist[j]
            elif this_cost == c3:
                bkptr_matrix_row[i][j] = i - 1
                bkptr_matrix_col[i][j] = rlist[j]
            elif this_cost == c1:
                bkptr_matrix_row[i][j] = i -1
                bkptr_matrix_col[i][j] = j
            ##################################

            
            dist_matrix[i][j] = min(c1, c2, c3)
            pass

    custom_print(bkptr_matrix_row, bkptr_matrix_col, dist_matrix, rlist, elist, nlist, word)
    pass


if __name__ == '__main__':
    main()

            
            
# `root` object is the root node loaded with all children.

#
