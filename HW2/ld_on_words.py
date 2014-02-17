import sys
import numpy as np
from termcolor import colored

prune_thresh = True
use_beam = True

# Creating a class for every element in the DTW matrix
class matrix_obj:
    def __init__(self):
        self.lowest_cost = 99999999 # Stores lowest cost
        self.back_ptr = None        # Tuple points to lowest input point
        self.is_on = True           # Is this on (for beam search)
        self.symb = None            # Used just for pretty printing


# Compare characters
def get_comp_cost(char_1, char_2):
    if char_1 == char_2:
        comp_cost=0
    else:
        comp_cost=1
    return comp_cost
    
    

if len(sys.argv) <= 2:
    print "Usage:\npython levenshtein_dist.py file1 file2\n"
    exit(0)

with open(sys.argv[1],'r') as f:
    for line in f:
        word1 = line.rstrip('\n')

with open(sys.argv[2],'r') as f:
    for line in f:
        word2 = line.rstrip('\n')


word1 = word1.lower()
word2 = word2.lower()

inp_chars = word1.split()
template_chars = word2.split()

# Appending a null character at the beginning of each string
inp_chars.insert(0,'*')
template_chars.insert(0,'*')

# Creating a list of lists for the DTW matrix
DTW_matrix =[]
for i in xrange(len(template_chars)):

    DTW_row = []
    for i in xrange(len(inp_chars)):

        DTW_row.append(matrix_obj())

    DTW_matrix.append(DTW_row)



# Let's do DTW !!
pruning_threshold = 3 
beam_thresh = 3
for i in xrange(len(inp_chars)):

    for j in xrange(len(template_chars)):

        # Let's make an array to find the minimum costs
        if (i==0 and j== 0):
            costs=[0 + get_comp_cost(inp_chars[i], template_chars[j]), 
                   99999, 99999]
            # diag | horiz|  vert|
            
        elif( i == 0 ):
            if DTW_matrix[j-1][i].is_on == True:
                costs = [99999, 99999, 
                         DTW_matrix[j-1][i].lowest_cost
                         + get_comp_cost(inp_chars[i], template_chars[j])]
            else:
                costs = [99999, 99999, 99999]


        elif ( j == 0):
            if DTW_matrix[j][i-1].is_on == True:
                costs = [99999, 
                         DTW_matrix[j][i-1].lowest_cost
                         + get_comp_cost(inp_chars[i], template_chars[j]), 
                         99999]
            else:
                costs = [99999, 99999, 99999]
            
        else:
            if (DTW_matrix[j-1][i-1].is_on == False and
                DTW_matrix[j][i-1].is_on == False and
                DTW_matrix[j-1][i].is_on == False):
                costs = [99999, 99999, 99999]
            else:
                costs = [DTW_matrix[j-1][i-1].lowest_cost 
                         + get_comp_cost(inp_chars[i], template_chars[j]), 
                         DTW_matrix[j][i-1].lowest_cost 
                         + get_comp_cost(inp_chars[i], template_chars[j]),
                         DTW_matrix[j-1][i].lowest_cost 
                         + get_comp_cost(inp_chars[i], template_chars[j])]

        
            
        DTW_matrix[j][i].lowest_cost =  np.min(costs)
        min_ptr = np.argmin(costs)

        if prune_thresh == True:

            if(DTW_matrix[j][i].lowest_cost > pruning_threshold):
                DTW_matrix[j][i].is_on = False
                DTW_matrix[j][i].lowest_cost = 99999
                min_ptr = 5



        if (i!=0 or j!=0):
            
            if min_ptr == 0:
                DTW_matrix[j][i].back_ptr = (j-1,i-1)
                DTW_matrix[j][i].symb = '/'
            elif min_ptr == 1:
                DTW_matrix[j][i].back_ptr = (j, i-1)
                DTW_matrix[j][i].symb = '-'
            elif min_ptr ==2:
                DTW_matrix[j][i].back_ptr = (j-1, i)
                DTW_matrix[j][i].symb = '|'
            else:
                DTW_matrix[j][i].back_ptr = (-1, -1)
                DTW_matrix[j][i].symb = 'x'
            
        else:
            DTW_matrix[j][i].back_ptr = (-9,-9)
            DTW_matrix[j][i].symb = '\\'

    if use_beam == True:
        best_cost_sofar = 99999
        for j in xrange(len(template_chars)):
            if  DTW_matrix[j][i].lowest_cost <= best_cost_sofar:
                best_cost_sofar = DTW_matrix[j][i].lowest_cost
        pruning_threshold = best_cost_sofar + beam_thresh
        

inserts = 0
delets = 0
substs = 0                                     
best_cost = 99999
best_ptr = None
for j in xrange(len(template_chars)):
    if DTW_matrix[j][len(inp_chars)-1].lowest_cost < best_cost:
        best_cost = DTW_matrix[j][len(inp_chars)-1].lowest_cost
        best_ptr = (j, len(inp_chars)-1)

if DTW_matrix[j][len(inp_chars)-1].symb == '-':
    inserts = inserts + 1
else:
    if DTW_matrix[j][len(inp_chars)-1].symb == '|':
        delets = delets + 1

DTW_matrix[j][len(inp_chars)-1].symb = '+'


right_word = []
# Let's unwrap and print out the closest word
while (template_chars[best_ptr[0]] != '*' and
       inp_chars[best_ptr[1]]!= '*'):

    #This is an insertion error                                                   
    if DTW_matrix[best_ptr[0]][best_ptr[1]].symb != '-':
        right_word.insert(0,template_chars[best_ptr[0]])
        
    if DTW_matrix[best_ptr[0]][best_ptr[1]].symb == '-':
        inserts = inserts + 1
    else:
        if DTW_matrix[best_ptr[0]][best_ptr[1]].symb == '|':
            delets = delets + 1

    DTW_matrix[best_ptr[0]][best_ptr[1]].symb = '+'
    best_ptr = DTW_matrix[best_ptr[0]][best_ptr[1]].back_ptr





# Print out the DTW matrix
print 'DTW Matrix'
for j in reversed(xrange(len(template_chars))):

    print colored(template_chars[j],'blue'),'\t',
    for i in xrange(len(inp_chars)):
        
        if DTW_matrix[j][i].is_on == False:
            print colored('x','red'),'\t',
        elif DTW_matrix[j][i].symb == '+':
            print colored(DTW_matrix[j][i].lowest_cost,'green'),'\t',
        else:
            print DTW_matrix[j][i].lowest_cost,'\t',

    print '\n',

print '\t',
for i in xrange(len(inp_chars)):
    print colored(inp_chars[i],'blue'),'\t',
print '\n\n\n',

# print 'DTW backpointers'
# # Print out the backpointer matrix
# for j in reversed(xrange(len(template_chars))):

#     print colored(template_chars[j],'blue'),'\t',
#     for i in xrange(len(inp_chars)):

#         print DTW_matrix[j][i].back_ptr,'\t',

#     print '\n',

# print '\t',
# for i in xrange(len(inp_chars)):
#     print colored(inp_chars[i],'blue'),'\t',
# print '\n\n\n',


print 'DTW backpointer visualization'
# Print out the backpointers visualization
for j in reversed(xrange(len(template_chars))):

    print colored(template_chars[j],'blue'),'\t',
    for i in xrange(len(inp_chars)):

        if DTW_matrix[j][i].is_on == False:
            print colored(DTW_matrix[j][i].symb,'red'),'\t',
        elif DTW_matrix[j][i].symb == '+':
            print colored(DTW_matrix[j][i].symb,'green'),'\t',
        else:
            print DTW_matrix[j][i].symb,'\t',

    print '\n',

print '\t',
for i in xrange(len(inp_chars)):
    print colored(inp_chars[i],'blue'),'\t',
print '\n\n',

print 'Edit distance: ',DTW_matrix[len(template_chars)-1][len(inp_chars)-1].lowest_cost

substs = DTW_matrix[len(template_chars)-1][len(inp_chars)-1].lowest_cost - (inserts + delets)

print "Insertions:",inserts
print "Substitutions:",substs
print "Deletions:",delets

print "\nAlignment:",right_word
