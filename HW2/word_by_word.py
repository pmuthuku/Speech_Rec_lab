import sys

import HW2_ids

if len(sys.argv) <= 1:
    print "Usage:\npython word_by_word.py pasted_stories\n"
    sys.exit(0)

inserts = 0
delets = 0
substs = 0

with open(sys.argv[1],'r') as f:
    for line in f:
        line = line.rstrip('\n')
        words = line.split()
        errs = HW2_ids.lvdiscount(words[0],words[1])
        inserts = inserts + errs[0]
        delets = delets + errs[1]
        substs = substs + errs[2]
        

total_errs = inserts + delets + substs

print "Insertions:\t",inserts
print "Deletions:\t", delets
print "Substitutions:\t", substs

print "\nTotal Errors:\t", total_errs 




