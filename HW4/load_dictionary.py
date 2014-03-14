# Change File Name to the required smaller dict for debugging
DICTIONARY_FILE_NAME = 'dict.txt'

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.end_of_word = False

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

if __name__ == '__main__':
    root = Node('*')

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
            if required_child_present and \
                    not tree_ptr.children[findall(tree_ptr.children, lambda x: x == Node(char))[-1]].end_of_word:
                tree_ptr = tree_ptr.children[tree_ptr.children.index(Node(char))]
            else:
                new_node = Node(char)
                if is_last_char : new_node.end_of_word = True
                tree_ptr.add_child(new_node)
                tree_ptr = new_node
pass
# `root` object is the root node loaded with all children.


