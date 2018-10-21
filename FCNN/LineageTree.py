import math

class LineageTree:
    frameno = 0

    def __init__(self, root, no):
        self.root = self.Node(root)         # Store Root of binary tree
        self.frameno = no

    # Insert New Centroid into Tree
    def insert(self, node):
        return self.root.insert(self.Node(node))

    #P Convert to Newick String format
    def __str__(self):
        return "("+str(self.root)+");"


    class Node:

        def __init__(self, data):
            self.position = data[0:2]
            self.id = data[2]  # 1.2.1.1.2
            self.tag = data[3] # number
            self.duration = 1
            self.left = None
            self.right = None

        def insert(self, leaf):

            if self.id == leaf.id:
                self.duration += 1
                self.position = leaf.position
                return True


            if self.id == leaf.id[:len(self.id)]:

                if leaf.id[len(self.id)+1]=='1':
                    if self.left:
                        return self.left.insert(leaf)

                    self.left = leaf

                elif leaf.id[len(self.id)+1]=='2':
                    if self.right:
                        return self.right.insert(leaf)

                    self.right = leaf

                return True

            return False


        def __str__(self):

            if not (self.left and self.right):
                return str(self.tag)+":"+str(self.duration)

            return "("+str(self.left)+","+str(self.right)+")"+str(self.tag)+":"+str(self.duration)

