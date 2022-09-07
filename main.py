from part1 import *

file = "data1.csv"
X, Y = Tree.load_dataset()

t = Node(X=X, Y=Y)

t = Tree.render(t, "", "none")