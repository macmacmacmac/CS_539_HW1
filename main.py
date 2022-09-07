from part1 import *

file = "data1.csv"
X, Y = Tree.load_dataset()

Header = np.array(["debt", "income","married", "owns_property", "gender", "risk"])
#Header = np.array(["cylinders","displacement","horsepower","weight","acceleration","Model", "year","maker", "mpg"])

Tom = np.array(["low", "low", "no", "Yes", "Male"])
Ana = np.array(["low", "medium", "yes", "Yes", "female"])

t = Tree.train(X, Y)

print("Tom's Predicted risk: ", Tree.inference(t, Tom))
print("Ana's Predicted risk: ", Tree.inference(t, Ana))

t = Tree.render(t, "", "", Header, "")

