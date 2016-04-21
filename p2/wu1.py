from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
import time
import datasets

# ---------------------
# WU1 OAA AND AVA TESTS
# ---------------------

# ------
# PART A
# ------

# Sauvignon-Blanc

print "Sauvignon-Blanc OAA\n"
OAA = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
OAA.train(WineDataSmall.X, WineDataSmall.Y)
util.showTree(OAA.f[0], WineDataSmall.words)

print "\n--------------------------------------------------\n"
print "Sauvignon-Blanc AVA"

AVA = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
AVA.train(WineDataSmall.X, WineDataSmall.Y)

print "\n1 vs 0"
util.showTree(AVA.f[1][0], WineDataSmall.words)
print "\n2 vs 0"
util.showTree(AVA.f[2][0], WineDataSmall.words)
print "\n3 vs 0"
util.showTree(AVA.f[3][0], WineDataSmall.words)
print "\n4 vs 0"
util.showTree(AVA.f[4][0], WineDataSmall.words)

print "\n--------------------------------------------------\n"

# Pinot-Noir

print "Pinot-Noir OAA\n"

util.showTree(OAA.f[2], WineDataSmall.words)

print "\n--------------------------------------------------\n"
print "Pinot-Noir AVA"

print "\n0 vs 2"
util.showTree(AVA.f[2][0], WineDataSmall.words)
print "\n1 vs 2"
util.showTree(AVA.f[2][1], WineDataSmall.words)
print "\n3 vs 2"
util.showTree(AVA.f[3][2], WineDataSmall.words)
print "\n4 vs 2"
util.showTree(AVA.f[4][2], WineDataSmall.words)

print "\n--------------------------------------------------\n"

# ------
# PART B
# ------

OAA = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))
t0 = time.time()
OAA.train(WineData.X, WineData.Y)
t1 = time.time()
p = OAA.predictAll(WineData.Xte)

print "OAA training time: %f" % (t1 - t0)
print "OAA accuracy: %f" % mean(WineData.Yte == p)

print "\n--------------------------------------------------\n"

AVA = multiclass.AVA(20, lambda: DecisionTreeClassifier(max_depth=3))
t0 = time.time()
AVA.train(WineData.X, WineData.Y)
t1 = time.time()
p = AVA.predictAll(WineData.Xte)

print "AVA training time: %f" % (t1 - t0)
print "AVA accuracy: %f" % mean(WineData.Yte == p)

print "\n--------------------------------------------------\n"

print "Viognier OAA\n"
util.showTree(OAA.f[17], WineData.words)

print "\n--------------------------------------------------\n"

print "Viognier AVA"
for i in range(0, 17):
	print "\n17 vs %d" % i
	util.showTree(AVA.f[17][i], WineData.words)

print "\n--------------------------------------------------\n"

# ------
# PART C
# ------

p = OAA.predictAll(WineData.Xte, useZeroOne=True)
print "OAA accuracy using zero/one: %f" % mean(WineData.Yte == p)

p = AVA.predictAll(WineData.Xte, useZeroOne=True)
print "AVA accuracy using zero/one: %f" % mean(WineData.Yte == p)