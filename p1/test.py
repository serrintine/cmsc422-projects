import runClassifier, datasets, binary, util, dumbClassifiers

h = dumbClassifiers.AlwaysPredictMostFrequent({})
runClassifier.trainTestSet(h, datasets.TennisData)
h
