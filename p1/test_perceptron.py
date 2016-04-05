import runClassifier, datasets, binary, util, dumbClassifiers, perceptron

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData)
print 'Expected: Training accuracy 0.642857, test accuracy 0.666667'

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)
print 'Expected: Training accuracy 0.857143, test accuracy 1'

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
print 'Expected: Training accuracy 0.835833, test accuracy 0.755'

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)
print 'Expected: Training accuracy 0.955, test accuracy 0.7975'
