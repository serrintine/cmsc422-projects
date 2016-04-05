import perceptron
import datasets
import runClassifier

# learning curve for epoch = 5
curve = runClassifier.learningCurveSet(perceptron.Perceptron({'numEpoch': 5}), datasets.SentimentData)
runClassifier.plotCurve('Perceptron on Sentiment Data', curve)

# different values for epoch
curve = runClassifier.hyperparamCurveSet(perceptron.Perceptron({}), 'numEpoch', [1,2,3,4,5,6,7,8,9,10], datasets.SentimentData)
runClassifier.plotCurve('Perceptron on Sentiment Data (hyperparameter)', curve)