import knn
import datasets
import runClassifier

# different values for K
curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True}), 'K', [1,2,3,4,5,6,7,8,9,10], datasets.DigitData)
runClassifier.plotCurve('KNN on Digit Data (hyperparameter K)', curve)

# different values for epsilon
curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False}), 'eps', [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], datasets.DigitData)
runClassifier.plotCurve('KNN on Digit Data (hyperparameter epsilon)', curve)

# learning curve for K = 5
curve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.DigitData)
runClassifier.plotCurve('KNN on Digit Data', curve)