from numpy import *
from matplotlib.pyplot import *
import util
import dr
import datasets

(X, Y) = datasets.loadDigits()
(P, Z, evals) = dr.pca(X, 784)

normalized_evals = evals/sum(evals)
eval_index = range(0, len(normalized_evals))

plot(eval_index, normalized_evals)
ylabel('eigenvalue')
xlabel('eigenvalue index')
title('Normalized Eigenvalues')
savefig('normalized_eigenvalues.png')

cumsum = cumsum(normalized_evals)
print np.argmax(cumsum > 0.9)
print np.argmax(cumsum > 0.95)

util.drawDigits(Z.T[:50,:], arange(50))
savefig('eigenvectors.png')