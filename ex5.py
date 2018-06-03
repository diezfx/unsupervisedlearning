import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os
import scipy.sparse.linalg as sla


def computeMeanFace(X):
    meanface = np.zeros(77760, dtype=np.float64)

    for i in range(77760):
        x = np.sum(X[:, i:i + 1]) / 165
        meanface[i] = x

    return meanface


num_eigenvalues = 60

# a)
X = np.zeros((165, 77760), dtype=np.float64)
indir = 'yalefaces'
i = 0
for root, dirs, filenames in os.walk(indir):

    for f in filenames:
        if not f.endswith('.txt'):
            x = plt.imread(indir + '/' + f)
            x = x.reshape(77760)
            X[i] = x
            i = i + 1

print(X)
# b)
mu = (computeMeanFace(X))

X_centered = X - np.tile(mu.T, (165, 1))
# c)

for neigenvalues in [1, 5, 10, 20, 40, 60]:
    u, s, vt = sla.svds(X, k=neigenvalues, which='LM')

    # d)
    vp = vt.T[:, :neigenvalues]
    Z = np.dot(X_centered, vp)

    # e)
    X_reconstructed = np.tile(mu.T, (165, 1)) + np.dot(Z, vp.T)

    error = 0
    for i in range(X.shape[0]):
        error = error + np.linalg.norm(X[i] - X_reconstructed[i]) ** 2
    print("the error is: ", error)

    print(np.dot(vp, s).reshape((243, 320)))
    # plt.figure(1)
    # plt.subplot(111)
    plt.imshow(X_reconstructed[5].reshape((243, 320)), cmap=plt.cm.gray)

    eigenface = mu.T + np.dot(s, vp.T)
    # plt.imshow(eigenface.reshape((243, 320)), cmap=plt.cm.gray_r)

    plt.show()
