import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os
import scipy.sparse.linalg as sla
import random


### pca##
def computeMeanFace(X):
    meanface = np.zeros(X.shape[1], dtype=np.float64)

    for i in range(X.shape[1]):
        x = np.sum(X[:, i:i + 1]) / X.shape[0]
        meanface[i] = x

    return meanface


def compute_pca(X, n_eigenvalues):
    meanface = (computeMeanFace(X))

    X_centered = X - np.tile(meanface.T, (X.shape[0], 1))
    u, s, vt = sla.svds(X, k=n_eigenvalues, which='LM')
    vp = vt.T[:, :n_eigenvalues]
    Z = np.dot(X_centered, vp)

    return Z, meanface


# clustering
# dont choose the same X twice
def choose_random_X(X, K):
    mu_index = []
    k = 0
    while k is not K:
        init_i = random.randint(0, X.shape[0] - 1)
        if init_i not in mu_index:
            mu_index.append(init_i)
            k += 1

    return mu_index


# initialize the mu
def random_start_mu(X, K):
    mu_inits = np.zeros((K, X.shape[1]), dtype=np.float32)

    mu_index = choose_random_X(X, K)
    for k in range(K):
        mu_inits[k] = X[mu_index[k]]

    return mu_inits


def recalculate_cluster(X, r, K):
    new_mu = np.zeros((K, X.shape[1]), dtype=np.float32)
    new_mu_count = np.zeros(K, dtype=np.float32)

    for k in range(K):

        for n in range(X.shape[0]):
            if r[n, k] == 1:
                new_mu[k] += X[n]
                new_mu_count[k] += 1

        new_mu[k] = new_mu[k] / new_mu_count[k]

    return new_mu


def assign_cluster(X, K, mu):
    r = np.zeros((pic_count, K))

    for n in range(X.shape[0]):
        # return the index from the best fitting k
        min_k = (np.argmin([np.linalg.norm(X[n] - mu[k]) ** 2 for k in range(K)]))

        # set the best k to 1
        r[n, min_k] = 1
    return r


def calculate_error(X, mu, r, K):
    error = 0
    for n in range(X.shape[0]):
        for k in range(K):
            error = error + r[n, k] * np.linalg.norm(X[n] - mu[k]) ** 2

    return error


def compute_k_means_clustering(X, K):
    r = np.zeros((X.shape[0], K))

    mu = random_start_mu(X, K)

    error = float("inf")
    converged = False

    while (not converged):
        r = assign_cluster(X, K, mu)
        mu = recalculate_cluster(X, r, K)

        error_old = error
        error = calculate_error(X, mu, r, K)
        if error_old <= error or np.isnan(error):
            converged = True

    return mu, error, r


def choose_best_try(X, I, K):
    error_best = float("inf")
    for i in range(I):
        mu, error, r = compute_k_means_clustering(X, K)
        if error_best > error:
            mu_best = mu
            error_best = error
            r_best = r

    return mu_best, error_best, r_best


def choose_sample_pictures(K, r):
    samples = []

    for k in range(K):
        samples.append([])
        while len(samples[k]) < 4:
            rand = random.randint(0, X.shape[0] - 1)
            if r[rand, k] == 1 and rand not in samples[k]:
                samples[k].append(rand)

    return samples


pic_count = 136

# clustercount



K = 20
plot_sizeX = 2
plot_sizeY = 2

X = np.zeros((pic_count, 155520), dtype=np.float32)
indir = 'yalefaces_cropBackground'
i = 0
for root, dirs, filenames in os.walk(indir):

    for f in filenames:
        if not f.endswith('.txt'):
            x = plt.imread(indir + '/' + f)
            x = x.reshape(155520)
            X[i] = x
            i = i + 1
""""
for k in range(4,K):
# print(X)
    mu, error,r =choose_best_try(X,10,k)
    print("the error for ", k, " is ", error)

    if(k in [4,10,15,20]):


        samples=choose_sample_pictures(k,r)

        # print(mu)
        for a in range(k):
            f, axarr = plt.subplots(plot_sizeX, plot_sizeY)
            for i in range (2):
                for j in range(2):
                    axarr[i, j].imshow(X[samples[a][i*2+j]].reshape((243, 160, 4)) / 255.0)

            plt.show()

"""""

### c)

K = 4
n_eigenvalues = 20

Z, bla = compute_pca(X, n_eigenvalues)

mu, error, r = choose_best_try(X, 10, K)
print(error)

mu, error, r = choose_best_try(Z, 10, K)

print(error)
samples = choose_sample_pictures(K, r)

# print(mu)
for a in range(K):
    f, axarr = plt.subplots(plot_sizeX, plot_sizeY)
    for i in range(2):
        for j in range(2):
            axarr[i, j].imshow(X[samples[a][i * 2 + j]].reshape((243, 160, 4)) / 255.0)

    plt.show()
