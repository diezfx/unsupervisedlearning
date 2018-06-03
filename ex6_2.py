import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import numpy as np
import os
import scipy.sparse.linalg as sla
import random

import sklearn.mixture


def choose_random_x(X, K):
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

    mu_index = choose_random_x(X, K)
    for k in range(K):
        mu_inits[k] = X[mu_index[k]]

    return mu_inits


def calculate_em(X, K):
    identity = np.identity(2)
    Sigma = np.full((K, 2, 2), identity, dtype=np.float32)

    mix_value = np.full(K, 1 / K)
    mu = random_start_mu(X, K)
    mu = np.array(mu)


    for i in range(10):
        gamma = evaluate(X, K, mix_value, mu, Sigma)
        print(gamma)
        print("....................................")
        mu, mix_value, Sigma = update(X, K, mu, mix_value, gamma, Sigma)



    return calc_propability(X, K, gamma, mix_value, mu, Sigma)


def evaluate(X, K, mix_value, mu, Sigma):
    gamma = np.zeros((X.shape[0], K))



    for i in range(X.shape[0]):
        gamma_all = 0
        for j in range(K):
            gamma_all += mix_value[j] * stats.multivariate_normal.pdf(x=X[i], mean=mu[j], cov=Sigma[j],
                                                                      allow_singular=True)
        for k in range(K):
            gamma_upper = mix_value[k] * stats.multivariate_normal.pdf(x=X[i], mean=mu[k], cov=Sigma[k],
                                                                       allow_singular=True)
            gamma[i, k] = gamma_upper / gamma_all


    return gamma


def update(X, K, mu, mix_value, gamma, Sigma):
    for k in range(K):
        N_k = 0
        Sigma[k] = np.zeros((X.shape[1], X.shape[1]))
        mu[k]=0
        for i in range(X.shape[0]):
            N_k += gamma[i, k]
            mu[k] += gamma[i, k] * X[i]
        mu[k] = mu[k] / N_k
        for i in range(X.shape[0]):
            Sigma[k] += gamma[i, k] * np.dot((X[i] - mu[k]), (X[i] - mu[k]).T)
        mix_value[k] = N_k / X.shape[0]

        Sigma[k] = Sigma[k] / N_k

    return mu, mix_value, Sigma


def calc_propability(X, K, gamma, mix_value, mu, Sigma):
    p = 0
    for i in range(X.shape[0]):
        for k in range(K):
            first = mix_value[k] ** gamma[i, k]
            second = stats.multivariate_normal.pdf(X[i], mean=mu[k], cov=Sigma[k], allow_singular=True) ** gamma[i, k]
            p += first * second

    return p


K = 3
X = np.loadtxt("mixture.txt", dtype=np.float32)

print(calculate_em(X, K))

plt.scatter(X[:, 0:1], X[:, 1:2])
plt.show()
