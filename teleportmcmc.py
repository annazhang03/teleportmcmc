# Anna Zhang

import matplotlib.pyplot as plt
import random  
import numpy as np

np.random.seed(32); random.seed(17)

D = 3 # dimension
N = 10 # number of walkers
T1 = 1000 # number of burn-in steps
T = 20000 # number of time steps

'''-----------------------------------
                density
-----------------------------------'''
# first parameter double well, other parameters gaussian
beta = 2
a1 = 0
def f(x):
    return np.exp(-(beta * (x[0]**4 - 2*x[0]**2 + a1*x[0]) + 0.5 * np.linalg.norm(x[1:])**2))
    
'''-----------------------------------
                proposal
-----------------------------------'''
# gaussian proposal
epsilon = 0.1 # variance of gaussian

# returns probability of x given y
def g(x,y):
    n = np.linalg.norm(x - y) ** 2 / epsilon
    return np.exp(-0.5 * n)

# returns proposal given x based on g
# gaussian centered at x with variance epsilon
def propose(x):
    return np.random.normal(x, np.sqrt(epsilon), D)

'''-----------------------------------
                method
-----------------------------------'''
# sum g(x|X[k]) for all k != i to use in main()
def S(X, i, x):
    s = 0
    for k in range(N):
        if k != i:
            s += g(x, X[k])
    return s

# returns T x D array of samples of f
# plots marginal of first parameter 
# prints progress every 5000 steps
# prints final acceptance & teleport probabilities
def main():
    j = 0
    X = np.random.randn(N,D) # N x D array of walkers
    # initialize walkers to be Normal(0,1)
    for n in range(N):
        while ( f(X[n]) == 0 
                or np.isnan(f(X[n])) 
                or np.isinf(f(X[n])) ):
            X[n] = np.random.randn(D)

    samples = np.zeros((T,D)) # T x D array of samples
    aProb = 0.0 # acceptance probability
    tProb = 0.0 # teleport probability
    C = np.zeros(N) # C[j] = the sum of g(x_j | x_k) for all k != j
    F = np.zeros(N) # F[j] = f(X[j])
    for l in range(N):
        C[l] = S(X, l, X[l])
        F[l] = f(X[l])
    
    for t in range(T1 + T):
        if t % 5000 == 0 and t > T1:
            print("N = " + str(N) + ", t = " + str(t))
            print("teleport: " + str(tProb / (t - T1 + 1)),)
            print("accept: " + str(aProb / (t - T1 + 1)))
        
        # clone walker j using given proposal
        j = (j + 1) % N
        z = propose(X[j])

        # get index i using importance weights
        W = np.zeros(N)
        inf = False
        Z = 0
        for l in range(N):
            W[l] = C[l]
            W[l] += g(X[l], z)
            W[l] /= F[l]
            if not np.isnan(W[l]) and not np.isinf(W[l]):
                Z += W[l]
            else:
                inf = True
        W /= Z
        if inf:
            for l in range(N):
                W[l] = 1 if np.isnan(W[l]) or np.isinf(W[l]) else 0
        r = random.random()
        done = False
        i = 0
        p = W[i]
        while not done:
            if r > p:
                if i < N - 1:
                    i += 1
                else:
                    done = True
                p += W[i]
            else:
                done = True
        if t >= T1 and i != j:
            tProb += 1
        
        # accept/reject proposal
        f_z = f(z)
        Z1 = Z
        Z1 -= (g(X[i], z) + S(X, i, X[i])) / F[i]
        Z1 += (g(z, X[i]) + S(X, i, z)) / f_z
        if random.random() < Z / Z1: 
            Xcopy = np.copy(X)
            X[i] = z
            for l in range(N):
                if l != i:
                    C[l] -= g(X[l], Xcopy[i])
                    C[l] += g(X[l], X[i])
                else:
                    C[l] = S(X, l, X[l])
                    F[l] = f_z
            if t >= T1:
                aProb += 1
        if t >= T1:
            samples[t-T1] = X[j]

    # print results & plot marginal of first parameter
    aProb /= T
    tProb /= T

    print("final teleport: " + str(tProb))
    print("final accept: " + str(aProb))

    bins = 400
    title = "N = " + str(N) + ", teleport = " + str(tProb) + ", accept = " + str(aProb)
    plt.title(title)
    plt.xlabel('Parameter 1')
    histogram = plt.hist(samples[:,0], bins, density=True)[0]
    plt.show()

    return samples

main()