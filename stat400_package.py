import math
import numpy as np
import scipy as sp

class Math:
    def fact(n): return math.prod(i for i in range(1, n + 1))
    def C(n, r): return Math.fact(n) / (Math.fact(n - r) * Math.fact(r))

class NegativeBinominal:
    def __init__(self, r, p):
        self.r = r
        self.prob = p
    def mean(self): return self.r / self.prob
    def variance(self): return self.r * (1 - self.prob) / (self.prob ** 2)
    def pmf(self, x): return Math.C(x - 1, self.r - 1) * (self.prob ** self.r) * ((1 - self.prob) ** (x - self.r))
    def cdf(self, x): return sum(self.pmf(i) for i in range(self.r, x + 1))
    def mgf(self, t): return self.prob ** self.r / ((1 - (1 - self.prob) * math.exp(t)) ** self.r)

class Binominal:
    def __init__(self, n, prob):
        self.n = n
        self.prob = prob
    def mean(self): return self.n * self.prob  
    def variance(self): return self.n * self.prob * (1 - self.prob)   
    def pmf(self, x): return Math.C(self.n, x) * (self.prob ** x) * ((1 - self.prob) ** (self.n - x))  
    def cdf(self, x): return sum(self.pmf(i) for i in range(x + 1))
    def mgf(self, t): return (self.prob * math.exp(t) + (1 - self.prob)) ** self.n

class Geometric:
    def __init__(self, p):
        self.prob = p
    def mean(self): return 1 / self.prob
    def variance(self): return (1 - self.prob) / self.prob ** 2
    def pmf(self, x): return self.prob * ((1 - self.prob) ** (x - 1))
    def cdf(self, x): return sum(self.pmf(i) for i in range(x + 1))
    def mgf(self, t): return self.prob / (1 - (1 - self.prob) * math.exp(t))

class Hypergeometric:
    def __init__(self, N, N1, n):
        self.N = N
        self.N1 = N1
        self.N2 = self.N - self.N1
        self.n = n
    def mean(self): return self.n * self.N1 / self.N
    def variance(self): return self.n * self.N1 / self.N * self.N2 / self.N * (self.N - self.n) / (self.N - 1)
    def pmf(self, x): return Math.C(self.N1, x) * Math.C(self.N2, self.n - x) / Math.C(self.N, self.n)
    def cdf(self, x): return sum(self.pmf(i) for i in range(x + 1))

class Multinominal:
    def __init__(self, n, probs):
        self.n = n
        self.probs = probs
        self.k = len(self.probs)
    def mean(self, idx): return self.n * self.probs[idx - 1]
    def variance(self, idx): return self.n * self.probs[idx - 1] * (1 - self.probs[idx - 1])
    def pmf(self, xs): return Math.fact(self.n) / math.prod(Math.fact(x) for x in xs) * math.prod(self.probs[i] ** xs[i] for i in range(self.k))

class Poisson:
    def __init__(self, lam):
        self.lam = lam
    def mean(self): return self.lam
    def variance(self): return self.lam
    def pmf(self, x): return (self.lam ** x) * math.exp(-self.lam) / Math.fact(x)
    def cdf(self, x): return sum(self.pmf(i) for i in range(x + 1))
    def mgf(self, t): return math.exp(self.lam * (math.exp(t) - 1))

class Normal:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
    def mean(self): return self.mu
    def variance(self): return self.var
    def pmf(self, x): return (1 / math.sqrt(2 * math.pi * self.var)) * math.exp(- ((x - self.mu) ** 2) / (2 * self.var))
    def cdf(self, z): return round(sp.integrate.quad(lambda w : (1 / math.sqrt(2 * math.pi)) * math.exp(-((w) ** 2) / 2), float('-inf'), z)[0], 4)
