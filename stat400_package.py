import math
import scipy as sp

class NegativeBinominalDistribution:
    def __init__(self, r, p):
        self.r = r
        self.prob = p
    def mean(self): return self.r / self.prob
    def variance(self): return self.r * (1 - self.prob) / (self.prob ** 2)
    def pdf(self, x): return math.comb(x - 1, self.r - 1) * (self.prob ** self.r) * ((1 - self.prob) ** (x - self.r))
    def cdf(self, x): return sum(self.pdf(i) for i in range(self.r, x + 1))
    def mgf(self, t): return self.prob ** self.r / ((1 - (1 - self.prob) * math.exp(t)) ** self.r)

class BinominalDistribution:
    def __init__(self, n, prob):
        self.n = n
        self.prob = prob
    def mean(self): return self.n * self.prob  
    def variance(self): return self.n * self.prob * (1 - self.prob)   
    def pdf(self, x): return math.comb(self.n, x) * (self.prob ** x) * ((1 - self.prob) ** (self.n - x))  
    def cdf(self, x): return sum(self.pdf(i) for i in range(x + 1))
    def mgf(self, t): return (self.prob * math.exp(t) + (1 - self.prob)) ** self.n

class GeometricDistribution:
    def __init__(self, p):
        self.prob = p
    def mean(self): return 1 / self.prob
    def variance(self): return (1 - self.prob) / self.prob ** 2
    def pdf(self, x): return self.prob * ((1 - self.prob) ** (x - 1))
    def cdf(self, x): return sum(self.pdf(i) for i in range(x + 1))
    def mgf(self, t): return self.prob / (1 - (1 - self.prob) * math.exp(t))

class HypergeometricDistribution:
    def __init__(self, N, N1, n):
        self.N = N
        self.N1 = N1
        self.N2 = self.N - self.N1
        self.n = n
    def mean(self): return self.n * self.N1 / self.N
    def variance(self): return self.n * self.N1 / self.N * self.N2 / self.N * (self.N - self.n) / (self.N - 1)
    def pdf(self, x): return math.comb(self.N1, x) * math.comb(self.N2, self.n - x) / math.comb(self.N, self.n)
    def cdf(self, x): return sum(self.pdf(i) for i in range(x + 1))

class MultinominalDistribution:
    def __init__(self, n, probs):
        self.n = n
        self.probs = probs
        self.k = len(self.probs)
    def mean(self, idx): return self.n * self.probs[idx - 1]
    def variance(self, idx): return self.n * self.probs[idx - 1] * (1 - self.probs[idx - 1])
    def pdf(self, xs): return math.factorial(self.n) / math.prod(math.factorial(x) for x in xs) * math.prod(self.probs[i] ** xs[i] for i in range(self.k))

class PoissonDistribution:
    def __init__(self, lam):
        self.lam = lam
    def mean(self): return self.lam
    def variance(self): return self.lam
    def pdf(self, x): return (self.lam ** x) * math.exp(-self.lam) / math.factorial(x)
    def cdf(self, x): return sum(self.pdf(i) for i in range(x + 1))
    def mgf(self, t): return math.exp(self.lam * (math.exp(t) - 1))

class NormalDistribution:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
    def mean(self): return self.mu
    def variance(self): return self.var
    def pdf(self, x): return (1 / math.sqrt(2 * math.pi * self.var)) * math.exp(- ((x - self.mu) ** 2) / (2 * self.var))
    def cdf(self, z): return round(sp.integrate.quad(lambda w : (1 / math.sqrt(2 * math.pi)) * math.exp(-((w) ** 2) / 2), float('-inf'), z)[0], 4)

class ExponentialDistribution:
    def __init__(self, theta):
        self.theta = theta
        self.lam = 1 / theta
    def mean(self): return self.theta
    def variance(self): return self.theta ** 2
    def pdf(self, x): return self.lam * math.exp(-x * self.lam)
    def cdf(self, w): return 1 - math.exp(-self.lam * w)

class GammaDistribution:
    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta
    def GammaFunction(t): return math.factorial(t-1)
    def mean(self): return self.alpha * self.theta
    def variance(self): return self.alpha * self.theta ** 2
    def pdf(self, x): return 1 / (GammaDistribution.GammaFunction(self.alpha) * self.theta**self.alpha) * x**(self.alpha - 1) * math.exp(-x / self.theta)

class UniformDistribution:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def mean(self): return (self.a + self.b) / 2
    def variance(self): return (self.b - self.a)**2 / 12
    def pdf(self): return 1 / (self.b - self.a)
    def cdf(self, x): return (x - self.a) / (self.b - self.a)
