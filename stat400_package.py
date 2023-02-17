class Math:
    def fact(n):
        ans = 1
        while n:
            ans *= n
            n -= 1
        return ans
    def C(n, r): return Math.fact(n) / (Math.fact(n - r) * Math.fact(r))

class NegativeBinominal:
    def __init__(self, r, p):
        self.r = r
        self.prob = p
    def mean(self): return self.r / self.prob
    def variance(self): return self.r * (1 - self.prob) / (self.prob ** 2)
    def pmf(self, x): return Math.C(x - 1, self.r - 1) * (self.prob ** self.r) * ((1 - self.prob) ** (x - self.r))
    def cdf(self, x): return sum(self.pmf(i) for i in range(self.r, x + 1))

class Binominal:
    def __init__(self, n, prob):
        self.n = n
        self.prob = prob
    def mean(self): return self.n * self.prob  
    def variance(self): return self.n * self.prob * (1 - self.prob)   
    def pmf(self, x): return Math.C(self.n, x) * (self.prob ** x) * ((1 - self.prob) ** (self.n - x))  
    def cdf(self, x): return sum(self.pmf(i) for i in range(x + 1))
