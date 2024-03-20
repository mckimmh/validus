"""
Validus Case Study
March 2024
"""
import matplotlib.pyplot as plt
import numpy as np

class Put:
    """
    European put option.
    Assumes u=1+nu, d=1-nu; P(u)=0.5, P(d)=0.5; r=0.
    """
    
    def __init__(self, S0, K, N):
        """
        S0 : initial price of asset
        K  : strike price
        N  : number of periods
        """
        
        # Checks on arguments
        if not isinstance(S0, float):
            raise TypeError("S0 not a float")
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if not isinstance(K, float):
            raise TypeError("K not a float")
        if K <= 0:
            raise ValueError("K must be positive")
        if type(N) != int:
            raise TypeError("N not an integer")
        if N <= 0:
            raise ValueError("N must be positive")

        self.S0 = S0
        self.K  = K
        self.N  = N

    def value(self, nu):
        """
        Compute the value of the option.

        Used:
        https://quantpy.com.au/binomial-tree-model/intro-to-binomial-trees/

        nu : asset price increase/decrease proportion
        """
        # Checks on nu
        if not isinstance(nu, float):
            raise TypeError("nu not an instance of float")
        if nu < 0.0 or nu >= 1.0:
            raise ValueError("nu < 0.0 or nu >= 1.0")

        # Asset prices at maturity:
        d_vec = (1-nu)**(np.arange(self.N, -1, -1))
        u_vec = (1+nu)**(np.arange(0, self.N+1))
        S = self.S0 * d_vec * u_vec

        # Option values at maturity
        V = np.maximum(self.K - S, np.zeros(self.N+1))

        # Step backwards through the tree
        for i in range(self.N, 0, -1):
            V = 0.5 * V[1:(i+1)] + 0.5 * V[0:i]

        return V[0]
    
    def calibrate(self, V):
        """
        Given the strike and value of the option, calibrate nu
        """

        # Checks on V
        if not isinstance(V, float):
            raise TypeError("V is not an instance of float")
        
        nu_lo = 0.0
        nu_hi = 0.999
        V_lo = self.value(nu_lo)
        V_hi = self.value(nu_hi)

        if V > V_hi or V < V_lo:
            raise ValueError("V outside the range of possible option values")
        
        epsilon = 0.0001
        while V_hi - V_lo > epsilon:

            nu_mid = (nu_hi + nu_lo) / 2.0
            V_mid = self.value(nu_mid)
            if V < V_mid:
                nu_hi = nu_mid
                V_hi = V_mid
            else:
                nu_lo = nu_mid
                V_lo = V_mid

        return (nu_hi + nu_lo) / 2.0
    
    def expected_max(self, nu):
        """
        The expectation of the maximum stock price, over the N periods
        """
        combos = 2**self.N
        paths = np.zeros((2**self.N, self.N+1))
        paths[:,0] = self.S0

        subgroup_size = combos
        for j in range(1, self.N+1):
            subgroup_size /= 2
            i = 0
            prop_change = nu
            while i < combos:
                paths[i,j] = (1 + prop_change) * paths[i,j-1]
                i += 1
                if i % subgroup_size == 0:
                    prop_change *= -1

        path_max = paths.max(axis=1)
            
        return path_max.mean()

######
# Test
######
    
S0 = 1.28065
K  = 1.28065
N = 3
my_option = Put(S0, K, N)

nu_single=0.05
V0 = my_option.value(nu_single)
print(V0)

nu0 = my_option.calibrate(V0)
print(nu0)

"""
nu = np.linspace(0.001, 0.99, 100)
V = np.zeros(nu.shape)
for i, value in enumerate(nu):
    V[i] = my_option.value(value)
plt.plot(nu, V)
plt.show()
"""

paths = my_option.expected_max(nu_single)

print(paths)