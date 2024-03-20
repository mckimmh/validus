"""
Validus Case Study
March 2024
"""
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
        nu : asset price increase/decrease proportion
        """

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
    
S0 = 1.28065
K  = 1.28065
N = 10
my_option = Put(S0, K, N)
nu = 0.05

V = my_option.value(nu)

print(V)
