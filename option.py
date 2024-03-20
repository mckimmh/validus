"""
Validus Case Study
March 2024
"""

class Put:
    """
    European put option.
    Assumes u=1+nu, d=1-nu; P(u)=0.5, P(d)=0.5.
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
        

