"""
Test against independent library
"""
from optionprice import Option

benchmark_option = Option(european=True,
                          kind='put',
                          s0=1.28065,
                          k=1.28065,
                          t=10,
                          sigma=0.05,
                          r=0)

price = benchmark_option.getPrice(method='BT', iteration=10000)

print(price)
