# https://realpython.com/numpy-array-programming/#array-programming-in-action-examples

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numba import jit, cuda, prange
from numba import vectorize, cuda

# tri = np.array([[1, 1],
#                 [3, 1],
#                 [2, 3]])
#
# centroid = tri.mean(axis=0)
# print(centroid)
#
# trishape = plt.Polygon(tri, edgecolor='r', alpha=0.2, lw=5)
# _, ax = plt.subplots(figsize=(4, 4))
# ax.add_patch(trishape)
# ax.set_ylim([.5, 3.5])
# ax.set_xlim([.5, 3.5])
# ax.scatter(*centroid, color='g', marker='D', s=70)
# ax.scatter(*tri.T, color='b',  s=70)
#
#
# np.sum(tri**2, axis=1) ** 0.5  # Or: np.sqrt(np.sum(np.square(tri), 1))
# np.linalg.norm(tri, axis=1)
# np.linalg.norm(tri - centroid, axis=1)


freq = 12  # 12 months per year
rate = .0675  # 6.75% annualized
nper = 30  # 30 years
pv = 200000  # Loan face value

print(type(freq))
print(type(rate))
print(type(nper))
print(type(pv))
rate /= freq  # Monthly basis
nper *= freq  # 360 months

periods = np.arange(1, nper + 1, dtype=int)
principal = np.ppmt(rate, periods, nper, pv)
interest = np.ipmt(rate, periods, nper, pv)
pmt = principal + interest  # Or: pmt = np.pmt(rate, nper, pv)


def balance(pv, rate, nper, pmt):
    d = (1 + rate) ** nper  # Discount factor
    return pv * d - pmt * (d - 1) / rate


import pandas as pd

start = timer()
cols = ['beg_bal', 'prin', 'interest', 'end_bal']
data = [balance(pv, rate, periods - 1, -pmt),
        principal,
        interest,
        balance(pv, rate, periods, -pmt)]

table = pd.DataFrame(data, columns=periods, index=cols).T
table.index.name = 'month'

with pd.option_context('display.max_rows', 6):
    # Note: Using floats for $$ in production-level code = bad
    print(table.round(2))

final_month = periods[-1]
np.allclose(table.loc[final_month, 'end_bal'], 0)

print("without GPU:", timer() - start)
