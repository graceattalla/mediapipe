'''
Simple test of parallelization to ensure it is working.

Written by Grace Attalla
'''

from joblib import Parallel, delayed
from math import sqrt

# print([sqrt(i ** 2) for i in range(10)])
# Parallel(n_jobs=2, verbose=10)(delayed(sqrt)(i ** 2) for i in range(10))


def square_root(x):
    return sqrt(x)

def calculate_square_root():
    results = Parallel(n_jobs=2)(delayed(square_root)(i ** 2) for i in range(10))
    return results

result_list = calculate_square_root()
print(result_list)

        
# square_root()