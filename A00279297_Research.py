

# this file is for code i am actually goiung to include in final submission
from datetime import time

import numpy as np
from numba import vectorize, cuda
from numba import jit, cuda, prange
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# vectorize combining two data frames
@vectorize(['float32(float32, float32)'], target='cuda')
def Add(array_one, array_two):
  return array_one + array_two

def do_vectorise():
    # Initialize arrays
    Number_of_values = 100000000
    first_array = np.ones(Number_of_values, dtype=np.float32)
    second_array = np.ones(first_array.shape, dtype=first_array.dtype)
    result_array = np.empty_like(first_array, dtype=first_array.dtype)


    # Add arrays on GPU
    start = time.time()
    result_array = Add(first_array, second_array)
    end = time.time()
    print("Elapsed (after compilation) Vector = %s" % (end - start))





############ good stuff use this
# #################
# from numba import jit, cuda, prange
# import numpy as np
# # to measure exec time
# from timeit import default_timer as timer
#
#
# normal function to run on cpu
def cpu_adding(array,number_of_values):
    for i in range(number_of_values):
        array[i] += 1
    for item in array:
        if item == 45678:
            print("45678 found")

    for item in array:
        if item == 78946:
            print("78946 found")

#

number_of_values = 10000000
array_of_ones = np.ones(number_of_values, dtype=np.float64)

start = timer()
cpu_adding(array_of_ones,number_of_values)
print("without GPU:", timer() - start)

start = timer()
cpu_adding(array_of_ones,number_of_values)
print("without GPU:", timer() - start)

print()
# function optimized to run on gpu


@jit
def jit_adding(array,number_of_values):
    for i in range(number_of_values):
        array[i] += 1
    for item in array:
        if item == 45678:
            print("hello")

    for item in array:
        if item == 78946:
            print("hello")

start = timer()
jit_adding(array_of_ones,number_of_values)
print("jit Lazy:", timer() - start)

start = timer()
jit_adding(array_of_ones,number_of_values)
print("jit Lazy:", timer() - start)

print()


@jit(nopython=True, parallel=True)
def jit_options_adding(array,number_of_values):
    for i in prange(number_of_values):
        array[i] += 1
    for item in array:
        if item == 45678:
            print("hello")

    for item in array:
        if item == 78946:
            print("hello")

start = timer()
jit_options_adding(array_of_ones,number_of_values)
print("jit parallel:", timer() - start)

start = timer()
jit_options_adding(array_of_ones,number_of_values)
print("jit parallel:", timer() - start)





# @cuda.jit
# @vectorize(['float64(float64, float64)'], target='cuda')

# @cuda.jit
# def cuda_adding(array,number_of_values):
#     # print("blah")
#     for i in range(number_of_values):
#         array[i] += 1
#     for item in array:
#         if item == 45678:
#             # print("hello")
#             continue
#
#     for item in array:
#         if item == 78946:
#             # print("hello")
#             continue
#
#
# start = timer()
# cuda_adding(array_of_ones,number_of_values)
# print("jit options:", timer() - start)
#
# start = timer()
# cuda_adding(array_of_ones,number_of_values)
# print("jit options:", timer() - start)


@jit(nopython=True, parallel=True)
def logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
    return w


# if __name__ == "__main__":
#     n = 10000000
#     a = np.ones(n, dtype=np.float64)
#     b = np.ones(n, dtype=np.float32)
#
#     start = timer()
#     func(a)
#     print("without GPU:", timer() - start)
#
#     start = timer()
#     func2(a)
#     print("with GPU:", timer() - start)
#
#     # with threads, functions stays in cache
#     start = timer()
#     func3(a)
#     print("with GPU threading:", timer() - start)
#
#     start = timer()
#     func3(a)
#     print("with GPU threading2:", timer() - start)
#
#     start = timer()
#     func3(a)
#     print("with GPU threading2:", timer() - start)
#



    # @jit(nopython=True, parallel=True)
    # def simulator(out):
    #     # iterate loop in parallel
    #     for i in prange(out.shape[0]):
    #         out[i] = run_sim()
##############################################