import numpy as np
from numba import jit, cuda, prange
# to measure exec time
from timeit import default_timer as timer
import cupy as cp

def get_array_size():
    while True:
        try:
            user_input = int(input("Enter size of arrays you want to create: "))
            break
        except:
            print("That's not a valid option!")
    return user_input


def create_array(max_value,count):
    generated_array = np.random.randint(max_value, size=(count), dtype="int32")
    return generated_array


def get_array_range(arraryNum):
    while True:
        try:
            user_input = int(input(f"Enter max value possible in the {arraryNum} array: "))
            break
        except:
            print("That's not a valid option!")

    return user_input


def getMenuOption():
    while True:
        try:
            user_input = int(input(f"What would you like to do ? "))
            break
        except:
            print("That's not a valid option!")

    return user_input


def cpu_adding(array):
    for i in range(len(array)):
        array[i] += 1
    for item in array:
        if item == 45678:
            pass

    for item in array:
        if item == 78946:
            pass

@jit
def jit_adding(array):
    for i in range(len(array)):
        array[i] += 1
    for item in array:
        if item == 45678:
            pass

    for item in array:
        if item == 78946:
            pass


@jit(nopython=True, parallel=True)
def jit_options_adding(array):
    for i in prange(len(array)):
        array[i] += 1
    for item in array:
        if item == 45678:
            pass

    for item in array:
        if item == 78946:
            pass

def do_stuff(array,array2):
    print(np.std(array))
    print(np.mean(array))
    print(np.median(array))
    print(np.var(array))
    # print(np.quantile(array,1))
    print(np.correlate(array,array2))

@jit
def do_stuff_jit(array,array2):
    print(np.std(array))
    print(np.mean(array))
    print(np.median(array))
    print(np.var(array))
    # print(np.quantile(array,1))
    print(np.correlate(array,array2))


def multiply_matrix(array,array2):
    print(np.matmul(array,array2))


def cupy_multiply_matrix(array,array2):

    cupy_array_one = cp.array(array)
    cupy_array_two = cp.array(array2)
    print(cp.matmul(cupy_array_one,cupy_array_two))

if __name__ == "__main__":
    count = get_array_size()
    max_number = get_array_range("first")
    max_number_2 = get_array_range("second")
    array_one = create_array(max_number,count)
    array_two = create_array(max_number_2,count)

    # print(array_one)
    # print(array_two)
    # print("////////////////////////////////////////////////////")
    # print("Add one and check each element twice")
    # start = timer()
    # cpu_adding(array_one)
    # print("without GPU:", timer() - start)
    #
    # start = timer()
    # cpu_adding(array_one)
    # print("without GPU:", timer() - start)
    #
    # print()
    #
    # start = timer()
    # jit_adding(array_one)
    # print("jit Lazy:", timer() - start)
    #
    # start = timer()
    # jit_adding(array_one)
    # print("jit Lazy:", timer() - start)
    #
    # print()
    #
    # start = timer()
    # jit_options_adding(array_one)
    # print("jit parallel:", timer() - start)
    #
    # start = timer()
    # jit_options_adding(array_one)
    # print("jit parallel:", timer() - start)
    # print("////////////////////////////////////////////////////")
    #
    # print("do stuff analysis")
    #
    # start = timer()
    # do_stuff(array_one,array_two)
    # print("do stuff cpu : ", timer() - start)
    #
    # start = timer()
    # do_stuff_jit(array_one,array_two)
    # print("do stuff gpu : ", timer() - start)
    #
    # print("/////////////////////////////////////////")


    start = timer()
    multiply_matrix(array_one,array_two)
    print("multiply_matrix cpu : ", timer() - start)

    start = timer()
    multiply_matrix(array_one,array_two)
    print("multiply_matrix cpu2 : ", timer() - start)

    start = timer()
    cupy_multiply_matrix(array_one,array_two)
    print("multiply_matrix gpu : ", timer() - start)

    start = timer()
    cupy_multiply_matrix(array_one,array_two)
    print("multiply_matrix gpu2 : ", timer() - start)