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


def create_array(max_value, count):
    generated_array = np.random.randint(max_value, size=(count), dtype="int32")
    return generated_array


def get_array_range(arrary_num):
    while True:
        try:
            user_input = int(input(f"Enter max value possible in the {arrary_num} array: "))
            break
        except ValueError:
            print("That's not a valid option!")

    return user_input


def cpu_adding(array):
    for i in range(len(array)):
        array[i] += 1


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


def multiply_matrix(array, array2):
    print(f"MatMul result: {np.matmul(array, array2)}")


def cupy_multiply_matrix(array, array2):
    cupy_array_one = cp.array(array)
    cupy_array_two = cp.array(array2)
    print(f"MatMul result: {cp.matmul(cupy_array_one, cupy_array_two)}")


def stat_analysis(array, array2):
    print(f"Standard Deviation: {np.std(array)}")
    print(f"Standard Deviation: {np.std(array2)}")
    print(f"Mean: {np.mean(array)}")
    print(f"Mean: {np.mean(array2)}")
    print(f"Median: {np.median(array)}")
    print(f"Median: {np.median(array2)}")
    print(f"Variance: {np.var(array)}")
    print(f"Variance: {np.var(array2)}")
    print(f"Correlation: {np.correlate(array, array2)}")


@jit
def jit_stat_analysis(array, array2):
    print("Standard Deviation: ", np.std(array))
    print("Standard Deviation: ", np.std(array2))
    print("Mean: ", np.mean(array))
    print("Mean: ", np.mean(array2))
    print("Median: ", np.median(array))
    print("Median: ", np.median(array2))
    print("Variance: ", np.var(array))
    print("Variance: ", np.var(array2))
    print("Correlation: ", np.correlate(array, array2))


@jit(nopython=True, parallel=True)
def jit_options_stat_analysis(array, array2):
    print("Standard Deviation: ", np.std(array))
    print("Standard Deviation: ", np.std(array2))
    print("Mean: ", np.mean(array))
    print("Mean: ", np.mean(array2))
    print("Median: ", np.median(array))
    print("Median: ", np.median(array2))
    print("Variance: ", np.var(array))
    print("Variance: ", np.var(array2))
    print("Correlation: ", np.correlate(array, array2))


def cupy_stat_analysis(array, array2):
    array = cp.array(array)
    array2 = cp.array(array2)
    print("Note: cupy does not have a Median function")
    print(f"Standard Deviation: {cp.std(array)}")
    print(f"Standard Deviation: {cp.std(array2)}")
    print(f"Mean: {cp.mean(array)}")
    print(f"Mean: {cp.mean(array2)}")
    print(f"Variance: {cp.var(array)}")
    print(f"Variance: {cp.var(array2)}")
    print(f"Correlation: {cp.corrcoef(array, array2)}")


def cpu_searching_linear(array, array2):
    for item in array:
        if item == 45678:
            pass

    for item in array2:
        if item == 78946:
            pass


@jit
def jit_searching_linear(array, array2):
    for item in array:
        if item == 45678:
            pass

    for item in array2:
        if item == 78946:
            pass


@jit(nopython=True, parallel=False)
def jit_options_searching_linear(array, array2):
    for item in array:
        if item == 45678:
            pass

    for item in array2:
        if item == 78946:
            pass


def compute_reciprocals(array):
    result_array = np.empty(len(array))
    for i in range(len(array)):
        if array[i] == 0:
            divide_by = 1
        else:
            divide_by = array[i]
        result_array[i] = 1.0 / divide_by

    return result_array


@jit
def jit_compute_reciprocals(array):
    result_array = np.empty(len(array))
    for i in range(len(array)):
        if array[i] == 0:
            divide_by = 1
        else:
            divide_by = array[i]
        result_array[i] = 1.0 / divide_by

@jit(nopython=True, parallel=False)
def jit_options_compute_reciprocals(array):
    result_array = np.empty(len(array))
    for i in range(len(array)):
        if array[i] == 0:
            divide_by = 1
        else:
            divide_by = array[i]
        result_array[i] = 1.0 / divide_by



def cupy_compute_reciprocals(array):
    array = cp.array(array)
    result_array = cp.empty(len(array))
    for i in range(len(array)):

        if array[i] == 0:
            divide_by = 1
        else:
            divide_by = array[i]
        result_array[i] = 1.0 / divide_by



def get_menu_options():
    print("1 add to each element of array, 2 array multiplication, "
          "3 statistical analysis , 4 check every element, 5 reciprocal, 6 print arrays. 99 to exit ")
    while True:
        try:
            user_input = int(input(f"What would you like to do ? "))
            # i rather like this way of comparison
            # instead of if user_input > 0 and user_input < 7
            if 0 < user_input < 7 or user_input == 99:
                break
            else:
                print("invalid input , out of range")
        except ValueError:
            print("That's not a valid option!")

    return user_input


def print_to_file(output_list):
    with open('program_timings.txt', 'w') as f:
        for line in output_list:
            f.write("%s\n" % line)


if __name__ == "__main__":
    # count = get_array_size()
    # max_number = get_array_range("first")
    # max_number_2 = get_array_range("second")
    ## for testing cos lazy
    count = 10
    max_number = 500
    max_number_2 = 750
    array_one = create_array(max_number, count)
    array_two = create_array(max_number_2, count)

    output_list = []
    while True:
        user_selection = get_menu_options()
        print(f"User Selection: {user_selection}")
        print()

        if user_selection == 99:
            break

        if user_selection == 1:

            start = timer()
            cpu_adding(array_one)
            print("cpu adding 1:", timer() - start)
            output_list.append(f"cpu adding 1: {timer() - start}")
            start = timer()
            cpu_adding(array_one)
            print("cpu adding 2:", timer() - start)
            output_list.append(f"cpu adding 2: {timer() - start}")

            print()

            start = timer()
            jit_adding(array_one)
            print("jit Lazy:", timer() - start)
            output_list.append(f"jit Lazy: {timer() - start}")

            start = timer()
            jit_adding(array_one)
            print("jit Lazy 2:", timer() - start)
            output_list.append(f"jit Lazy 2: {timer() - start}")

            print()

            start = timer()
            jit_options_adding(array_one)
            print("jit parallel:", timer() - start)
            output_list.append(f"jit parallel : {timer() - start}")

            start = timer()
            jit_options_adding(array_one)
            print("jit parallel 2:", timer() - start)
            output_list.append(f"jit parallel 2: {timer() - start}")

            print()

        elif user_selection == 2:

            start = timer()
            multiply_matrix(array_one, array_two)
            print("Multiply Matrix cpu : ", timer() - start)
            output_list.append(f"Multiply Matrix cpu: {timer() - start}")

            start = timer()
            multiply_matrix(array_one, array_two)
            print("Multiply Matrix cpu 2 : ", timer() - start)
            output_list.append(f"Multiply Matrix cpu 2: {timer() - start}")

            print()

            start = timer()
            cupy_multiply_matrix(array_one, array_two)
            print("Multiply Matrix cupy : ", timer() - start)
            output_list.append(f"Multiply Matrix cupy: {timer() - start}")

            start = timer()
            cupy_multiply_matrix(array_one, array_two)
            print("Multiply Matrix cupy 2 : ", timer() - start)
            output_list.append(f"Multiply Matrix cupy 2: {timer() - start}")

            print()

        elif user_selection == 3:

            start = timer()
            stat_analysis(array_one, array_two)
            print("Statistical Analysis cpu : ", timer() - start)
            output_list.append(f"Statistical Analysis cpu: {timer() - start}")

            start = timer()
            stat_analysis(array_one, array_two)
            print("Statistical Analysis cpu 2 : ", timer() - start)
            output_list.append(f"Statistical Analysis cpu 2: {timer() - start}")

            print()

            start = timer()
            jit_stat_analysis(array_one, array_two)
            print("Statistical Analysis jit : ", timer() - start)
            output_list.append(f"Statistical Analysis jit: {timer() - start}")

            start = timer()
            jit_stat_analysis(array_one, array_two)
            print("Statistical Analysis jit 2 : ", timer() - start)
            output_list.append(f"Statistical Analysis jit 2: {timer() - start}")

            print()

            start = timer()
            jit_options_stat_analysis(array_one, array_two)
            print("Statistical Analysis jit 2 : ", timer() - start)
            output_list.append(f"Statistical Analysis jit options: {timer() - start}")

            start = timer()
            jit_options_stat_analysis(array_one, array_two)
            print("Statistical Analysis jit 2 : ", timer() - start)
            output_list.append(f"Statistical Analysis jit options 2: {timer() - start}")

            print()

            start = timer()
            cupy_stat_analysis(array_one, array_two)
            print("Statistical Analysis cupy : ", timer() - start)
            output_list.append(f"Statistical Analysis cupy: {timer() - start}")

            start = timer()
            cupy_stat_analysis(array_one, array_two)
            print("Statistical Analysis cupy 2 : ", timer() - start)
            output_list.append(f"Statistical Analysis cupy 2: {timer() - start}")

            print()

        elif user_selection == 4:

            start = timer()
            cpu_searching_linear(array_one, array_two)
            print("Element comparison cpu : ", timer() - start)
            output_list.append(f"Element comparison cpu: {timer() - start}")

            start = timer()
            cpu_searching_linear(array_one, array_two)
            print("Element comparison cpu 2 : ", timer() - start)
            output_list.append(f"Element comparison cpu 2: {timer() - start}")

            print()

            start = timer()
            jit_searching_linear(array_one, array_two)
            print("Element comparison jit : ", timer() - start)
            output_list.append(f"Element comparison jit: {timer() - start}")

            start = timer()
            jit_searching_linear(array_one, array_two)
            print("Element comparison jit 2 : ", timer() - start)
            output_list.append(f"Element comparison jit 2: {timer() - start}")

            print()

            start = timer()
            jit_options_searching_linear(array_one, array_two)
            print("Element comparison jit : ", timer() - start)
            output_list.append(f"Element comparison jit options: {timer() - start}")

            start = timer()
            jit_options_searching_linear(array_one, array_two)
            print("Element comparison jit : ", timer() - start)
            output_list.append(f"Element comparison jit options 2: {timer() - start}")

            print()

        elif user_selection == 5:

            start = timer()
            compute_reciprocals(array_one)
            print("Compute reciprocals CPU :", timer() - start)
            output_list.append(f"Compute reciprocals CPU: {timer() - start}")

            start = timer()
            compute_reciprocals(array_two)
            print("Compute reciprocals CPU 2:", timer() - start)
            output_list.append(f"Compute reciprocals CPU 2: {timer() - start}")

            print()

            start = timer()
            jit_compute_reciprocals(array_one)
            print("Compute reciprocals jit:", timer() - start)
            output_list.append(f"Compute reciprocals jit: {timer() - start}")

            start = timer()
            jit_compute_reciprocals(array_two)
            print("Compute reciprocals jit 2:", timer() - start)
            output_list.append(f"Compute reciprocals jit 2: {timer() - start}")

            print()

            start = timer()
            jit_options_compute_reciprocals(array_one)
            print("Compute reciprocals jit:", timer() - start)
            output_list.append(f"Compute reciprocals jit options: {timer() - start}")

            start = timer()
            jit_options_compute_reciprocals(array_one)
            print("Compute reciprocals jit:", timer() - start)
            output_list.append(f"Compute reciprocals jit options 2: {timer() - start}")

            print()

            start = timer()
            cupy_compute_reciprocals(array_one)
            print("Compute reciprocals cupy:", timer() - start)
            output_list.append(f"Compute reciprocals cupy: {timer() - start}")

            start = timer()
            cupy_compute_reciprocals(array_two)
            print("Compute reciprocals cupy 2:", timer() - start)
            output_list.append(f"Compute reciprocals cupy 2: {timer() - start}")

            print()
        elif user_selection == 6:

            print(array_one)
            print(array_two)

    print_to_file(output_list)