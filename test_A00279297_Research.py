# from unittest import TestCase

import pytest
import A00279297_Research

from A00279297_Research import *

import numpy as np


# import A00279297_Research

class TestClass:

    def test_get_array_size(self):
        # self.fail()
        A00279297_Research.input = lambda valid: 1
        output = A00279297_Research.get_array_size()

        assert output == 1

        A00279297_Research.input = lambda valid: 10
        output = A00279297_Research.get_array_size()

        assert output == 10

    def test_get_array_range(self):
        A00279297_Research.input = lambda valid: 1
        output = A00279297_Research.get_array_range("first")

        assert output == 1

        A00279297_Research.input = lambda valid: 10
        output = A00279297_Research.get_array_range("second")

        assert output == 10

    def test_create_array(self):
        output = create_array(5, 10)

        assert len(output) == 10
        assert max(output) <= 5

        output = create_array(4865, 300)

        assert len(output) == 300
        assert max(output) <= 4865

    def test_compute_reciprocals(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        compute_reciprocals(array)

    def test_jit_compute_reciprocals(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        compute_reciprocals(array)

    def test_jit_options_compute_reciprocals(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        compute_reciprocals(array)

    def test_cupy_compute_reciprocals(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        compute_reciprocals(array)

    def test_get_menu_options(self):
        A00279297_Research.input = lambda valid: 5
        output = get_menu_options()
        assert output == 5

    def test_cpu_searching_linear(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        cpu_searching_linear(array, array2)

    def test_jit_searching_linear(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        jit_searching_linear(array, array2)

    def test_jit_options_searching_linear(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        jit_options_searching_linear(array, array2)

    def test_cpu_adding(self):
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        cpu_adding(array2)

    def test_jit_adding(self):
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        jit_adding(array2)

    def test_jit_options_adding(self):
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        jit_options_adding(array2)

    def test_multiply_matrix(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        multiply_matrix(array,array2)

    def test_cupy_multiply_matrix(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])

        cupy_multiply_matrix(array,array2)

    def test_jit_stat_analysis(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])
        jit_stat_analysis(array,array2)

    def test_jit_options_stat_analysis(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])
        jit_options_stat_analysis(array,array2)


    def test_stat_analysis(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])
        stat_analysis(array,array2)


    def test_cupy_stat_analysis(self):
        array = np.array([275, 245, 259, 419, 360, 274, 439, 281, 454, 155])
        array2 = np.array([2754, 2425, 2549, 4139, 3607, 2874, 4399, 2821, 4534, 1515])
        cupy_stat_analysis(array,array2)

    def teardown_method(self, method):
        # pytest function that is called after each test is ran
        # reverts the input back to the python standard
        # as opposed to custom set above
        A00279297_Research.input = input
