# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
from matplotlib.pyplot import imshow

# print("hello paddy my boi")

import cupy as cp
import numpy as np
# x = cp.arange(6).reshape(2, 3).astype('f')
#
# cp.array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype="float32")
# x.sum(axis=1)
# cp.array([  3.,  12.], dtype="float32")
#
# print(x)
#
#
# y = np.arange(6).reshape(2, 3).astype('f')
#
# np.array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype="float32")
# y.sum(axis=1)
# np.array([  3.,  12.], dtype="float32")
#
# # print(y)
# import cupy as cp
# import numpy as np
# import time
#
# np.random.seed(0)
#
#
# def compute_reciprocals(values):
#        output = np.empty(len(values))
#        for i in range(len(values)):
#               output[i] = 1.0 / values[i]
#        return output
#
# def compute_reciprocals_gpu(values):
#        output = cp.empty(len(values))
#        for i in range(len(values)):
#               output[i] = 1.0 / values[i]
#        return output

# values = np.random.randint(1, 10, size=5)
# compute_reciprocals(values)

# # cpu takes about 3 seconds
# start_time = time.time()
#
# big_array = np.random.randint(1, 100, size=1000000)
# compute_reciprocals(big_array)
#
# print("--- %s seconds ---" % (time.time() - start_time))
#
#
# ## gpu takes 33 seconds
# start_time = time.time()
#
# big_array = cp.random.randint(1, 100, size=1000000)
# compute_reciprocals_gpu(big_array)
#
# print("--- %s seconds ---" % (time.time() - start_time))

## great small exmaple of the fact using gpu does not speed up everything

## example of speeeeeddddd

# ## speed up as the numbes get higher
# ### Numpy and CPU
# s = time.time()
# x_cpu = np.ones((1000,1000,500))
# e = time.time()
# print(e - s)
# ### CuPy and GPU
# s = time.time()
# x_gpu = cp.ones((1000,1000,500))
# cp.cuda.Stream.null.synchronize()
# e = time.time()
# print(e - s)
#
# ### Numpy and CPU
# s = time.time()
# x_cpu *= 20
# e = time.time()
# print(e - s)
# ### CuPy and GPU
# s = time.time()
# x_gpu *= 20
# cp.cuda.Stream.null.synchronize()
# e = time.time()
# print(e - s)



# ############ good stuff use this
# #################
# ### numba??
# ## 5.9 vs 0.17
# from numba import jit, cuda, prange
# import numpy as np
# # to measure exec time
# from timeit import default_timer as timer
#
#
# # normal function to run on cpu
# def func(a):
#     for i in range(10000000):
#         a[i] += 1
#     for item in a:
#         if item == 45678:
#             print("hello")
#
#     for item in a:
#         if item == 78946:
#             print("hello")
#
#
# # function optimized to run on gpu
#
#
# @jit
# def func2(a):
#     for i in range(10000000):
#         a[i] += 1
#     for item in a:
#         if item == 45678:
#             print("hello")
#
#     for item in a:
#         if item == 78946:
#             print("hello")
#
#
# @jit(nopython=True, parallel=True)
# def func3(a):
#     for i in prange(10000000):
#         a[i] += 1
#     for item in a:
#         if item == 45678:
#             print("hello")
#
#     for item in a:
#         if item == 78946:
#             print("hello")
#
# @jit(nopython=True, parallel=True)
# def logistic_regression(Y, X, w, iterations):
#     for i in range(iterations):
#         w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
#     return w
#
#
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
#
#
#
#     # @jit(nopython=True, parallel=True)
#     # def simulator(out):
#     #     # iterate loop in parallel
#     #     for i in prange(out.shape[0]):
#     #         out[i] = run_sim()
# ##############################################

# from numba import jit
# import numpy as np
# import time
#
# x = np.arange(100).reshape(10, 10)
#
# @jit(nopython=True)
# def go_fast(a): # Function is compiled and runs in machine code
#     trace = 0.0
#     for i in range(a.shape[0]):
#         trace += np.tanh(a[i, i])
#     return a + trace
#
# # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
# start = time.time()
# go_fast(x)
# end = time.time()
# # print("Elapsed (with compilation) = %s" % (end - start))
#
# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = time.time()
# go_fast(x)
# end = time.time()
# # print("Elapsed (after compilation) = %s" % (end - start))






# import cv2
# # init video capture with video
# cap = cv2.VideoCapture(video)
#
# # get default video FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # get total number of video frames
# num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)


# data = np.genfromtxt("Data_Cycling.csv", dtype=float, delimiter=',', names=True)
# data = np.loadtxt("Data_Cycling.csv" , delimiter=",", skiprows=1, usecols=[4-17], encoding="utf8")
#
# print(data)
#
# print(data.head())
# import pandas as pd
# import numpy as np
# import cupy as cp
# df = pd.read_csv("data.csv")
# print(df.head())
# # 
# # print(df['Power'])
# # print(type(df['Power']))
# # print(df['Power'].mean())
# 
# 
# list_power = df['Power']
# 
# list_power_numpy = list_power.to_numpy()
# 
# print(list_power_numpy)
# print(type(list_power_numpy))
# 
# from timeit import default_timer as timer
# 
# start = timer()
# 
# print(np.mean(list_power_numpy))
# print(np.sort(list_power_numpy))
# print("numpy", timer() - start)
# #
# start = timer()
# print(cp.mean(list_power_numpy))
# print(cp.sort(list_power_numpy))
# print("cupy", timer() - start)


# import cv2
# import sys
#
# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     )
#
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()


##############################################


import numpy as np
from numba import vectorize
import time

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
  return a + b

# Initialize arrays
N = 100000000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)



# Add arrays on GPU
start = time.time()
C = Add(A, B)
end = time.time()
print("Elapsed (after compilation) Vector = %s" % (end - start))

start = time.time()

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

@vectorize(['float32(float32, float32)'], target='cpu')
def Add2(a, b):
  return a + b

# Initialize arrays
N = 100000000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

start = time.time()
# Add arrays on GPU
C = Add2(A, B)
end = time.time()
print("Elapsed (after compilation) CPU = %s" % (end - start))

# Initialize arrays
Nn = 100000000
Aa = np.ones(N, dtype=np.float32)
Bb = np.ones(A.shape, dtype=A.dtype)
Cc = np.empty_like(A, dtype=A.dtype)

start = time.time()
# Add arrays on GPU
Cc = Add(Aa, Bb)
end = time.time()
print("Elapsed (after compilation) gpu2 = %s" % (end - start))

###############################################################
#
# from numba import jit, cuda, prange
# import time
# # from numba import autojit
#
# @cuda.jit(device=True)
# def mandel(x, y, max_iters):
#   """
#   Given the real and imaginary parts of a complex number,
#   determine if it is a candidate for membership in the Mandelbrot
#   set given a fixed number of iterations.
#   """
#   c = complex(x, y)
#   z = 0.0j
#   for i in range(max_iters):
#     z = z*z + c
#     if (z.real*z.real + z.imag*z.imag) >= 4:
#       return i
#
#   return max_iters
#
# @cuda.jit
# def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
#   height = image.shape[0]
#   width = image.shape[1]
#
#   pixel_size_x = (max_x - min_x) / width
#   pixel_size_y = (max_y - min_y) / height
#
#   startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
#   startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
#   gridX = cuda.gridDim.x * cuda.blockDim.x;
#   gridY = cuda.gridDim.y * cuda.blockDim.y;
#
#   for x in range(startX, width, gridX):
#     real = min_x + x * pixel_size_x
#     for y in range(startY, height, gridY):
#       imag = min_y + y * pixel_size_y
#       image[y, x] = mandel(real, imag, iters)
#
# gimage = np.zeros((1024, 1536), dtype = np.uint8)
# blockdim = (32, 8)
# griddim = (32,16)
#
# start = time.time()
# d_image = cuda.to_device(gimage)
# mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20)
# d_image.to_host()
# end = time.time()
# print("Mandelbrot created on GPU in  = %s" % (end - start))
#
#
# imshow(gimage)

################################
# another one
##########################################################
# import numpy as np
# from numba import jit , vectorize
# from timeit import default_timer as timer
# import cupy
# # 1000000000 minimim
# count =1000000000
#
# random_array = np.random.randint(2000, size=(count), dtype="int32")
# random_array2 = np.random.randint(4872, size=(count), dtype="int32")
#
# def do_stuff(array,array2):
#     print(np.std(array))
#     print(np.mean(array))
#     print(np.median(array))
#     print(np.var(array))
#     # print(np.quantile(array,1))
#     # print(np.correlate(array,array2))
#
# @jit
# def do_stuff_jit(array,array2):
#     print(np.std(array))
#     print(np.mean(array))
#     print(np.median(array))
#     print(np.var(array))
#     # print(np.quantile(array,1))
#     # print(np.correlate(array,array2))
#
# # @vectorize(['integer(integer, integer)'], target='cuda')
# # def do_stuff_cupy(array):
# #     print(cupy.std(random_array))
# #     print(cupy.mean(random_array))
#
#
#
# print("normal")
# # 25 seconds
# start = timer()
# # print(np.std(random_array))
# # print(np.mean(random_array))
# do_stuff(random_array,random_array2)
# print("without GPU:", timer() - start)
#
# print("starting jit")
# # 19 seconds
# # much more realistic so a nice example
# start = timer()
# # print(np.std(random_array))
# # print(np.mean(random_array))
# do_stuff_jit(random_array,random_array2)
# print("with jit GPU:", timer() - start)

# maybe an exmaple of not plain sailing , maybe explain why ?
# start = timer()
# # print(np.std(random_array))
# # print(np.mean(random_array))
# do_stuff_cupy(random_array)
# print("with cupy GPU:", timer() - start)

##########################################################



############################

# ### so cup kinda sucks for standard implementation ?
# # maybe an entery that all code is not good code ?
# import numpy as np
# from numba import jit , vectorize
# from timeit import default_timer as timer
# import cupy
# # 1000000000 minimim
# count =1000000
#
# random_array = np.random.randint(2000, size=(count), dtype="int32")
# random_array2 = np.random.randint(4872, size=(count), dtype="int32")
#
# # print(random_array)
#
# def is_in_cpu(array,array2):
#     print(np.isin(41,array))
#     print(np.isin(81,array))
#     print(np.isin(81,array2))
#     print(np.isin(81,array2))
#
## is in no work for jit
# so good proof of whyy this is not pluf and play omg my code is sudeenly amazingly fast
# # @jit
# # def is_in_gpu(array):
# #     print(np.isin(41, array))
#
# def is_in_cupy(array,array2):
#     print(cupy.isin(cp.array(array), cp.array([14])))
#     print(cupy.isin(cp.array(array), cp.array([81])))
#     print(cupy.isin(cp.array(array2), cp.array([81])))
#     resultarray = cupy.isin(cp.array(array2), cp.array([81]))
#
#
#
# start = timer()
# is_in_cpu(random_array,random_array2)
# # is_in_gpu(random_array)
# print("with cupy GPU:", timer() - start)
#
# start = timer()
# is_in_cupy(random_array,random_array2)
# print("with cupy GPU:", timer() - start)
#############################################

#### useless i think

# import cupy as cp
#
# squared_diff = cp.ElementwiseKernel(
# 'float32 x, float32 y',
# 'float32 z',
# 'z = (x - y) * (x - y)',
# 'squared_diff')
#
#
# x = cp.arange(10, dtype=np.float32).reshape(2, 5)
# y = cp.arange(5, dtype=np.float32)
# print(squared_diff(x, y))
# # array([[ 0., 0., 0., 0., 0.],
# # [25., 25., 25., 25., 25.]], dtype=float32)
# squared_diff(x, 5)
# # array([[25., 16., 9., 4., 1.],
# # [ 0., 1., 4., 9., 16.]], dtype=float32)

###########################


# input two matrices of size n x m
# matrix1 = [[12, 7, 3],
#            [4, 5, 6],
#            [7, 8, 9]]
# matrix2 = [[5, 8, 1],
#            [6, 7, 3],
#            [4, 5, 9]]

from timeit import default_timer as timer

# count = 1000000
# matrix1 = np.random.randint(2000, size=(count), dtype="int32")
# # matrix2 = np.random.randint(4872, size=(count), dtype="int32")
#
#
# import numpy as np
#
# print(np.sin(matrix1))
# print(np.cos(matrix1))
# print(np.tan(matrix1))


