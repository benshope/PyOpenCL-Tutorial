# Use OpenCL To Add Two Large Random Arrays (Using PyOpenCL Arrays and Elementwise)

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import numpy

context = cl.create_some_context()
queue = cl.CommandQueue(context)

a = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))
b = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))
c = cl_array.empty_like(a)

linear_combination = ElementwiseKernel(context,
        "float *a, float *b, float *c",
        "c[i] = a[i] + b[i]",
        "linear_combination")
# arguments – a string formatted as a C argument list
# operation – a snippet of C that carries out the desired ‘map’ operation. The current index is available as the variable i. operation may contain the statement PYOPENCL_ELWISE_CONTINUE, which will terminate processing for the current element.
# name – the function name as which the kernel is compiled

linear_combination(a, b, c)

assert numpy.linalg.norm((c - (5*a+6*b)).get()) < 1e-5

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked