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
        "float a, float *x, "
        "float b, float *y, "
        "float *z",
        "z[i] = a*x[i] + b*y[i]",
        "linear_combination")

linear_combination(5, a, 6, b, c)

assert numpy.linalg.norm((c - (5*a+6*b)).get()) < 1e-5

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked