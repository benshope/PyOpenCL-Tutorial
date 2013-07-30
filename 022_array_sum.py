# Use OpenCL To Add Two Large Random Arrays (Using PyOpenCL Arrays and Elementwise)

import pyopencl as cl
import pyopencl.array as cl_array
import numpy

context = cl.create_some_context()
queue = cl.CommandQueue(context)

a = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))
b = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))
c = cl_array.empty_like(a)

sum = cl.elementwise.ElementwiseKernel(context, "float *a, float *b, float *c", "c[i] = a[i] + b[i]", "sum")
# arguments - a string formatted as a C argument list
# operation - a snippet of C that carries out the desired map operatino
# name - the fuction name as which the kernel is compiled

sum(a, b, c)

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))