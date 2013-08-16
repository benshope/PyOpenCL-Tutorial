# Use OpenCL To Add Two Random Arrays (Using PyOpenCL Arrays and Elementwise)

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as cl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

a = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))  # Create a random pyopencl array
b = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))  # Create a random pyopencl array
c = cl_array.empty_like(a)  # Create an empty pyopencl destination array

sum = cl.elementwise.ElementwiseKernel(context, "float *a, float *b, float *c", "c[i] = a[i] + b[i]", "sum")
# Create an elementwise kernel object
#  - Arguments: a string formatted as a C argument list
#  - Operation: a snippet of C that carries out the desired map operatino
#  - Name: the fuction name as which the kernel is compiled

sum(a, b, c)  # Call the elementwise kernel

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
# Print all three arrays, to show sum() worked