# Use OpenCL To Add Two Large Random Arrays (Using PyOpenCL Arrays and Elementwise)

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as cl_array  # Import PyOpenCL Arrays - Numpy arrays attached to an OpenCL buffer object
import numpy  # Import Numpy - tools to create and manipulate numbers

context = cl.create_some_context()  # Initialize the Context (one per computer)
queue = cl.CommandQueue(context)  # Instantiate a Command Queue (one per device)

a = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))  # Create a large random pyopencl array
b = cl_array.to_device(queue, numpy.random.randn(10).astype(numpy.float32))  # Create a large random pyopencl array
c = cl_array.empty_like(a)  # Create an empty pyopencl destination array

sum = cl.elementwise.ElementwiseKernel(context, "float *a, float *b, float *c", "c[i] = a[i] + b[i]", "sum")
# Create an elementwise kernel object
# Arguments - a string formatted as a C argument list
# Operation - a snippet of C that carries out the desired map operatino
# Name - the fuction name as which the kernel is compiled

sum(a, b, c)  # Call the elementwise kernel

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked