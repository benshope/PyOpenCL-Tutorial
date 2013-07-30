# Use OpenCL To Add Two Large Random Arrays (Using PyOpenCL Arrays)

import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy  # Import Numpy - tools to create and manipulate numbers
import pyopencl.array as cl_array  # Import PyOpenCL Arrays - Numpy arrays linked to an OpenCL buffer object

context = cl.create_some_context()  # Initialize the Context (one per computer)
queue = cl.CommandQueue(context)    # Instantiate a Command Queue (one per device)

a = cl_array.to_device(queue, numpy.random.rand(50000).astype(numpy.float32))
b = cl_array.to_device(queue, numpy.random.rand(50000).astype(numpy.float32)) # Create two large random pyopencl arrays
c = cl_array.empty_like(a)  # Create an empty pyopencl destination array

program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}""").build()  # Create the OpenCL program

program.sum(queue, a.shape, None, a.data, b.data, c.data)  # Enqueue the program for execution and store the result in c

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked