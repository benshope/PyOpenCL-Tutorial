# Use OpenCL To Add Two Random Arrays (This Way Hides Details)

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np# Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

a = pycl_array.to_device(queue, np.random.rand(50000).astype(numpy.float32))
b = pycl_array.to_device(queue, np.random.rand(50000).astype(numpy.float32))  
# Create two random pyopencl arrays
c = pycl_array.empty_like(a)  # Create an empty pyopencl destination array

program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}""").build()  # Create the OpenCL program

program.sum(queue, a.shape, None, a.data, b.data, c.data)  # Enqueue the program for execution and store the result in c

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))  
# Print all three arrays, to show sum() worked