# Use OpenCL To Add Two Random Arrays (The More Explicit Way)

import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy as np  # Import Np number tools

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
context = cl.Context([device]) # Create a context with your device
queue = cl.CommandQueue(context)  # Create a command queue with your context

np_a = np.random.rand(50000).astype(np.float32)  # Create a random np array
np_b = np.random.rand(50000).astype(np.float32)  # Create a random np array
np_c = np.empty_like(np_a)  # Create an empty destination array

cl_a = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=np_a)
cl_b = cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=np_b)
cl_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np_c.nbytes)
# Create three buffers (plans for areas of memory on the device)

kernel = """__kernel void sum(__global float* a, __global float* b, __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}"""  # Create a kernel (a string containing C-like OpenCL device code)

program = cl.Program(context, kernel).build()
# Compile the kernel code into an executable OpenCL program

program.sum(queue, np_a.shape, None, cl_a, cl_b, cl_c)
# Enqueue the program for execution, causing data to be copied to the device
#  - queue: the command queue the program will be sent to
#  - np_a.shape: a tuple of the arrays' dimensions
#  - cl_a, cl_b, cl_c: the memory spaces this program deals with

np_arrays = [np_a, np_b, np_c]
cl_arrays = [cl_a, cl_b, cl_c]

for x in range(3):
    cl.enqueue_copy(queue, cl_arrays[x], np_arrays[x])
queue.finish()
# Copy the data for array c back to the host

for x in np_arrays:
	print(x)
# Print all three host arrays, to show sum() worked