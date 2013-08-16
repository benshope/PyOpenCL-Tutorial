# Use OpenCL To Add Two Random Arrays

import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

a = numpy.random.rand(50000).astype(numpy.float32)  # Create a random numpy array
b = numpy.random.rand(50000).astype(numpy.float32)  # Create a random numpy array
c = numpy.empty_like(a)  # Create an empty destination array

a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)
# Create three buffers (plans for areas of memory on the device)

kernel = """__kernel void sum(__global float* a, __global float* b, __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}"""  # Create a kernel (a string containing C-like OpenCL device code)

program = cl.Program(context, kernel).build()  
# Compile the kernel code into an executable OpenCL program

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)
# Enqueue the program for execution, causing data to be copied to the device
#  - queue: the command queue the program will be sent to
#  - a.shape: a tuple of the arrays' dimensions
#  - a.buffer, b.buffer, c.buffer: the memory spaces this program deals with

cl.enqueue_read_buffer(queue, c_buffer, c).wait()
# Copy the data for array c back to the host

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
# Print all three host arrays, to show sum() worked