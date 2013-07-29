# A simple PyOpenCL program that sums two arrays

import pyopencl as cl # Access to the OpenCL API
import numpy # Tools to create and manipulate numbers

a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)  # Create two large random arrays

context = cl.create_some_context()  # Create a Context (one per computer)
queue = cl.CommandQueue(context)  # Create a Command Queue (usually one per processor)

flags = cl.mem_flags # Make a shortcut to get to cl.mem_flags

kernel = """__kernel void sum(__global float* a, __global float* b, __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}"""  # The C-like code that will run on the GPU

a_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, flags.WRITE_ONLY, b.nbytes)  # Allocate memory ???

program = cl.Program(context, kernel).build() # Compile the kernel into a Program

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)
# queue - the command queue this program will be sent to
# a.shape - a tuple of the array's dimensions
# a.buffer, b.buffer, c.buffer - the memory spaces this program deals with

c = numpy.empty_like(a) # Create an empty array the size of a
cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Copy the device data back to c

print "a: ", a
print "b: ", b
print "c: ", c  # Print everything, to show it worked