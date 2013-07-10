# The simplest possible PyOpenCL program - Sums two arrays

import pyopencl as cl # Access to the OpenCL API
import numpy # Tools to create and manipulate numbers

context = cl.create_some_context()  # Create a Context (one per computer)
queue = cl.CommandQueue(context)  # Create a Command Queue (usually one per processor)
kernel = """__kernel void sum(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);
    c[i] = a[i] + b[i];
}"""  # The C-like code that will run on the GPU

program = cl.Program(context, kernel).build() # Compile the kernel into a Program

flags = cl.mem_flags # Create a shortcut to OpenCL's memory instructions

a = numpy.random.rand(5).astype(numpy.float32)
b = numpy.random.rand(5).astype(numpy.float32)  # Create two large float arrays


a_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, flags.WRITE_ONLY, b.nbytes)  # Allocate memory ???

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)
# call the program with arguments:
# queue - the command queue this program will be sent to
# a.shape - a tuple of the array's dimensions
# a.buffer, b.buffer, c.buffer - the three memory spaces this program deals with

c = numpy.empty_like(a) # Create a correctly-sized array
cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Execute everything and copy back c

print "a", a
print "b", b
print "c", c  # Print everything, to show the demo worked