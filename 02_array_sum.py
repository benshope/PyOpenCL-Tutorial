# Use OpenCL To Add Two Large Random Arrays

import pyopencl as cl # Import access to the OpenCL GPU computing API
import numpy # Import numpy tools to create and manipulate numbers

context = cl.create_some_context()  # Initialize the Context (one per computer)
queue = cl.CommandQueue(context)  # Instantiate a Command Queue (one per device)

a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)  # Create two large random numpy arrays
c = numpy.empty_like(a) # Create an empty numpy destination array

a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)  # Specify three buffers (areas of memory) on the GPU.  Creating these buffers does not copy the data; the data will be copied automatically when the program requires it.

kernel = """__kernel void sum(__global float* a, __global float* b, __global float* c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}"""  # The C-like code that will run on the GPU

program = cl.Program(context, kernel).build() # Compile the kernel code into an OpenCL program

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)  # Enqueue and call the OpenCL program
# queue - the command queue this program will be sent to
# a.shape - a tuple of the array's dimensions
# a.buffer, b.buffer, c.buffer - the memory spaces this program deals with

cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Copy the device data back to numpy array c

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked