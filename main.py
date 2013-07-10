# The simplest possible PyOpenCL program

import pyopencl as cl
import numpy

context = cl.create_some_context()  # create a Context (one per computer)
queue = cl.CommandQueue(context) # create a Command Queue (one per processor)
kernel = """__kernel void multiply(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);
    c[i] = a[i] * b[i];
}"""  # The kernel is the C code that will run on the GPU

program = cl.Program(context, kernel).build() # compile the kernel into a Program
# ??? Why doesn't compilation require the name of a device ???
# It seems like this step would need a specific processor to compile to
flags = cl.mem_flags
# No idea what these 'flags' are

a = numpy.array(range(10), dtype=numpy.float32)
b = numpy.array(range(10), dtype=numpy.float32)

a_buf = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(context, flags.WRITE_ONLY, b.nbytes)

program.multiply(queue, a.shape, None, a_buf, b_buf, c_buf)
c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buf, c).wait()

print "a", a
print "b", b
print "c", c