# Use the Time module to test the speed of your program

import pyopencl as cl
import numpy
import numpy.linalg as la
from time import time

a = numpy.random.rand(1000).astype(numpy.float32)
b = numpy.random.rand(1000).astype(numpy.float32)

c_cpu = numpy.empty_like(a)
c_gpu = numpy.empty_like(a)

time_cpu_start = time()
for i in range(1000):
        for j in range(1000):
                c_cpu[i] = a[i] + b[i]
                c_cpu[i] = c_cpu[i] * (a[i] + b[i])
                c_cpu[i] = c_cpu[i] * (a[i] / 2.0)
time_cpu_end = time()

print("\nSerial Time: {0} s".format(time_cpu_end - time_cpu_start))

context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)

prg = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
    int loop;
    int i = get_global_id(0);
    for(loop = 0; loop < 1000; loop++)
    {
        c[i] = a[i] + b[i];
        c[i] = c[i] * (a[i] + b[i]);
        c[i] = c[i] * (a[i] / 2.0);
    }
}""").build()

exec_evt = prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
exec_evt.wait()
elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

print("Parallel Time: {0} s\n".format(elapsed))

c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, dest_buf, c).wait()
error = 0
for i in range(1000):
        if c[i] != c_cpu[i]:
                error = 1
if error:
        print("Results doesn't match!!")
else:
        print("Results OK")
