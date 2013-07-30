# Use the Time module to test the speed of your program

import pyopencl as cl
import numpy
from time import time

a = numpy.random.rand(1000).astype(numpy.float32)
b = numpy.random.rand(1000).astype(numpy.float32)

time_cpu_start = time()
c_cpu = numpy.empty_like(a)
for i in range(1000):
        for j in range(1000):
                c_cpu[i] = a[i] + b[i]
time_cpu_end = time()
print("\nCPU Time: {0} s".format(time_cpu_end - time_cpu_start))

time_gpu_start = time()
context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)

program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
    int loop;
    int i = get_global_id(0);
    for(loop = 0; loop < 1000; loop++)
    {
        c[i] = a[i] + b[i];
    }
}""").build()
event = program.sum(queue, a.shape, None, a_buffer, b_buffer, c_buffer)
event.wait()
elapsed = 1e-9*(event.profile.end - event.profile.start)
print("GPU Kernel Time: {0} s".format(elapsed))

c_gpu = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buffer, c_gpu).wait()
time_gpu_end = time()

print("GPU Time: {0} s\n".format(time_gpu_end - time_gpu_start))