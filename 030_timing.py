# Test the speed of your PyOpenCL program

import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy  # Import number tools
from time import time  # Import time tools
 
a = numpy.random.rand(1000).astype(numpy.float32)  # Create a random array to add
b = numpy.random.rand(1000).astype(numpy.float32)  # Create a random array to add

def cpu_array_sum(a, b):  # Sum two arrays on the CPU
    c_cpu = numpy.empty_like(a)  # Create the destination array
    cpu_start_time = time()  # Get the CPU start time
    for i in range(1000):
            for j in range(1000):  # 1000 times add each number and store it
                    c_cpu[i] = a[i] + b[i]  # This add operation happens 1,000,000 times XXX
    cpu_end_time = time()  # Get the CPU end time
    print("CPU Time: {0} s".format(cpu_end_time - cpu_start_time))  # Print how long the CPU took
    return c_cpu  # Return the sum of the arrays

def gpu_array_sum(a, b):
    context = cl.create_some_context()  # Initialize the Context
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Instantiate a Queue with profiling (timing) enabled
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)  # Create three buffers (plans for areas of memory on the device)
    program = cl.Program(context, """
    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
    {
        int i = get_global_id(0);
        int j;
        for(j = 0; j < 1000; j++)
        {
            c[i] = a[i] + b[i];
        }
    }""").build()  # Compile the device program
    gpu_start_time = time()  # Get the GPU start time
    event = program.sum(queue, a.shape, None, a_buffer, b_buffer, c_buffer)  # Enqueue the GPU sum program XXX
    event.wait()  # Wait until the event finishes XXX
    elapsed = 1e-9*(event.profile.end - event.profile.start)  # Calculate the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))  # Print the time it took to execute the kernel
    c_gpu = numpy.empty_like(a)  # Create an empty array the same size as array a
    cl.enqueue_read_buffer(queue, c_buffer, c_gpu).wait()  # Read back the data from GPU memory into array c_gpu
    gpu_end_time = time()  # Get the GPU end time
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))  # Print the time the GPU program took, including both memory copies
    return c_gpu  # Return the sum of the two arrays

cpu_array_sum(a, b)  # Call the function that sums two arrays on the CPU
gpu_array_sum(a, b)  # Call the function that sums two arrays on the GPU
