import pyopencl as cl
import pyopencl.array as cl_array
import numpy

a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)

context = cl.create_some_context()
queue = cl.CommandQueue(context)

mf = cl.mem_flags
a_dev = cl_array.to_device(queue, a)
b_dev = cl_array.to_device(queue, b)
dest_dev = cl_array.empty_like(a_dev)

prg = cl.Program(context, """
    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
    {
      int i = get_global_id(0);
      c[i] = a[i] + b[i];
    }""").build()

prg.sum(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)

print(numpy.linalg.norm((dest_dev - (a_dev+b_dev)).get()))


"""Buffers are CL's version of malloc, while pyopencl.array.Array is a workalike of numpy arrays on the compute device.

So for the second version of the first part of your question, you may write a_gpu + 2 to get a new arrays that has 2 added to each number in your array, whereas in the case of the Buffer, PyOpenCL only sees a bag of bytes and cannot perform any such operation.

The second part of your question is the same in reverse: If you've got a PyOpenCL array, .get() copies the data back and converts it into a (host-based) numpy array. Since numpy arrays are one of the more convenient ways to get contiguous memory in Python, the second variant with enqueue_copy also ends up in a numpy array--but note that you could've copied this data into an array of any size (as long as it's big enough) and any type--the copy is performed as a bag of bytes, whereas .get() makes sure you get the same size and type on the host.

Bonus fact: There is of course a Buffer underlying each PyOpenCL array. You can get it from the .data attribute."""