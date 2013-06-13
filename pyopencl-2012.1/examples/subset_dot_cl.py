import numpy as np
import pyopencl as cl
import pyopencl.array

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 3

a = np.array(range(0,n**2))
print a
a = a.astype(np.float32)
print a
g_a = cl.array.to_device(queue, a)

start = 2
end = 7
subset = cl.array.to_device(queue, np.array(range(start,end)))
print subset.dtype

print "Subset Array", subset
print cl.array.subset_dot(subset,g_a,g_a)
print np.dot(a[start:end], a[start:end])

