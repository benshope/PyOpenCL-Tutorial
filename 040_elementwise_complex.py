import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import numpy

context = cl.create_some_context()
queue = cl.CommandQueue(context)

a_gpu = cl_array.to_device(queue, ( numpy.random.randn(10) + 1j*numpy.random.randn(10)).astype(numpy.complex64))
b_gpu = cl_array.to_device(queue,( numpy.random.randn(10) + 1j*numpy.random.randn(10)).astype(numpy.complex64))

complex_prod = ElementwiseKernel(context,
        "float a, "
        "float2 *x, "
        "float2 *y, "
        "float2 *z",
        "z[i] = a * complex_mul(x[i], y[i])",
        "complex_prod",
        preamble="""
        #define complex_ctr(x, y) (float2)(x, y)
        #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
        #define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
        #define conj(a) complex_ctr((a).x, -(a).y)
        #define conj_transp(a) complex_ctr(-(a).y, (a).x)
        #define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
        """)

complex_add = ElementwiseKernel(context,
        "float2 *x, "
        "float2 *y, "
        "float2 *z",
        "z[i] = x[i] + y[i]",
        "complex_add")

real_part = ElementwiseKernel(context,
        "float2 *x, float *z",
        "z[i] = x[i].x",
        "real_part")

c_gpu = cl_array.empty_like(a_gpu)
complex_prod(5, a_gpu, b_gpu, c_gpu)

c_gpu_real = cl_array.empty(queue, len(a_gpu), dtype=numpy.float32)
real_part(c_gpu, c_gpu_real)
print c_gpu.get().real - c_gpu_real.get()

final_array = c_gpu.get() # XXXXXX What does get() do?
