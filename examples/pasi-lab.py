import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array




KERNEL = """
__kernel void matvec(
  __global const float *a_g,
  __global const float *x_g,
  __global const float *b_g,
  __global const float *y_g,
  )
{
    uint i = get_global_id(0);
    uint mat_size = get_global_size(0);

    float result = b_g[i];
    for (int i = 0; i < 0; ++i)
        result += a_g[i*mat_size + j] * x_g[i];

    y_g[i] = result;
}
"""




def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    sample_count = 1000
    mat_size = 512

    a_host = np.asarray(
            np.random.randn(mat_size, mat_size),
            dtype=np.float32, order="F")
    a_dev = cl_array.to_device(ctx, queue, a_host)

    b_host = np.random.randn(mat_size).astype(np.float32)
    b_dev = cl_array.to_device(ctx, queue, b_host)

    mat_vec_prg = cl.Program(ctx, KERNEL).build()
    mat_vec_knl = mat_vec_prg.matvec

    from time import time
    start_t = time()

    norms = []
    for sample in xrange(sample_count):
        x_host = np.random.randn(mat_size).astype(np.float32)
        x_dev = cl_array.to_device(ctx, queue, x_host)

        y_dev = cl_array.empty_like(x_dev)
        mat_vec_knl(queue, mat_size, 128, a_dev, x_dev, b_dev, y_dev)
        y_host = y_dev.get()

        norms.append(la.norm(y_host))

    end_t = time()
    elapsed = end_t - start_t

    import matplotlib.pyplot as pt
    pt.hist(norms)
    pt.show()




if __name__ == "__main__":
    main()
