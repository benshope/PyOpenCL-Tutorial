import pyopencl as cl
from time import time
import numpy

block_size = 16  # XXX Why sixteen? XXX

context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

# XXX What do all these options do? XXX
if False:
    a_height = 4096
    #a_height = 1024
    a_width = 2048
    #a_width = 256
    #b_height == a_width
    b_width = a_height
elif False:
    # like PyCUDA
    a_height = 2516
    a_width = 1472
    b_height = a_width
    b_width = 2144
else:
    # CL SDK
    a_width = 50*block_size
    a_height = 100*block_size
    b_width = 50*block_size
    b_height = a_width

c_width = b_width
c_height = a_height

h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
h_b = numpy.random.rand(b_height, b_width).astype(numpy.float32)
h_c = numpy.empty((c_height, c_width)).astype(numpy.float32)


kernel_params = {"block_size": block_size,
        "w_a":a_width, "h_a":a_height, "w_b":b_width}

if "NVIDIA" in queue.device.vendor:
    options = "-cl-mad-enable -cl-fast-relaxed-math"
else:
    options = ""

kernel = """

#define BLOCK_SIZE %(block_size)d  // Thread block size

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
matrixMul( __global float* C, __global float* A, __global float* B)
{
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[tx + ty * BLOCK_SIZE] = A[a + WA * ty + tx];
        Bs[tx + ty * BLOCK_SIZE] = B[b + WB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[k + ty * BLOCK_SIZE] * Bs[tx + k * BLOCK_SIZE];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;

}"""


program = cl.Program(context, kernel % kernel_params,
        ).build(options=options)
kernel = program.matrixMul
#print program.binaries[0]

assert a_width % block_size == 0
assert a_height % block_size == 0
assert b_width % block_size == 0

# transfer host -> device -----------------------------------------------------
mf = cl.mem_flags

t1 = time()

d_a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c_buf = cl.Buffer(context, mf.WRITE_ONLY, size=h_c.nbytes)

push_time = time()-t1

# warmup ----------------------------------------------------------------------
for i in range(5):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size), 
            d_c_buf, d_a_buf, d_b_buf)
    event.wait()

queue.finish()

# actual benchmark ------------------------------------------------------------
t1 = time()

count = 20
for i in range(count):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size),
            d_c_buf, d_a_buf, d_b_buf)

event.wait()

gpu_time = (time()-t1)/count

# transfer device -> host -----------------------------------------------------
t1 = time()
cl.enqueue_copy(queue, h_c, d_c_buf)
pull_time = time()-t1

# timing output ---------------------------------------------------------------
gpu_total_time = gpu_time+push_time+pull_time

print "GPU push+compute+pull total [s]:", gpu_total_time
print "GPU push [s]:", push_time
print "GPU pull [s]:", pull_time
print "GPU compute (host-timed) [s]:", gpu_time
print "GPU compute (event-timed) [s]: ", (event.profile.end-event.profile.start)*1e-9

gflop = h_c.size * (a_width * 2.) / (1000**3.)
gflops = gflop / gpu_time

print
print "GFlops/s:", gflops

# cpu comparison --------------------------------------------------------------
t1 = time()
h_c_cpu = numpy.dot(h_a,h_b)
cpu_time = time()-t1

print
print "GPU==CPU:",numpy.allclose(h_c, h_c_cpu)
print
print "CPU time (s)", cpu_time
print

print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
print "GPU speedup (without transfer): ", cpu_time/gpu_time

