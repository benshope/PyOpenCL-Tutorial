# Display the mandelbrot set

import pyopencl as cl
import numpy
from time import time

w = 512
h = 512
# Set the width and height of the window

def calc_fractal_opencl(q, maxiter):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    output = numpy.empty(q.shape, dtype=numpy.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

    program = cl.Program(context, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, real = 0;
        float imag = 0;

        output[gid] = 0;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;

            if (real*real + imag*imag > 4.0f)
                 output[gid] = curiter;
        }
    }
    """).build()

    program.mandelbrot(queue, output.shape, None, q_opencl,
            output_opencl, numpy.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

def calc_fractal_numpy(q, maxiter):
    output = numpy.resize(numpy.array(0,), q.shape)
    z = numpy.zeros(q.shape, numpy.complex64)

    for iter in range(maxiter):
        z = z*z + q
        done = numpy.greater(abs(z), 2.0)
        q = numpy.where(done,0+0j, q)
        z = numpy.where(done,0+0j, z)
        output = numpy.where(done, iter, output)
    return output

def calc_fractal_serial(q, maxiter):
    z = numpy.zeros(q.shape, numpy.complex64)
    output = numpy.resize(numpy.array(0,), q.shape)
    for i in range(len(q)):
        for iter in range(maxiter):
            z[i] = z[i]*z[i] + q[i]
            if abs(z[i]) > 2.0:
                q[i] = 0+0j
                z[i] = 0+0j
                output[i] = iter
    return output

calc_fractal = calc_fractal_opencl
# calc_fractal = calc_fractal_serial
# calc_fractal = calc_fractal_numpy

if __name__ == '__main__':
    import Tkinter as tk  # XXX What are these three imports?  Why are they here? XXX
    import Image          # PIL
    import ImageTk        # PIL

    class Mandelbrot(object):
        def __init__(self):
            # Create the window
            self.root = tk.Tk()
            self.root.title("Mandelbrot Set")
            self.create_image()
            self.create_label()
            # Start the event loop
            self.root.mainloop()

        def draw(self, x1, x2, y1, y2, maxiter=30):
            # draw the Mandelbrot set, from numpy example
            xx = numpy.arange(x1, x2, (x2-x1)/w)
            yy = numpy.arange(y2, y1, (y1-y2)/h) * 1j
            q = numpy.ravel(xx+yy[:, numpy.newaxis]).astype(numpy.complex64)

            start_main = time()
            output = calc_fractal(q, maxiter)
            end_main = time()

            secs = end_main - start_main
            print("Main took", secs)

            self.mandel = (output.reshape((h,w)) /
                    float(output.max()) * 255.).astype(numpy.uint8)

        def create_image(self):
            """"
            create the image from the draw() string
            """
            # you can experiment with these x and y ranges
            self.draw(-2.13, 0.77, -1.3, 1.3)
            self.im = Image.fromarray(self.mandel)
            self.im.putpalette(reduce(
                lambda a,b: a+b, ((i,0,0) for i in range(255))
            ))


        def create_label(self):
            # put the image on a label widget
            self.image = ImageTk.PhotoImage(self.im)
            self.label = tk.Label(self.root, image=self.image)
            self.label.pack()

    # test the class
    test = Mandelbrot()

