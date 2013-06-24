#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy
import inspect

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()  # why is this named with underscores?
        self.queue = cl.CommandQueue(self.ctx) # why is this named with no underscores?

    def loadProgram(self, kernelfile):
        #read in the OpenCL source file as a string
        f = open(kernelfile, 'r')
        kernel = "".join(f.readlines())
        print kernel
        #create the program
        #to build the program - why do we only need the context and kernel?  why not the queue too?
        self.program = cl.Program(self.ctx, kernel).build()

    def popCorn(self): # why is this called popcorn?
        mf = cl.mem_flags # memory flags are some kind of sub-module of cl?  why are they called flags?
        print inspect.getdoc(mf)
        #initialize client side (CPU) arrays
        self.a = numpy.array(range(10), dtype=numpy.float32)
        self.b = numpy.array(range(10), dtype=numpy.float32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.b.nbytes)

    def execute(self):
        self.program.part1(self.queue, self.a.shape, None, self.a_buf, self.b_buf, self.dest_buf)
        c = numpy.empty_like(self.a)
        cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
        print "a", self.a
        print "b", self.b
        print "c", c



if __name__ == "__main__":
    example = CL()
    example.loadProgram("part1.cl")
    example.popCorn()
    example.execute()

