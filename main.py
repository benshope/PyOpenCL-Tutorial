# OpenCL + OpenGL program - visualization of particles with gravity

import pyopencl as cl # OpenCL - GPU computing interface
from OpenGL.GL import *  # OpenGL - GPU rendering interface
from OpenGL.GLU import *  #  OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import *  # OpenGL tool to make a visualization window
import numpy # Tools to manipulate numbers
import sys # System tools (path, modules, maxint)
import time # What does this do?

context = cl.create_some_context()  # Create a Context (one per computer)
queue = cl.CommandQueue(context)  # Create a Command Queue (one per processor)

particles = 20000  # Number of particles
time_step = .001  # Time between each frame

kernel = """__kernel void part2(__global float4* pos, __global float4* color, __global float4* vel, __global float4* pos_gen, __global float4* vel_gen, float dt)
{
    //get our index in the array
    unsigned int i = get_global_id(0);
    //copy position and velocity for this iteration to a local variable
    //note: if we were doing many more calculations we would want to have opencl
    //copy to a local memory array to speed up memory access (this will be the subject of a later tutorial)
    float4 p = pos[i];
    float4 v = vel[i];

    //we've stored the life in the fourth component of our velocity array
    float life = vel[i].w;
    //decrease the life by the time step (this value could be adjusted to lengthen or shorten particle life
    life -= dt;
    //if the life is 0 or less we reset the particle's values back to the original values and set life to 1
    if(life <= 0.f)
    {
        p = pos_gen[i];
        v = vel_gen[i];
        life = 1.0f;
    }

    //we use a first order euler method to integrate the velocity and position (i'll expand on this in another tutorial)
    //update the velocity to be affected by "gravity" in the z direction
    v.z -= 9.8f*dt;
    //update the position with the new velocity
    p.x += v.x*dt;
    p.y += v.y*dt;
    p.z += v.z*dt;
    //store the updated life in the velocity array
    v.w = life;

    //update the arrays with our newly computed values
    pos[i] = p;
    vel[i] = v;

    //you can manipulate the color based on properties of the system
    //here we adjust the alpha
    color[i].w = life;
}"""  # The C-like code that will run on the GPU

program = cl.Program(context, kernel).build() # Compile the kernel into a Program

flags = cl.mem_flags # Create a shortcut to OpenCL's memory instructions

a = numpy.random.rand(5).astype(numpy.float32)
b = numpy.random.rand(5).astype(numpy.float32)  # Create two large float arrays

a_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, flags.WRITE_ONLY, b.nbytes)  # Allocate memory ???

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)
# queue - the command queue this program will be sent to
# a.shape - a tuple of the array's dimensions
# a.buffer, b.buffer, c.buffer - the memory spaces this program deals with

c = numpy.empty_like(a) # Create a correctly-sized array
cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Execute everything and copy back c