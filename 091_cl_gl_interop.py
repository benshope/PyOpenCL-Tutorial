# Game of life

import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import numpy

num_points = 10000 # Number of data structures (vectors/vertices) describing points in 2D space.

kernel = """__kernel void generate_sin(__global float2* a)
{
    int id = get_global_id(0);
    int global_size = get_global_size(0);
    float r = (float)id / (float)global_size;
    float x = r * 16.0f * 3.1415f;
    a[id].x = r * 2.0f - 1.0f;
    a[id].y = native_sin(x);
}"""

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_LINE_STRIP, 0, num_points)
    glFlush()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)


glutInit()
glutInitWindowSize(800, 600)
glutInitWindowPosition(0, 0)
glutCreateWindow('OpenCL/OpenGL Interop Tutorial: Sin Generator')
glutDisplayFunc(display)
glutReshapeFunc(reshape)

platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())
glClearColor(1, 1, 1, 1)  # Set the background color to white
glColor(0, 0, 1)  # Set the foreground color to blue
vertex_buffer = glGenBuffers(1)  # Generate the OpenGL Buffer
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer) # Bind the vertex buffer to a target
rawGlBufferData(GL_ARRAY_BUFFER, num_points * 2 * 4, None, GL_STATIC_DRAW) # Allocate memory for the buffer
glEnableClientState(GL_VERTEX_ARRAY)  # The vertex array is enabled for writing and used during rendering when glDrawArrays is called
glVertexPointer(2, GL_FLOAT, 0, None)  # Define an array of vertex data (size, type, stride, pointer)
cl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vertex_buffer))  #  An OpenCL buffer object from an OpenGL buffer object (context, flags, GL buffer)
prog = cl.Program(context, kernel).build()
queue = cl.CommandQueue(context)
cl.enqueue_acquire_gl_objects(queue, [cl_buffer])
prog.generate_sin(queue, (num_points,), None, cl_buffer)
cl.enqueue_release_gl_objects(queue, [cl_buffer])
queue.finish()
glFlush()

glutMainLoop()