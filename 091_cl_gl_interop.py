import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLUT import *
import numpy

width = 800
height = 600
num_points = width * height

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

glutInit()
glutInitWindowSize(width, height)
glutCreateWindow('OpenCL/OpenGL Interop')
glutDisplayFunc(display)
glClearColor(1, 1, 1, 1)  # Set the background color to white
glColor(0, 0, 0)  # Set the foreground color to black

platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())

vertex_buffer = glGenBuffers(1)  # Generate the OpenGL Buffer
print vertex_buffer
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer) # Bind the vertex buffer to a target
rawGlBufferData(GL_ARRAY_BUFFER, num_points * 2 * 4, None, GL_STATIC_DRAW) # XXX Allocate memory for the buffer
glEnableClientState(GL_VERTEX_ARRAY)  # XXX The vertex array is enabled for writing and used during rendering when glDrawArrays is called
glVertexPointer(2, GL_FLOAT, 0, None)  # Define an array of vertex data (size, type, stride, pointer)
cl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vertex_buffer))

program = cl.Program(context, kernel).build()
queue = cl.CommandQueue(context)
cl.enqueue_acquire_gl_objects(queue, [cl_buffer])
program.generate_sin(queue, (num_points,), None, cl_buffer)
cl.enqueue_release_gl_objects(queue, [cl_buffer])
queue.finish()
glFlush()

glutMainLoop()