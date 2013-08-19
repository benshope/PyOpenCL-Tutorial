# Visualize a sine wave

import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo 
import numpy
import sys

num_points = 10000
width = 800
height = 200
offset = .01

def glut_window():
    glutInit()
    glutInitWindowSize(800, 200)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow('Cellular Automata')
    glutDisplayFunc(on_display)
    glutTimerFunc(30, on_timer, 30)
    glClearColor(1, 1, 1, 1)  # Set the background color to white
    glColor(0, 0, 0)  # Set the foreground color to black

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)

    return(window)

def initial_buffers():

    data = numpy.ndarray((num_points, 2), dtype=numpy.float32)
    data[:,:] = [0.,1.]

    cl_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)

    gl_buffer = vbo.VBO(data=data, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    # gl_buffer.bind()

    buffer_name = glGenBuffers(1)  # Generate the OpenGL Buffer name
    glBindBuffer(GL_ARRAY_BUFFER, buffer_name) # Bind the vertex buffer to a target
    rawGlBufferData(GL_ARRAY_BUFFER, num_points * 2 * 4, None, GL_DYNAMIC_DRAW) # Allocate memory for the buffer
    glEnableClientState(GL_VERTEX_ARRAY)  # The vertex array is enabled for client writing and used for rendering
    glVertexPointer(2, GL_FLOAT, 0, None)  # Define an array of vertex data (size xyz, type, stride, pointer)
    cl_gl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(buffer_name))
    return (cl_buffer, gl_buffer, cl_gl_buffer)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def on_display():
    cl.enqueue_acquire_gl_objects(queue, [cl_gl_buffer])
    program.generate_sin(queue, (num_points,), None, cl_gl_buffer, numpy.float32(offset))
    cl.enqueue_release_gl_objects(queue, [cl_gl_buffer])
    queue.finish()
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # pos_vbo.bind()
    # glVertexPointer(4, GL_FLOAT, 0, pos_vbo)
    # glEnableClientState(GL_VERTEX_ARRAY)
    glDrawArrays(GL_POINTS, 0, num_points)
    # glDisableClientState(GL_VERTEX_ARRAY)
    glutSwapBuffers()


window = glut_window()
platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())
queue = cl.CommandQueue(context)

(cl_buffer, gl_buffer, cl_gl_buffer) = initial_buffers()

kernel = """__kernel void generate_sin(__global float2* a, float offset)
{
    int id = get_global_id(0);
    int global_size = get_global_size(0);
    float r = (float)id / (float)global_size;
    float x = r * 16.0f * 3.1415f;
    a[id].x = r * 2.0f - 1.0f;
    a[id].y = native_sin(x) + offset;
}"""
program = cl.Program(context, kernel).build()

glutMainLoop()