# Visualize a sine wave

import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import numpy
import sys

num_points = 10000

kernel = """__kernel void generate_sin(__global float2* a)
{
    int id = get_global_id(0);
    int global_size = get_global_size(0);
    float r = (float)id / (float)global_size;
    float x = r * 16.0f * 3.1415f;
    a[id].x = r * 2.0f - 1.0f;
    a[id].y = native_sin(x);
}"""


def glut_window():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Particle Simulation")

    glutDisplayFunc(on_draw)  # Called by GLUT every frame
    glutTimerFunc(30, on_timer, 30)  # Call draw every 30 ms

    glClearColor(1, 1, 1, 1)  # Set the background color to white
    glColor(0, 0, 0)  # Set the foreground color to black

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)

    return(window)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_LINE_STRIP, 0, num_points)
    glFlush()


window = glut_window()

platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())

vertex_buffer = glGenBuffers(1)  # Generate the OpenGL Buffer name
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer) # Bind the vertex buffer to a target
rawGlBufferData(GL_ARRAY_BUFFER, num_points * 2 * 4, None, GL_DYNAMIC_DRAW) # Allocate memory for the buffer
glEnableClientState(GL_VERTEX_ARRAY)  # The vertex array is enabled for client writing and used for rendering
glVertexPointer(2, GL_FLOAT, 0, None)  # Define an array of vertex data (size xyz, type, stride, pointer)

cl_buffer = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vertex_buffer))
program = cl.Program(context, kernel).build()
queue = cl.CommandQueue(context)
cl.enqueue_acquire_gl_objects(queue, [cl_buffer])
program.generate_sin(queue, (num_points,), None, cl_buffer)
cl.enqueue_release_gl_objects(queue, [cl_buffer])
queue.finish()
glFlush()

glutMainLoop()
