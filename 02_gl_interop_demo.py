# Visualize a sine wave

from OpenGL.GL import *  # Import the GPU rendering interface
from OpenGL.GLUT import *  # Import the OpenGL tool to make a visualization window
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData  # Import C-style access to the buffers
import pyopencl as cl  # Import the GPU computing interface

num_vertices = 10000 # Why is this called vertices?

kernel = """
__kernel void generate_sin(__global float2* a)
{
    int id = get_global_id(0);  /* Where are we in this array */
    int global_size = get_global_size(0);  /* How big is this entire array */
    float r = (float)id / (float)global_size;  /* r goes from 0 to 1 */
    float x = r * 16.0f * 3.1415f;  /* Some math thing */
    a[id].x = r * 2.0f - 1.0f; /* x coordinate of this graph point */
    a[id].y = native_sin(x);  /* y coordinate of this graph point */
}
"""

def initialize():
    # Use the first vendor's OpenCL implementation installed on this machine
    platform = cl.get_platforms()[0]

    from pyopencl.tools import get_gl_sharing_context_properties
    import sys
    if sys.platform == "darwin":
        context = cl.Context(properties=get_gl_sharing_context_properties(),
                devices=[])
    else:
        # Some OSs prefer clCreateContextFromType, some prefer
        # clCreateContext. Try both.
        try:
            context = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties())
        except:
            context = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties(),
                devices = [platform.get_devices()[0]])

    glClearColor(1, 1, 1, 1)
    glColor(0, 0, 1)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer) # Link the vertex buffer to a target
    """Different types of objects have different sets of targets in OpenGL.  
    Each target has a specific meaning: to bind one object to one target means 
    to use that object in whatever manner that target uses objects bound to it.

    The reason targets exist is because OpenGL is accessing special-purpose hardware
    on the GPU.  Certain hardware (containing sets of functionality) is only 
    available to certain memory areas and certain memory structures.

    To understand targets visually, think about a letter.  In order to send a letter you 
    have to fill in several targets (State, Zip, City, Street...) in order to use 
    the special-purpose hardware that is the postal system.  If you wrote in 
    Alaska as the state, you couldn't also add Broadway as the street and 90210 
    as the zip code.
    """

    rawGlBufferData(GL_ARRAY_BUFFER, num_vertices * 2 * 4, None, GL_STATIC_DRAW) # Allocate memory for the buffer
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(2, GL_FLOAT, 0, None)
    coords_dev = cl.GLBuffer(context, cl.mem_flags.READ_WRITE, int(vertex_buffer))
    prog = cl.Program(context, kernel).build()
    queue = cl.CommandQueue(context)
    cl.enqueue_acquire_gl_objects(queue, [coords_dev])
    prog.generate_sin(queue, (num_vertices,), None, coords_dev)
    cl.enqueue_release_gl_objects(queue, [coords_dev])
    queue.finish()
    glFlush()

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_LINE_STRIP, 0, num_vertices)
    glFlush()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)

if __name__ == '__main__':
    import sys
    glutInit(sys.argv)
    if len(sys.argv) > 1:
        num_vertices = int(sys.argv[1])
    glutInitWindowSize(800, 160)
    glutInitWindowPosition(0, 0)
    glutCreateWindow('OpenCL/OpenGL Interop Tutorial: Sin Generator')
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    initialize()
    glutMainLoop()