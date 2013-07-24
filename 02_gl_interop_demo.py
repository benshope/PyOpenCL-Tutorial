# Visualize a sine wave

from OpenGL.GL import *  # Import the GPU rendering interface
from OpenGL.GLUT import *  # Import the OpenGL tool to make a visualization window
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData  # Import C-style access to the buffers
import pyopencl as cl  # Import the GPU computing interface
from pyopencl.tools import get_gl_sharing_context_properties
import sys  # Import system tools (path, modules, maxint)

num_points = 10000 # Number of data structures (vectors/vertices) describing points in 2D space.

kernel = """__kernel void generate_sin(__global float2* a)
{
    int id = get_global_id(0);    /* Location in the entire array */
    int global_size = get_global_size(0);    /* Size of the entire array */
    float r = (float)id / (float)global_size;    /* r goes from 0 to 1 */
    float x = r * 16.0f * 3.1415f;    /* Some math thing */
    a[id].x = r * 2.0f - 1.0f;   /* X-coordinate of this graph point */
    a[id].y = native_sin(x);    /* Y-coordinate of this graph point */
}"""

def initialize():
    platform = cl.get_platforms()[0]  # Use the first vendor's OpenCL implementation installed on this machine
    context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())  # Create a Context (one per computer)
    """An OpenCL context is created with one or more devices. Contexts are used by the OpenCL runtime for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context."""
    glClearColor(1, 1, 1, 1)  # Set the background color to white
    glColor(0, 0, 1)  # Set the foreground color to blue
    vertex_buffer = glGenBuffers(1)  # Generate the OpenGL Buffer
    """Buffer Objects are OpenGL Objects that store an array of unformatted memory allocated by the OpenGL context (aka: the GPU). These can be used to store vertex data, pixel data retrieved from images or the framebuffer, and a variety of other things."""
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer) # Bind the vertex buffer to a target
    """Different types of objects are bound to different sets of targets in GL.  Each target has a specific meaning: to bind one object to one target means to use that object in whatever manner that target uses objects bound to it.

    The GL_ARRAY_BUFFER target, for buffer objects, indicates the intent to use that buffer object for vertex data. However, binding to this target alone does not do anything; it's only the call to glVertexAttribPointer (or an equivalent function) that uses whatever buffer was bound to that target for the attribute data for that attribute.

    The reason targets exist is because OpenGL is accessing special-purpose hardware on the GPU.  Certain hardware (containing sets of functionality) is only available to certain memory structures."""
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

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_LINE_STRIP, 0, num_points)
    glFlush()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)

if __name__ == '__main__':
    glutInit(sys.argv)
    if len(sys.argv) > 1:
        num_points = int(sys.argv[1])
    glutInitWindowSize(800, 160)
    glutInitWindowPosition(0, 0)
    glutCreateWindow('OpenCL/OpenGL Interop Tutorial: Sin Generator')
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    initialize()
    glutMainLoop()