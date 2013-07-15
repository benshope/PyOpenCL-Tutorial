# OpenCL + OpenGL program - visualization of particles with gravity

import pyopencl as cl # OpenCL - GPU computing interface
from OpenGL.GL import *  # OpenGL - GPU rendering interface
from OpenGL.GLU import *  # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import *  # OpenGL tool to make a visualization window
import math # Simple number tools
import numpy # Complicated number tools
import sys # System tools (path, modules, maxint)
import time # What does this do?

num_particles = 20000  # Number of particles
time_step = .001  # Time between each animation frame
window_width = 640
window_height = 480
mouse_down = False
mouse_old = Vec([0., 0.])
rotate = Vec([0., 0., 0.])
translate = Vec([0., 0., 0.])
initrans = Vec([0., 0., -2.])

glutDisplayFunc(draw)
glutKeyboardFunc(on_key)
glutMouseFunc(on_click)
glutMotionFunc(on_mouse_motion)
glutTimerFunc(30, timer, 30)
glutInit(sys.argv)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(window_width, window_height)
glutInitWindowPosition(0, 0)
window = glutCreateWindow("Python Particles Simulation")

glViewport(0, 0, window_width, window_height)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(60., window_width / float(window_height), .1, 1000.)
glMatrixMode(GL_MODELVIEW)

(pos_vbo, col_vbo, vel) = fountain(num_particles)  # Set up OpenGL scene

glutMainLoop()

def timer(t):
    glutTimerFunc(t, timer, t)
    glutPostRedisplay()

def on_key(*args):
    if args[0] == 'q':
        sys.exit()
    elif args[0] == 't':
        print cle.timings

def on_click(button, state, x, y):
    if state == GLUT_DOWN:
        mouse_down = True
        button = button
    else:
        mouse_down = False
    mouse_old.x = x
    mouse_old.y = y

def on_mouse_motion(x, y):
    dx = x - mouse_old.x
    dy = y - mouse_old.y
    if mouse_down and button == 0: #left button
        rotate.x += dy * .2
        rotate.y += dx * .2
    elif mouse_down and button == 2: #right button
        translate.z -= dy * .01
    mouse_old.x = x
    mouse_old.y = y

def draw():
    """Render the particles"""
    #update or particle positions by calling the OpenCL kernel
    cle.execute(10)
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    #handle mouse transformations
    glTranslatef(initrans.x, initrans.y, initrans.z)
    glRotatef(rotate.x, 1, 0, 0)
    glRotatef(rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
    glTranslatef(translate.x, translate.y, translate.z)

    #render the particles
    cle.render()

    #draw the x, y and z axis as lines
    glutil.draw_axes()

    glutSwapBuffers()



def buffers(num_particles):
    """Initialize position, color and velocity arrays we also make Vertex
    Buffer Objects for the position and color arrays"""

    pos = numpy.ndarray((num, 4), dtype=numpy.float32)
    col = numpy.ndarray((num, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num, 4), dtype=numpy.float32)

    pos[:,0] = numpy.sin(numpy.arange(0., num) * 2.001 * numpy.pi / num)
    pos[:,0] *= numpy.random.random_sample((num,)) / 3. + .2
    pos[:,1] = numpy.cos(numpy.arange(0., num) * 2.001 * numpy.pi / num)
    pos[:,1] *= numpy.random.random_sample((num,)) / 3. + .2
    pos[:,2] = 0.
    pos[:,3] = 1.

    col[:,0] = 0.
    col[:,1] = 1.
    col[:,2] = 0.
    col[:,3] = 1.

    vel[:,0] = pos[:,0] * 2.
    vel[:,1] = pos[:,1] * 2.
    vel[:,2] = 3.
    vel[:,3] = numpy.random.random_sample((num, ))

    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo
    pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    return (pos_vbo, col_vbo, vel)








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


context = cl.create_some_context()  # Create a Context (one per computer)
queue = cl.CommandQueue(context)  # Create a Command Queue (one per processor)

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
cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Execute everything and copy back