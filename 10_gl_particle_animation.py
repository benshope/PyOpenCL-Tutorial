# OpenCL + OpenGL program - visualization of particles with gravity


# ==== WHAT HAPPENS IN THIS PROGRAM ====
# 1. We import the environment
# 2. We set up the environment
# 3. We XXXX

# This code is unfinished.  Finish it.
# Find out if vertex buffers have any downsides when compared to buffers.
# Can vertex buffers be used as normal buffers?  This quesion might be answered along the way.


import pyopencl as cl # OpenCL - GPU computing interface
from OpenGL.GL import *  # OpenGL - GPU rendering interface
from OpenGL.GLU import *  # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import *  # OpenGL tool to make a visualization window
import math # Simple number tools
import numpy # Complicated number tools
import sys # System tools (path, modules, maxint)
import time # XXX



class Vec(np.ndarray):
    props = ['x', 'y', 'z', 'w']

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if len(obj) < 2 or len(obj) > 4:
            #not a 2,3 or 4 element vector!
            return None
        return obj

    def __array_finalize__(self, obj):
        #this gets called after object creation by numpy.ndarray
        #print "array finalize", obj
        if obj is None: return
        #we set x, y etc properties by iterating through ndarray
        for i in range(len(obj)):
            setattr(self, Vec.props[i], obj[i])

    def __array_wrap__(self, out_arr, context=None):
        #this gets called after numpy functions are called on the array
        #out_arr is the output (resulting) array
        for i in range(len(out_arr)):
            setattr(out_arr, Vec.props[i], out_arr[i])
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        desc="""Vec2(data=%(data)s,"""  # x=%(x)s, y=%(y)s)"""
        for i in range(len(self)):
            desc += " %s=%s," % (Vec.props[i], getattr(self, Vec.props[i]))
        desc = desc[:-1]    #cut off last comma
        return desc % {'data': str(self) }


    def __setitem__(self, ind, val):
        #print "SETITEM", self.shape, ind, val
        if self.shape == ():    #dirty hack for dot product
            return
        self.__dict__[Vec.props[ind]] = val
        return np.ndarray.__setitem__(self, ind, val)
    
    def __setattr__(self, item, val):
        self[Vec.props.index(item)] = val




num_particles = 20000  # Number of particles
time_step = .001  # Time between each animation frame
window_width = 640  # Viewing window width
window_height = 480  # Viewing window height
mouse_down = False
mouse_old = Vec([0., 0.])
rotate = Vec([0., 0., 0.])
translate = Vec([0., 0., 0.])
initrans = Vec([0., 0., -2.])

# Functions that are called on window events
glutDisplayFunc(draw)
glutKeyboardFunc(key)
glutMouseFunc(click)
glutMotionFunc(mouse_move)
glutTimerFunc(30, tick, 30)  # Call a function in x msecs (msecs, func(value), value)

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

# ===== VERB FUNCTIONS ===== these have side-effects

def tick(msecs):
    glutTimerFunc(msecs, tick, msecs)   # Call a function in x msecs (msecs, func(value), value)
    glutPostRedisplay()  # Mark the current window as needing to be redisplayed

def key_press(*args):
    if args[0] == 'q':
        sys.exit()

def click(button, state, x, y):
    if state == GLUT_DOWN:
        mouse_down = True
        button = button
    else:
        mouse_down = False
    mouse_old.x = x
    mouse_old.y = y

def mouse_move(x, y):
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

    glutSwapBuffers()

def lights():
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)

    light_position = [10., 10., 200., 0.]
    light_ambient = [.2, .2, .2, 1.]
    light_diffuse = [.6, .6, .6, 1.]
    light_specular = [2., 2., 2., 0.]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glEnable(GL_LIGHT0)

    mat_ambient = [.2, .2, 1.0, 1.0]
    mat_diffuse = [.2, .8, 1.0, 1.0]
    mat_specular = [1.0, 1.0, 1.0, 1.0]
    high_shininess = 3.

    mat_ambient_back = [.5, .2, .2, 1.0]
    mat_diffuse_back = [1.0, .2, .2, 1.0]

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    glMaterialfv(GL_BACK, GL_AMBIENT,   mat_ambient_back);
    glMaterialfv(GL_BACK, GL_DIFFUSE,   mat_diffuse_back);
    glMaterialfv(GL_BACK, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_BACK, GL_SHININESS, high_shininess);


def draw_line(v1, v2):
    glBegin(GL_LINES)
    glVertex3f(v1.x, v1.y, v1.z)
    glVertex3f(v2.x, v2.y, v2.z)
    glEnd()


# ===== NOUN FUNCTIONS ===== these return something

def buffers(num_particles):  # Initialize the arrays of particle data: position, color, and velocity
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
    from OpenGL.arrays import vbo  # XXXXXXXXXXXXXXX
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
}"""  # The C-like code that will run on the device


context = cl.create_some_context()  # Create a Context (one per computer)
queue = cl.CommandQueue(context)  # Create a Command Queue (one per processor)

program = cl.Program(context, kernel).build() # Compile the kernel into a Program

flags = cl.mem_flags # Create a shortcut to OpenCL's memory instructions

a = numpy.random.rand(5).astype(numpy.float32)
b = numpy.random.rand(5).astype(numpy.float32)  # Create two large float arrays
c = numpy.empty_like(a) # Create a correctly-sized array

a_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, flags.WRITE_ONLY, b.nbytes)  # Allocate memory ???

program.sum(queue, a.shape, a_buffer, b_buffer, c_buffer)
# queue - the command queue this program will be sent to
# a.shape - a tuple of the array's dimensions
# a.buffer, b.buffer, c.buffer - the memory spaces this program deals with

cl.enqueue_read_buffer(queue, c_buffer, c).wait()  # Execute everything and copy back