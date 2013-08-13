# OpenCL + OpenGL program - visualization of particles with gravity

import pyopencl as cl # OpenCL - GPU computing interface
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import * # OpenGL - GPU rendering interface
from OpenGL.GLU import * # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import * # OpenGL tool to make a visualization window
import math # Simple number tools
import numpy # Complicated number tools
import sys # System tools (path, modules, maxint)

width = 800
height = 600
num_particles = 50000
time_step = .001

mouse_down = False
mouse_old = {'x': 0., 'y': 0.}
rotate = {'x': 0., 'y': 0., 'z': 0.}
translate = {'x': 0., 'y': 0., 'z': 0.}
initrans = {'x': 0., 'y': 0., 'z': -2.}

# Create buffer of initial positions
def particle_start_values(num_particles):
    """Initialize position, color and velocity arrays we also make Vertex
    Buffer Objects for the position and color arrays"""

    pos = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    col = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num_particles, 4), dtype=numpy.float32)

    pos[:,0] = numpy.sin(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    pos[:,0] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    pos[:,1] = numpy.cos(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    pos[:,1] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    pos[:,2] = 0.
    pos[:,3] = 1.

    col[:,:] = [1.,1.,1.,1.] # White particles

    vel[:,0] = pos[:,0] * 2.
    vel[:,1] = pos[:,1] * 2.
    vel[:,2] = 3.
    vel[:,3] = numpy.random.random_sample((num_particles, ))
    
    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo 
    pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    return (pos_vbo, col_vbo, vel)
    

# GL callback functions
def timer(t):
    glutTimerFunc(t, timer, t)
    glutPostRedisplay()

def on_key(*args):
    if args[0] == '\033' or args[0] == 'q':
        sys.exit()

def on_click(button, state, x, y):
    mouse_old['x'] = x
    mouse_old['y'] = y

def on_mouse_move(x, y):
    rotate['x'] += (y - mouse_old['y']) * .2
    rotate['y'] += (x - mouse_old['x']) * .2

    mouse_old['x'] = x
    mouse_old['y'] = y

def draw():
    """Render the particles"""        
    #update or particle positions by calling the OpenCL kernel
    execute(10) 
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    rotate['x'] += .5
    rotate['y'] += -.5
    #handle mouse transformations
    glTranslatef(initrans['x'], initrans['y'], initrans['z'])
    glRotatef(rotate['x'], 1, 0, 0)
    glRotatef(rotate['y'], 0, 1, 0) #we switched around the axis so make this rotate_z
    glTranslatef(translate['x'], translate['y'], translate['z'])
    
    #render the particles
    render()

    glutSwapBuffers()

def glut_setup():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Particle Simulation")

    glutDisplayFunc(draw)  # Called by GLUT every frame
    glutKeyboardFunc(on_key)
    glutMouseFunc(on_click)
    glutMotionFunc(on_mouse_move)
    glutTimerFunc(30, timer, 30)  # Call draw every 30 ms
    return(window)

def gl_setup():
    # Set up the OpenGL scene
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)

window = glut_setup()

gl_setup()


# Set up initial conditions
(pos_vbo, col_vbo, vel) = particle_start_values(num_particles)


def execute(sub_intervals):
    cl.enqueue_acquire_gl_objects(queue, gl_objects)

    global_size = (num_particles,)
    local_size = None

    kernelargs = (pos_cl, col_cl, vel_cl, pos_gen_cl, vel_gen_cl, numpy.float32(time_step))

    for i in xrange(0, sub_intervals):
        program.particle_fountain(queue, global_size, local_size, *(kernelargs))

    cl.enqueue_release_gl_objects(queue, gl_objects)
    queue.finish()

def render():
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    #setup the VBOs
    col_vbo.bind()
    glColorPointer(4, GL_FLOAT, 0, col_vbo)

    pos_vbo.bind()
    glVertexPointer(4, GL_FLOAT, 0, pos_vbo)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    #draw the VBOs
    glDrawArrays(GL_POINTS, 0, num_particles)

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)

platforms = cl.get_platforms()
context = cl.Context(properties=[
    (cl.context_properties.PLATFORM, platforms[0])]
    + get_gl_sharing_context_properties(), devices=None)  
queue = cl.CommandQueue(context)

kernel = """
__kernel void particle_fountain(__global float4* pos, __global float4* color, __global float4* vel, __global float4* pos_gen, __global float4* vel_gen, float time_step)
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
    life -= time_step;
    //if the life is 0 or less we reset the particle's values back to the original values and set life to 1
    if(life <= 0.f)
    {
        p = pos_gen[i];
        v = vel_gen[i];
        life = 1.0f;    
    }

    //we use a first order euler method to integrate the velocity and position (i'll expand on this in another tutorial)
    //update the velocity to be affected by "gravity" in the z direction
    v.z -= 9.8f*time_step;
    //update the position with the new velocity
    p.x += v.x*time_step;
    p.y += v.y*time_step;
    p.z += v.z*time_step;
    //store the updated life in the velocity array
    v.w = life;

    //update the arrays with our newly computed values
    pos[i] = p;
    vel[i] = v;

    //you can manipulate the color based on properties of the system
    //here we adjust the alpha
    color[i].w = life;

}"""

program = cl.Program(context, kernel).build()

# Set up vertex buffer objects and share them with OpenCL as GLBuffers
pos_vbo.bind()
col_vbo.bind()
pos_cl = cl.GLBuffer(context, mf.READ_WRITE, int(pos_vbo.buffers[0]))
col_cl = cl.GLBuffer(context, mf.READ_WRITE, int(col_vbo.buffers[0]))

# Pure OpenCL arrays
vel_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
pos_gen_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos_vbo.data)
vel_gen_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
queue.finish()

# Set up the list of GL objects to share with OpenCL
gl_objects = [pos_cl, col_cl]


glutMainLoop()