# Visualization of particles with gravity

import pyopencl as cl # OpenCL - GPU computing interface
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import * # OpenGL - GPU rendering interface
from OpenGL.GLU import * # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import * # OpenGL tool to make a visualization window
from OpenGL.arrays import vbo 
import numpy # Number tools
import sys # System tools (path, modules, maxint)

width = 800
height = 600
num_particles = 100000
time_step = .01
mouse_down = False
mouse_old = {'x': 0., 'y': 0.}
rotate = {'x': 0., 'y': 0., 'z': 0.}
translate = {'x': 0., 'y': 0., 'z': 0.}
initial_translate = {'x': 0., 'y': 0., 'z': -2.5}

def glut_window():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Particle Simulation")

    glutDisplayFunc(on_display)  # Called by GLUT every frame
    glutKeyboardFunc(on_key)
    glutMouseFunc(on_click)
    glutMotionFunc(on_mouse_move)
    glutTimerFunc(10, on_timer, 10)  # Call draw every 30 ms

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)

    return(window)

def initial_buffers(num_particles):
    np_position = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    np_color = numpy.ndarray((num_particles, 4), dtype=numpy.float32)
    np_velocity = numpy.ndarray((num_particles, 4), dtype=numpy.float32)

    np_position[:,0] = numpy.sin(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    np_position[:,0] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    np_position[:,1] = numpy.cos(numpy.arange(0., num_particles) * 2.001 * numpy.pi / num_particles) 
    np_position[:,1] *= numpy.random.random_sample((num_particles,)) / 3. + .2
    np_position[:,2] = 0.
    np_position[:,3] = 1.

    np_color[:,:] = [1.,1.,1.,1.] # White particles

    np_velocity[:,0] = np_position[:,0] * 2.
    np_velocity[:,1] = np_position[:,1] * 2.
    np_velocity[:,2] = 3.
    np_velocity[:,3] = numpy.random.random_sample((num_particles, ))
    
    gl_position = vbo.VBO(data=np_position, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_position.bind()
    gl_color = vbo.VBO(data=np_color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_color.bind()

    return (np_position, np_velocity, gl_position, gl_color)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
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

def on_display():
    """Render the particles"""        
    # Update or particle positions by calling the OpenCL kernel
    cl.enqueue_acquire_gl_objects(queue, cl_gl_objects)
    kernelargs = (cl_gl_position, cl_gl_color, cl_velocity, cl_position_gen, cl_velocity_gen, numpy.float32(time_step))
    program.particle_fountain(queue, (num_particles,), None, *(kernelargs))
    cl.enqueue_release_gl_objects(queue, cl_gl_objects)
    queue.finish()
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Handle mouse transformations
    glTranslatef(initial_translate['x'], initial_translate['y'], initial_translate['z'])
    glRotatef(rotate['x'], 1, 0, 0)
    glRotatef(rotate['y'], 0, 1, 0) #we switched around the axis so make this rotate_z
    glTranslatef(translate['x'], translate['y'], translate['z'])
    
    # Render the particles
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    #setup the VBOs
    gl_color.bind()
    glColorPointer(4, GL_FLOAT, 0, gl_color)
    gl_position.bind()
    glVertexPointer(4, GL_FLOAT, 0, gl_position)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    #draw the VBOs
    glDrawArrays(GL_POINTS, 0, num_particles)

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)

    glutSwapBuffers()

window = glut_window()

(np_position, np_velocity, gl_position, gl_color) = initial_buffers(num_particles)

platform = cl.get_platforms()[0]
context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())  
queue = cl.CommandQueue(context)

cl_velocity = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_velocity)
cl_position_gen = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_position)
cl_velocity_gen = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_velocity)

cl_gl_position = cl.GLBuffer(context, mf.READ_WRITE, int(gl_position.buffers[0]))
cl_gl_color = cl.GLBuffer(context, mf.READ_WRITE, int(gl_color.buffers[0]))
cl_gl_objects = [cl_gl_position, cl_gl_color]

kernel = """__kernel void particle_fountain(__global float4* position, __global float4* color, __global float4* velocity, __global float4* position_gen, __global float4* vel_gen, float time_step)
{
    //get our index in the array
    unsigned int i = get_global_id(0);
    //copy position and velocity for this iteration to a local variable
    //note: if we were doing many more calculations we would want to have opencl
    //copy to a local memory array to speed up memory access (this will be the subject of a later tutorial)
    float4 p = position[i];
    float4 v = velocity[i];

    //we've stored the life in the fourth component of our velocity array
    float life = velocity[i].w;
    //decrease the life by the time step (this value could be adjusted to lengthen or shorten particle life
    life -= time_step;
    //if the life is 0 or less we reset the particle's values back to the original values and set life to 1
    if(life <= 0.f)
    {
        p = position_gen[i];
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
    position[i] = p;
    velocity[i] = v;

    //you can manipulate the color based on properties of the system
    //here we adjust the alpha
    color[i].w = life;

}"""

program = cl.Program(context, kernel).build()

glutMainLoop()