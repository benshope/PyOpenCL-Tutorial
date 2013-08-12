# OpenCL + OpenGL program - visualization of particles with gravity

import pyopencl as cl # OpenCL - GPU computing interface
mf = cl.mem_flags
from OpenGL.GL import *  # OpenGL - GPU rendering interface
from OpenGL.GLU import *  # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import *  # OpenGL tool to make a visualization window
import math # Simple number tools
import numpy # Complicated number tools
import sys # System tools (path, modules, maxint)
import time # XXX

width = 800
height = 600

class Part2(object):
    def __init__(self, num, dt, *args, **kwargs):
        self.clinit()
        
        kernel = """
        __kernel void part2(__global float4* pos, __global float4* color, __global float4* vel, __global float4* pos_gen, __global float4* vel_gen, float dt)
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

        }"""

        self.program = cl.Program(self.ctx, kernel).build()

        self.num = num
        self.dt = numpy.float32(dt)


    def loadData(self, pos_vbo, col_vbo, vel):
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data
        self.vel = vel

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        #For some there is no single buffer but an array of buffers
        #https://github.com/enjalot/adventures_in_opencl/commit/61bfd373478767249fe8a3aa77e7e36b22d453c4
        try:
            self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffer))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffer))
        except AttributeError:
            self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))
        self.col_vbo.bind()

        #pure OpenCL arrays
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
        self.pos_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        self.vel_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vel)
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.pos_cl, self.col_cl]
        

    def execute(self, sub_intervals):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)

        global_size = (self.num,)
        local_size = None

        kernelargs = (self.pos_cl, 
                      self.col_cl, 
                      self.vel_cl, 
                      self.pos_gen_cl, 
                      self.vel_gen_cl, 
                      self.dt)

        for i in xrange(0, sub_intervals):
            self.program.part2(self.queue, global_size, local_size, *(kernelargs))

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        self.queue.finish()
 

    def clinit(self):
        platforms = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platforms[0])]
                + get_gl_sharing_context_properties(), devices=None)
                
        self.queue = cl.CommandQueue(self.ctx)


    def render(self):
        
        glEnable(GL_POINT_SMOOTH)
        glPointSize(2)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, self.num)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)


def fountain_np(num):
    """numpy way of initializing data using ufuncs instead of loops"""
    import numpy
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

    return pos, col, vel


def fountain(num):
    """Initialize position, color and velocity arrays we also make Vertex
    Buffer Objects for the position and color arrays"""

    #pos, col, vel = fountain_loopy(num)
    pos, col, vel = fountain_np(num)
    
    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo 
    pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    return (pos_vbo, col_vbo, vel)
    

#number of particles
num = 20000
#time step for integration
dt = .001

class window(object):
    def __init__(self, *args, **kwargs):
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = {'x': 0., 'y': 0.}
        self.rotate = {'x': 0., 'y': 0., 'z': 0.}
        self.translate = {'x': 0., 'y': 0., 'z': 0.}
        self.initrans = {'x': 0., 'y': 0., 'z': -2.}

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("Part 2: Python")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        #setup OpenGL scene
        self.glinit()

        #set up initial conditions
        (pos_vbo, col_vbo, vel) = fountain(num)
        #create our OpenCL instance
        self.cle = Part2(num, dt)
        self.cle.loadData(pos_vbo, col_vbo, vel)

        glutMainLoop()
        
    def glinit(self):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., width / float(height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)

    # GL Callbacks
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old['x'] = x
        self.mouse_old['y'] = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old['x']
        dy = y - self.mouse_old['y']
        if self.mouse_down and self.button == 0: #left button
            self.rotate['x'] += dy * .2
            self.rotate['y'] += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate['z'] -= dy * .01 
        self.mouse_old['x'] = x
        self.mouse_old['y'] = y


    def draw(self):
        """Render the particles"""        
        #update or particle positions by calling the OpenCL kernel
        self.cle.execute(10) 
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans['x'], self.initrans['y'], self.initrans['z'])
        glRotatef(self.rotate['x'], 1, 0, 0)
        glRotatef(self.rotate['y'], 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate['x'], self.translate['y'], self.translate['z'])
        
        #render the particles
        self.cle.render()

        glutSwapBuffers()



if __name__ == "__main__":
    p2 = window()