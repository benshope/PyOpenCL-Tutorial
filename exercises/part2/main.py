# http://pyopengl.sourceforge.net/documentation/manual-3.0/index.xhtml
# http://pyopengl.sourceforge.net/context/tutorials/nehe1.xhtml

# === LIBRARIES ===
from OpenGL.GL import *  # Wrapper to communicate with OpenGL
from OpenGL.GLU import *  #  Some tools for OpenGL (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import *  # Make a visualization window
import sys # System tools (path, modules, maxint)

# === LOCAL FILES ===
import glutil
from vector import Vec
import part2 # OpenCL Kernels
import initialize  # Functions for initial values of particles

num = 20000  # Number of particles
dt = .001  # Time step ??? for integration ???

class window(object):
    def __init__(self, *args, **kwargs):
        # Mouse handling initial values
        self.mouse_down = False
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -2.])

        self.width = 640
        self.height = 480

        # Initialize the window
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("Part 2: Python")

        # Draw current frame
        glutDisplayFunc(self.draw)

        # Accept user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)

        # Call draw every 30 milliseconds
        glutTimerFunc(30, self.timer, 30)

        # Set up OpenGL scene
        self.glinit()
        (pos_vbo, col_vbo, vel) = initialize.fountain(num)

        # Create the OpenCL instance
        self.cle = part2.Part2(num, dt)
        self.cle.loadData(pos_vbo, col_vbo, vel)

        glutMainLoop()

    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., self.width / float(self.height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)

    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        if args[0] == 'q':
            sys.exit()
        elif args[0] == 't':
            print self.cle.timings

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y

    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01
        self.mouse_old.x = x
        self.mouse_old.y = y

    def draw(self):
        """Render the particles"""
        #update or particle positions by calling the OpenCL kernel
        self.cle.execute(10)
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)

        #render the particles
        self.cle.render()

        #draw the x, y and z axis as lines
        glutil.draw_axes()

        glutSwapBuffers()


if __name__ == "__main__":
    p2 = window()