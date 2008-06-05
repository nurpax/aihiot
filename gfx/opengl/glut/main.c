
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void cleanup(void);       

static int wWidth = 512;
static int wHeight = 512;

void display(void) 
{
    glClearColor(0.1f, 0.2f, 0.4f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glutSwapBuffers();
    glutPostRedisplay();
}

void idle(void) 
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    x; y;

    switch( key) 
    {
    case 27:
        exit (0);

    default: 
        break;
    }
}

void click(int button, int updown, int x, int y) 
{
    button; updown; x; y;
}

void motion (int x, int y) 
{
    x; y;
    glutPostRedisplay();
}

void reshape(int x, int y) 
{
    wWidth = x; 
    wHeight = y;

    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1); 
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void cleanup(void) 
{
}

int main(int argc, char** argv) {

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("GLUT sample");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    atexit(cleanup);

    glutMainLoop();

    return 0;
}
