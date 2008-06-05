
/* See ../README for what this program is supposed to do. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void cleanup(void);       

#define TEX_WIDTH 256
#define TEX_HEIGHT 256

static GLuint* texData;

static int wWidth = 256;
static int wHeight = 256;

static GLuint texHandle;

void display(void) 
{
    glClearColor(1.f, 1.0f, 1.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texHandle);
    glTexImage2D(GL_TEXTURE_2D, 0, 4, TEX_WIDTH, TEX_HEIGHT,0,GL_RGBA, GL_UNSIGNED_BYTE,
                 texData);

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);    

    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.0f); glVertex3f(0.f, 0.f, 0.f);
    glTexCoord2f(1.f, 0.0f); glVertex3f(1.f, 0.f, 0.f);

    glTexCoord2f(1.f, 1.0f); glVertex3f(1.f, 1.f, 0.f);
    glTexCoord2f(0.f, 1.0f); glVertex3f(0.f, 1.f, 0.f);
    glEnd();

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

/* Note: this gets called when the window is created, so glViewport,
   glOrtho are guaranteed to be called even if the window is never
   resized. */
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
    glDeleteTextures(1, &texHandle);

    free(texData);
}

int main(int argc, char** argv) 
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    
    {
        int x;
        glGenTextures(1, &texHandle);
        texData = malloc(TEX_WIDTH * TEX_HEIGHT * 4);
    
        for (x = 0; x < TEX_WIDTH; x++)
        {
            int   y;
            float a    = (float)x * 0.1f;
            int   xcol = (sin(a)*0.5f+0.5f)*255.f;

            /* Force first line of the texture data to white so we can
               see which side is up. */
            for (y = 0; y < TEX_HEIGHT; y++)
                texData[x + y*TEX_HEIGHT] = (xcol | y*256) | (y == 0 ? 0xFFFFFF : 0); 
        }
    }

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
