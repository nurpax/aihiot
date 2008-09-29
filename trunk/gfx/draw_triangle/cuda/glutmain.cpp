
/* Original code from CUDA SDK's Mandelbrot example. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <cutil.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

#include "triangle_kernel.h"

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;

//Source image on the host side
uchar4 *h_Src = 0;

//Original image width and height
int imageW, imageH;

// Timer ID
unsigned int hTimer;

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false, g_AutoQuit = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

#define BUFFER_DATA(i) ((char *)0 + i)

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetTimerValue(hTimer) / 1000.f);
        sprintf(fps, "%sBasic CUDA sample %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        CUT_SAFE_CALL(cutResetTimer(hTimer));  
    }
}

// OpenGL display function
void displayFunc(void)
{
    float timeEstimate;
    uchar4 *d_dst = NULL;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));

    // Render anti-aliasing passes until we run out time (60fps approximately)
    RunTriangleRender(d_dst, imageW, imageH);
    cudaThreadSynchronize();
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(gl_PBO));

    // display image
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();

    glutSwapBuffers();
    computeFPS();
}

void cleanup()
{
    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);

    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}


// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    int seed;
    switch (k){
    case '\033':
    case 'q':
    case 'Q':
        printf("Shutting down...\n");
        CUT_SAFE_CALL(cutStopTimer(hTimer) );
        CUT_SAFE_CALL(cutDeleteTimer(hTimer));
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(gl_PBO));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        printf("Shutdown done.\n");
        exit(0);
        break;

    default:
        break;
    }

}

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
}

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
}

void idleFunc()
{
    glutPostRedisplay();
}

void createBuffers(int w, int h)
{
    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }
    h_Src = (uchar4*)malloc(w * h * 4);

    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }
    if (gl_PBO) {
        cudaGLUnregisterBufferObject(gl_PBO);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }
    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used as the
    // destination for OpenGL drawing calls.  But in our particular
    // case OpenGL is only used to display the content of the PBO,
    // specified by CUDA kernels, so we need to register/unregister it
    // only once.
    CUDA_SAFE_CALL( cudaGLRegisterBufferObject(gl_PBO) );
    printf("PBO created.\n");

    // This is the buffer we use to readback results into
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    createBuffers(w, h);
    imageW = w;
    imageH = h;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // check for hardware double precision support
    int dev = 0;
    cutGetCmdLineArgumenti(argc, (const char **) argv, "device", &dev);

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute capability %d.%d\n", deviceProp.major, deviceProp.minor);
    int version = deviceProp.major*10 + deviceProp.minor;

    // parse command line arguments
    bool bQAReadback = false;
    bool bFBODisplay = false;

    if (argc > 1) {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest")) {
            bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "fbo")) {
            bFBODisplay = true;
            fpsLimit = frameCheckNumber;
        }
    }
    
    imageW = 256;
    imageH = 256;

    printf("Initializing GLUT...\n");
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
    if (bFBODisplay) {
        if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 2.0\n");
            fprintf(stderr, "  GL_ARB_fragment_program\n");
            fprintf(stderr, "  GL_EXT_framebuffer_object\n");
            cleanup();
            exit(-1);
        }
    } else {
        if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 1.5\n");
            fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
            fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            cleanup();
            exit(-1);
        }
    }
    printf("OpenGL window created.\n");

    // Creating the Auto-Validation Code
    if (bQAReadback) {
        if (bFBODisplay) {
            g_CheckRender = new CheckFBO(imageW, imageH, 4);
        } else {
            g_CheckRender = new CheckBackBuffer(imageW, imageH, 4);
        }
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }

    printf("Starting GLUT main loop...\n");
    printf("\n");

    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutKeyboardFunc(keyboardFunc);
    glutMouseFunc(clickFunc);
    glutMotionFunc(motionFunc);
    glutReshapeFunc(reshapeFunc);

    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

    atexit(cleanup);

    glutMainLoop();

    CUT_EXIT(argc, argv);
}
