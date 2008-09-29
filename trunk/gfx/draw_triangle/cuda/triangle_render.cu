
#include <stdio.h>
#include "cutil.h"
#include "triangle_kernel.h"

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
} // iDivUp

__global__ void renderTriangle(uchar4 *dst, const int imageW, const int imageH)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) 
    {
        int pixel = imageW * iy + ix;

        uchar4 color;

        // Hardcoded triangle renderer (draws the same picture as the
        // OCaml example).
        if (abs(iy-imageH/2) < (imageW-ix)/2)
        {
            color.x = 255;
            color.y = 0;
            color.z = 0;
            color.w = 0;
        } else
        {
            color.x = 255;
            color.y = 255;
            color.z = 255;
            color.w = 0;
        }

        dst[pixel] = color;
    }
}

// The host CPU Mandebrot thread spawner
void RunTriangleRender(uchar4 *dst, const int imageW, const int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    renderTriangle<<<grid, threads>>>(dst, imageW, imageH);

    CUT_CHECK_ERROR("Mandelbrot0_sm10 kernel execution failed.\n");
} // RunMandelbrot0
