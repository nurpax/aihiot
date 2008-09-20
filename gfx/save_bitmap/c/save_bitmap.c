
#include <stdio.h>
#include <string.h>

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef int int32;

static int exportPNG(const char* filename, 
                     int width, 
                     int height, 
                     const uint32* data)
{
    FILE*  fp;
    uint32 crcTbl[256];
    uint32 crc     = 0;
    int    bpp     = 4;
    int    idatLen = 6 + height * (6 + width * bpp);
    int    adler1  = 1;
    int    adler2  = 0;
    int    i;
    int    j;

    fp = fopen(filename, "wb");
    if (!fp)
        return 0;

    for (i = 0; i < 256; i++)
    {
        uint32 tmp = i;
        for (j = 0; j < 8; j++)
            tmp = (tmp >> 1) ^ (0xedb88320u & (~(tmp & 1) + 1));
        crcTbl[i] = tmp;
    }

#define CRC(VALUE)      crc = crcTbl[(crc ^ (uint32)(VALUE)) & 0xff] ^ (crc >> 8)
#define ADLER(VALUE)    adler1 = (adler1 + ((VALUE) & 0xff)) % 65521; adler2 = (adler2 + adler1) % 65521;
#define OUTC(VALUE)     fputc((VALUE), fp); CRC(VALUE)
#define OUTCA(VALUE)    fputc((VALUE), fp); CRC(VALUE); ADLER(VALUE)

    fprintf(fp, "\x89PNG\xd\xa\x1a\xa%c%c%c\xdIHDR", 0, 0, 0);

    crc = 0x575e51f5;
    OUTC(width >> 24); OUTC(width >> 16); OUTC(width >> 8); OUTC(width);
    OUTC(height >> 24); OUTC(height >> 16); OUTC(height >> 8); OUTC(height);

    OUTC(8); OUTC(6); OUTC(0); OUTC(0); OUTC(0);

    fprintf(fp, "%c%c%c%c%c%c%c%cIDAT\x78\x1",
        ~crc >> 24, ~crc >> 16, ~crc >> 8, ~crc,
        idatLen >> 24, idatLen >> 16, idatLen >> 8, idatLen);

    crc = 0x13e5812d;
    for (i = 0; i < height; i++)
    {
        int blockLen = width * bpp + 1;
        OUTC((i == height - 1) ? 1 : 0);
        OUTC(blockLen); OUTC(blockLen >> 8);
        OUTC(~blockLen); OUTC(~blockLen >> 8);

        OUTCA(0);
        for (j = 0; j < width; j++)
        {
            uint32 argb = data[i * width + j];
            OUTCA((uint8)argb);
            OUTCA((uint8)(argb >> 8));
            OUTCA((uint8)(argb >> 16));
            OUTCA((uint8)(argb >> 24));
        }
    }

    OUTC(adler2 >> 8); OUTC(adler2);
    OUTC(adler1 >> 8); OUTC(adler1);
    fprintf(fp, "%c%c%c%c%c%c%c%cIEND\xae\x42\x60\x82",
        ~crc >> 24, ~crc >> 16, ~crc >> 8, ~crc, 0, 0, 0, 0);

#undef CRC
#undef ADLER
#undef OUTC
#undef OUTCA
    fclose(fp);
    return 1;
}

int main()
{
    int x, y;
#define WIDTH  128
#define HEIGHT 128
    uint32 pixels[WIDTH*HEIGHT];

    /* Draw a couple of lines on the screen. */
    memset(pixels, 0xFF, sizeof(pixels));

    for (y = 0; y < HEIGHT; y++)
    {
        pixels[y*WIDTH+y] = (255<<24)|0xff;
        pixels[y*WIDTH+WIDTH-y-1] = (255<<24)|(0xff<<16);
        pixels[y*WIDTH+WIDTH/2] = (255<<24)|(0xff<<8);
    }

    printf("saving framebuffer to `screenshot.png'\n");
    /* Save our piece of art into screenshot.png */
    exportPNG("screenshot.png", WIDTH, HEIGHT, pixels);
}
