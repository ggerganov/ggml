#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include "superpoint-image.h"



static void draw_box(superpoint_image & a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w-1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w-1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h-1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h-1;

    for (int i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for (int i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width(superpoint_image & a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    for (int i = 0; i < w; ++i) {
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

bool save_image(const superpoint_image & im, const char *name, int quality)
{
    uint8_t *data = (uint8_t*)calloc(im.w*im.h*im.c, sizeof(uint8_t));
    for (int k = 0; k < im.c; ++k) {
        for (int i = 0; i < im.w*im.h; ++i) {
            data[i*im.c+k] = (uint8_t) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_jpg(name, im.w, im.h, im.c, data, quality);
    free(data);
    if (!success) {
        fprintf(stderr, "Failed to write image %s\n", name);
        return false;
    }
    return true;
}

/**
 * force image to be in grayscale
*/
void load_data(int w, int h, int c, const uint8_t * data, std::vector<float>& img_data, bool is_grey)
{
    if(is_grey)
    {
        img_data.resize(w*h);
    }
    else
    {
        img_data.resize(c*w*h);

    }


    if (c ==1)
    {
        for (int k = 0; k < c; ++k){
            for (int j = 0; j < h; ++j){
                for (int i = 0; i < w; ++i){
                    //rgb, I guess
                    int dst_index = i + w*j + w*h*k;
                    int src_index = k + c*i + c*w*j;
                    img_data[dst_index] = (float)data[src_index]/255.;
                }
            }
        }
    }
    else if (c == 3)
    {
        if(is_grey)
        {
            for (int j = 0; j < h; ++j)
            {
                for (int i = 0; i < w; ++i)
                {
                    //rgb, I guess
                    int dst_index = i + w*j;
                    // int src_index = k + c*i + c*w*j;
                    int red  =  0 + 3 * i + 3 * w * j;
                    int green = 1 + 3 * i + 3 * w * j;
                    int blue  = 2 + 3 * i + 3 * w * j;

                    float grey = 0.299 * data[red] + 0.587 * data[green] + 0.114 * data[blue];
                    // float grey = 0.333 * data[red] + 0.333 * data[green] + 0.333 * data[blue];

                    img_data[dst_index] = grey/255.;

                }
            }

        }
        else
        {
            for (int k = 0; k < c; ++k){
                for (int j = 0; j < h; ++j){
                    for (int i = 0; i < w; ++i){
                        int dst_index = i + w*j + w*h*k;
                        int src_index = k + c*i + c*w*j;
                        img_data[dst_index] = (float)data[src_index]/255.;
                    }
                }
            }
        }



    }
}

/*In superpoint, image is read as grey and normalized to 0 -- 1*/
bool load_image(const char *fname, superpoint_image & img, bool be_grey)
{
    //assert channel is 3
    int w, h, c;
    uint8_t * data = nullptr;
    if(be_grey)
    {
        data = stbi_load(fname, &w, &h, &c, 1);
    }
    else
    {
        data = stbi_load(fname, &w, &h, &c, 3);
    }
    img.h = h;
    img.w = w;
    img.c = be_grey?1:3;

    if(c == 1)
    {
        printf("load grey image\n");
    }
    else if (c == 3)
    {
        printf("load RGB image\n");

    }
    if (!data) {
        return false;
    }
    if(c == 3 && be_grey)
    {
        load_data(w,h,1,data, img.data, be_grey);
    }
    else if (c ==3 &&(!be_grey))
    {
        load_data(w,h,3,data, img.data, be_grey);

    }

    stbi_image_free(data);
    return true;
}

/*
https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
*/


static superpoint_image resize_image(const superpoint_image & im, int w, int h)
{
    superpoint_image resized(w, h, im.c);
    superpoint_image part(w, im.h, im.c);
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (int k = 0; k < im.c; ++k){
        for (int r = 0; r < im.h; ++r) {
            for (int c = 0; c < w; ++c) {
                float val = 0;
                if (c == w-1 || im.w == 1){
                    val = im.get_pixel(im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.get_pixel(ix, r, k) + dx * im.get_pixel(ix+1, r, k);
                }
                part.set_pixel(c, r, k, val);
            }
        }
    }
    for (int k = 0; k < im.c; ++k){
        for (int r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for (int c = 0; c < w; ++c){
                float val = (1-dy) * part.get_pixel(c, iy, k);
                resized.set_pixel(c, r, k, val);
            }
            if (r == h-1 || im.h == 1) continue;
            for (int c = 0; c < w; ++c){
                float val = dy * part.get_pixel(c, iy+1, k);
                resized.add_pixel(c, r, k, val);
            }
        }
    }
    return resized;
}

static void embed_image(const superpoint_image & source, superpoint_image & dest, int dx, int dy)
{
    for (int k = 0; k < source.c; ++k) {
        for (int y = 0; y < source.h; ++y) {
            for (int x = 0; x < source.w; ++x) {
                float val = source.get_pixel(x, y, k);
                dest.set_pixel(dx+x, dy+y, k, val);
            }
        }
    }
}

superpoint_image letterbox_image(const superpoint_image & im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    superpoint_image resized = resize_image(im, new_w, new_h);
    superpoint_image boxed(w, h, im.c);
    boxed.fill(0.5);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    return boxed;
}

static superpoint_image tile_images(const superpoint_image & a, const superpoint_image & b, int dx)
{
    if (a.w == 0) {
        return b;
    }
    superpoint_image c(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, a.c);
    c.fill(1.0f);
    embed_image(a, c, 0, 0);
    embed_image(b, c, a.w + dx, 0);
    return c;
}

static superpoint_image border_image(const superpoint_image & a, int border)
{
    superpoint_image b(a.w + 2*border, a.h + 2*border, a.c);
    b.fill(1.0f);
    embed_image(a, b, border, border);
    return b;
}

superpoint_image get_label(const std::vector<superpoint_image> & alphabet, const std::string & label, int size)
{
    size = size/10;
    size = std::min(size, 7);
    superpoint_image result(0,0,0);
    for (int i = 0; i < (int)label.size(); ++i) {
        int ch = label[i];
        superpoint_image img = alphabet[size*128 + ch];
        result = tile_images(result, img, -size - 1 + (size+1)/2);
    }
    return border_image(result, (int)(result.h*.25));
}

void draw_point(superpoint_image & im, int y, int x, float r, float g, float b)
{
    for(int row = y -2; row<y+2;row++)
        for(int col = x -2; col<x+2;col++)

    {
        im.data[col + row*im.w + 0*im.w*im.h] = r;

        im.data[col + row*im.w + 1*im.w*im.h] = g;

        im.data[col + row*im.w + 2*im.w*im.h] = b;
        // im.set_pixel(col, row, k, rgb[k] * val);
    }
}