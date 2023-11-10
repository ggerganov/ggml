#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "yolo-image.h"

static void draw_box(yolo_image & a, int x1, int y1, int x2, int y2, float r, float g, float b)
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

void draw_box_width(yolo_image & a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    for (int i = 0; i < w; ++i) {
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

bool save_image(const yolo_image & im, const char *name, int quality)
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

bool load_image(const char *fname, yolo_image & img)
{
    int w, h, c;
    uint8_t * data = stbi_load(fname, &w, &h, &c, 3);
    if (!data) {
        return false;
    }
    c = 3;
    img.w = w;
    img.h = h;
    img.c = c;
    img.data.resize(w*h*c);
    for (int k = 0; k < c; ++k){
        for (int j = 0; j < h; ++j){
            for (int i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                img.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    stbi_image_free(data);
    return true;
}

static yolo_image resize_image(const yolo_image & im, int w, int h)
{
    yolo_image resized(w, h, im.c);
    yolo_image part(w, im.h, im.c);
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

static void embed_image(const yolo_image & source, yolo_image & dest, int dx, int dy)
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

yolo_image letterbox_image(const yolo_image & im, int w, int h)
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
    yolo_image resized = resize_image(im, new_w, new_h);
    yolo_image boxed(w, h, im.c);
    boxed.fill(0.5);
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    return boxed;
}

static yolo_image tile_images(const yolo_image & a, const yolo_image & b, int dx)
{
    if (a.w == 0) {
        return b;
    }
    yolo_image c(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, a.c);
    c.fill(1.0f);
    embed_image(a, c, 0, 0);
    embed_image(b, c, a.w + dx, 0);
    return c;
}

static yolo_image border_image(const yolo_image & a, int border)
{
    yolo_image b(a.w + 2*border, a.h + 2*border, a.c);
    b.fill(1.0f);
    embed_image(a, b, border, border);
    return b;
}

yolo_image get_label(const std::vector<yolo_image> & alphabet, const std::string & label, int size)
{
    size = size/10;
    size = std::min(size, 7);
    yolo_image result(0,0,0);
    for (int i = 0; i < (int)label.size(); ++i) {
        int ch = label[i];
        yolo_image img = alphabet[size*128 + ch];
        result = tile_images(result, img, -size - 1 + (size+1)/2);
    }
    return border_image(result, (int)(result.h*.25));
}

void draw_label(yolo_image & im, int row, int col, const yolo_image & label, const float * rgb)
{
    int w = label.w;
    int h = label.h;
    if (row - h >= 0) {
        row = row - h;
    }
    for (int j = 0; j < h && j + row < im.h; j++) {
        for (int i = 0; i < w && i + col < im.w; i++) {
            for (int k = 0; k < label.c; k++) {
                float val = label.get_pixel(i, j, k);
                im.set_pixel(i + col, j + row, k, rgb[k] * val);
            }
        }
    }
}