#pragma once

#include <string>
#include <vector>
#include <cassert>

struct yolo_image {
    int w, h, c;
    std::vector<float> data;

    yolo_image() : w(0), h(0), c(0) {}
    yolo_image(int w, int h, int c) : w(w), h(h), c(c), data(w*h*c) {}

    float get_pixel(int x, int y, int c) const {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        return data[c*w*h + y*w + x];
    }

    void set_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] = val;
    }

    void add_pixel(int x, int y, int c, float val) {
        assert(x >= 0 && x < w && y >= 0 && y < h && c >= 0 && c < this->c);
        data[c*w*h + y*w + x] += val;
    }

    void fill(float val) {
        std::fill(data.begin(), data.end(), val);
    }
};

bool load_image(const char *fname, yolo_image & img);
void draw_box_width(yolo_image & a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
yolo_image letterbox_image(const yolo_image & im, int w, int h);
bool save_image(const yolo_image & im, const char *name, int quality);
yolo_image get_label(const std::vector<yolo_image> & alphabet, const std::string & label, int size);
void draw_label(yolo_image & im, int row, int col, const yolo_image & label, const float * rgb);
