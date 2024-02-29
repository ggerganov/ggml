#pragma once

#include <string>
#include <vector>
#include <cassert>

typedef struct PointType
{
    float x;  /* col */
    float y;  /* row */
    float conf;

    PointType() = default;

    PointType(int x_, int y_, float c_)
            : x(x_), y(y_), conf(c_) {  }
} PointT;


struct superpoint_image {
    int w, h, c;
    std::vector<float> data;

    superpoint_image() : w(0), h(0), c(0) {}
    superpoint_image(int w, int h, int c) : w(w), h(h), c(c), data(w*h*c) {}

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

bool load_image(const char *fname, superpoint_image & img, bool be_grey);

void draw_point(superpoint_image & im, int row, int col, float r, float g, float b);


void load_data(int w, int h, int c, const uint8_t * data, std::vector<float>& img_data, bool is_grey);

void draw_box_width(superpoint_image & a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
superpoint_image letterbox_image(const superpoint_image & im, int w, int h);
bool save_image(const superpoint_image & im, const char *name, int quality);
superpoint_image get_label(const std::vector<superpoint_image> & alphabet, const std::string & label, int size);
void draw_label(superpoint_image & im, int row, int col, const superpoint_image & label, const float * rgb);
