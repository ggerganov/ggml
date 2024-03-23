#pragma once


#include "ggml/ggml.h"
#include "superpoint-image.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <functional>
#include <numeric>


static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}


static void write_array(const std::string& fname, const ggml_tensor * t)
{
    GGML_ASSERT(ggml_is_contiguous(t));
    int size = t->ne[0] * t->ne[1] * t->ne[2]  * t->ne[3];
    printf("write %lld data to  %s\n", size, fname.c_str());
        // Open a file for writing
    std::ofstream outfile(fname);
    float* data = ggml_get_data_f32(t);
    // Write the elements of the array to the file
    if (outfile.is_open()) {
        for (int i = 0; i < size; ++i)
        {
            float value = data[i];
            outfile << value << std::endl;
        }
        outfile.close();
    }
}

static void write_points(const std::string& fname, const std::vector<PointT>& pts)
{

    printf("write %d points to  %s\n", pts.size(), fname.c_str());
        // Open a file for writing
    std::ofstream outfile(fname);

    if (outfile.is_open()) {
        for (auto& pt: pts)
        {
            // float value = data[i];
            outfile << int(pt.x)<<"  " << int(pt.y)<<"  "<<pt.conf<< std::endl;
        }
        outfile.close();
    }
}

static void write_descriptors(const std::string& fname, const std::vector<std::vector<double>>& descs)
{

    printf("write %d descs to  %s\n", descs.size(), fname.c_str());
        // Open a file for writing
    std::ofstream outfile(fname);

    if (outfile.is_open()) {
        for (auto& desc: descs)
        {
            for(auto& digit: desc)
            {
                outfile << digit<<"  ";
            }
           outfile<< std::endl;
            // float value = data[i];
        }
        outfile.close();
    }
}

static void print_larger_number(struct ggml_tensor *input, float thre)
{
    // float thre = 0.015;
    int num = input->ne[0] * input->ne[1] * input->ne[2] * input->ne[3];
    float* data = ggml_get_data_f32(input);
    size_t cnt =0;

    for (size_t i = 0; i< num; i++)
    {
        float value = data[i];
        if (value > 0.015)
        {
            cnt++;
            printf("index %zu: %f\n", cnt, value);
        }
    }
}

static void print_data(struct ggml_tensor *input)
{
    int w = input->ne[0];
    int h = input->ne[1];
    int c = input->ne[2];
    printf("Shape:  %3d x %3d x %4d x %3d\n", w, h, c, (int)input->ne[3]);
    printf("nb:  %3d x %3d x %4d x %3d\n", input->nb[0], input->nb[1], input->nb[2], (int)input->nb[3]);

    int num = 10;
    if(input->type == GGML_TYPE_F16)
    {
        ggml_fp16_t* data = static_cast<ggml_fp16_t *>(ggml_get_data(input));
        // for (size_t i =0; i< w*h*c; i++)
        for (size_t i = 0; i< num; i++)
        {
            float value = ggml_fp16_to_fp32(data[i]);
            printf("index %zu: %f\n", i, value);
        }
    }
    else
    {
        float* data = ggml_get_data_f32(input);
        // for (size_t i =0; i< w*h*c; i++)
        for (size_t i = 0; i< 0+num; i++)
        {
            float value = data[i];
            printf("index %zu: %f\n", i, value);
        }
    }
}
