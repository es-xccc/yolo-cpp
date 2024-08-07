/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <math.h>
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <dnndk/n2cube.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

//dpu kernel info
#define YOLOKERNEL "yolo"
#define INPUTNODE "conv2d_1_convolution"
// ANSI escape codes for text colors

extern vector<string> outputs_node;
extern vector<float> biases;
extern vector<string> class_names;
extern chrono::steady_clock::time_point start_total_time;

class image {
public:
    int w;
    int h;
    int c;
    float *data;
    image(int ww, int hh, int cc, float fill);
    void free();
};

void get_output(const int8_t* input, float scale, int channels, int height, int width, float* output);
void set_input_image(DPUTask* task, const cv::Mat& img, const char* nodename);
inline float sigmoid(float p);
inline float overlap(float x1, float w1, float x2, float w2);
inline float cal_iou(vector<float> box, vector<float>truth);
vector<vector<float>> nms(vector<vector<float>>& boxes, int classes, float threshold);
static float get_pixel(image m, int x, int y, int c);
static void set_pixel(image m, int x, int y, int c, float val);
static void add_pixel(image m, int x, int y, int c, float val);
image resize_image(image im, int w, int h);
image load_image_cv(const cv::Mat& img);
image letterbox_image(image im, int w, int h);
void resize_coor(vector<vector<float>>& boxes, int n, int w, int h, int netw, int neth);
void save_to_txt(const string& class_name, int cls, float xmin, float ymin, float xmax, float ymax);
void deal(DPUTask* task, cv::Mat& img, int sw, int sh);
void detect(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int width, int num, int sh, int sw);

#endif