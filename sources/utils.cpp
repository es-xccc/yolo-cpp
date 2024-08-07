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
#include "utils.hpp"


vector<string>outputs_node= {"conv2d_59_convolution", "conv2d_67_convolution", "conv2d_75_convolution"};
vector<float> biases { 116,90,  156,198,  373,326, 30,61,  62,45,  59,119, 10,13,  16,30,  33,23};

vector<string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

image::image(int ww, int hh, int cc, float fill):w(ww),h(hh),c(cc){
    data = new float[h*w*c];
    for(int i = 0; i < h*w*c; ++i) data[i] = fill;
}

void image::free() {
    delete[] data;
}

void get_output(const int8_t* input, float scale, int channels, int fh, int width, float* output) {
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < fh; ++h) {
            for (int w = 0; w < width; ++w) {
                int inIdx = h * channels * width + w * channels + c;
                int outIdx = c * fh * width + h * width + w;
                output[outIdx] = input[inIdx] * scale;
            }
        }
    }
}

void set_input_image(DPUTask* task, const cv::Mat& img, const char* nodename) {
    int height = dpuGetInputTensorHeight(task, nodename);
    int width = dpuGetInputTensorWidth(task, nodename);
    int size = dpuGetInputTensorSize(task, nodename);
    int8_t* data = dpuGetInputTensorAddress(task, nodename);
    float scale = dpuGetInputTensorScale(task, nodename);

    image img_new = load_image_cv(img);
    image img_yolo = letterbox_image(img_new, width, height);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < 3; ++c) {
                int srcIdx = c * height * width + h * width + w;
                int dstIdx = h * width * 3 + w * 3 + c;
                
                float scaledValue = img_yolo.data[srcIdx] * scale;
                data[dstIdx] = (scaledValue < 0) ? 127 : static_cast<int8_t>(scaledValue);
            }
        }
    }

    img_new.free();
    img_yolo.free();
}


inline float sigmoid(float p) {
    return 1.0 / (1.0 + exp(-p));
}

inline float overlap(float x1, float w1, float x2, float w2) {
    return max(0.0, min(x1 + w1/2.0, x2 + w2/2.0) - max(x1 - w1/2.0, x2 - w2/2.0));
}

inline float cal_iou(vector<float> bbox, vector<float>truth) {
    float w = overlap(bbox[0], bbox[2], truth[0], truth[2]);
    float h = overlap(bbox[1], bbox[3], truth[1], truth[3]);
    if(w < 0 || h < 0) return 0;
    float inter_area = w * h;
    float union_area = bbox[2] * bbox[3] + truth[2] * truth[3] - inter_area;
    return inter_area * 1.0 / union_area;
}

vector<vector<float>> nms(vector<vector<float>>& bboxs, int classes, float threshold) {
    vector<vector<float>> result;
    
    for (int k = 0; k < classes; ++k) {
        auto compareScores = [k](const vector<float>& a, const vector<float>& b) {
            return a[5 + k] > b[5 + k];
        };
        
        for (auto& bbox : bboxs) {
            bbox[4] = k;
        }
        
        sort(bboxs.begin(), bboxs.end(), compareScores);
        
        for (const auto& bbox : bboxs) {
            if (bbox[5 + k] < 0.5) break;
            
            result.push_back(bbox);
            
            bboxs.erase(
                remove_if(bboxs.begin() + (&bbox - &bboxs[0]) + 1, bboxs.end(),
                    [&bbox, threshold](const vector<float>& other) {
                        return cal_iou(other, bbox) >= threshold;
                    }),
                bboxs.end()
            );
        }
    }
    
    return result;
}


static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image resize_image(image im, int w, int h)
{
    image resized(w, h, im.c,0);   
    image part(w, im.h, im.c,0);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }
    part.free();
    return resized;
}

image load_image_cv(const cv::Mat &img) {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    image im = make_image(w, h, c);

    unsigned char *data = img.data;

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            im.data[0 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 2] * 0.00390625;
            im.data[1 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 1] * 0.00390625;
            im.data[2 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 0] * 0.00390625;
        }
    }
    return im;
}

image letterbox_image(image im, int w, int h)
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
    image resized = resize_image(im, new_w, new_h);
    image boxed(w, h, im.c, .5);
    
    int dx = (w-new_w)/2;
    int dy = (h-new_h)/2;
    for(int k = 0; k < resized.c; ++k){
        for(int y = 0; y < new_h; ++y){
            for(int x = 0; x < new_w; ++x){
                float val = get_pixel(resized, x,y,k);
                set_pixel(boxed, dx+x, dy+y, k, val);
            }
        }
    }
    resized.free();
    return boxed;
}

void resize_coor(vector<vector<float>>& bboxs, int n,
    int w, int h, int netw, int neth, int relative = 0) {
    int new_w=0;
    int new_h=0;

    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (int i = 0; i < n; ++i){
        bboxs[i][0] =  (bboxs[i][0] - (netw - new_w)/2./netw) / ((float)new_w/(float)netw);
        bboxs[i][1] =  (bboxs[i][1] - (neth - new_h)/2./neth) / ((float)new_h/(float)neth);
        bboxs[i][2] *= (float)netw/new_w;
        bboxs[i][3] *= (float)neth/new_h;
    }
}

void save_to_txt(const string& class_name, int cls, float xmin, float ymin, float xmax, float ymax) {
    chrono::steady_clock::time_point current_time = chrono::steady_clock::now();
    chrono::duration<double> elapsed = current_time - start_total_time;
    
    ofstream file("../output.txt", ios::app);
    if (file.is_open()) {
        file << elapsed.count() << "," << class_name << "," << xmin << "," << ymin << "," << xmax << "," << ymax << "\n";
        file.close();
    } else {
        cerr << "Unable to open file";
    }
}

void deal(DPUTask* task, cv::Mat& img, int sw, int sh)
{
    vector<vector<float>> bboxs;
    for (size_t i = 0; i < outputs_node.size(); i++) {
        string output_node = outputs_node[i];
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());

        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());

        vector<float> result(sizeOut);
        bboxs.reserve(sizeOut);
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);
        detect(bboxs, result, channel, height, width, i, sh, sw);
    }
    resize_coor(bboxs, bboxs.size(), img.cols, img.rows, sw, sh);
    vector<vector<float>> coor = nms(bboxs, classification, 0.2);

    float h = img.rows;
    float w = img.cols;
    for (size_t i = 0; i < coor.size(); ++i) {
        float xleft = (coor[i][0] - coor[i][2] / 2.0) * w;
        float yup = (coor[i][1] - coor[i][3] / 2.0) * h;
        float xright = (coor[i][0] + coor[i][2] / 2.0) * w;
        float ydown = (coor[i][1] + coor[i][3] / 2.0) * h;
        int classIdx = static_cast<int>(coor[i][4]);

        saveToTxt(class_names[classIdx], xleft, yup, xright, ydown);
    }

}

void detect(vector<vector<float>> &bboxs, vector<float> result, int channel, int fh, int fw, int num, int sh, int sw) 
{
    float map[fh * fw][3][85];
    for (int i = 0; i < fh * fw; ++i) {
        for (int c = 0; c < channel; ++c) {
            map[i][c / 85][c % 85] = result[c * fh * fw + i];
        }
    }

    for (int y = 0; y < fh; ++y) {
        for (int x = 0; x < fw; ++x) {
            for (int c = 0; c < 3; ++c) {
                float class_conf = sigmoid(map[y * fw + x][c][4]);
                if (class_conf < 0.5)
                    continue;
                vector<float> bbox;
                int idx = y * fw + x;
                float *feat = map[c][idx];
                vector<float> bbox = {
                    (x + sigmoid(feat[0])) / fw,
                    (y + sigmoid(feat[1])) / fh,
                    exp(feat[2]) * biases[2 * c + 3 * 2 * num] / sw,
                    exp(feat[3]) * biases[2 * c + 3 * 2 * num + 1] / sh,
                    -1
                };

                for (int p = 0; p < classification; p++) {
                    bbox.push_back(class_conf * sigmoid(feat[5 + p]));
                }
                bboxs.push_back(bbox);
            }
        }
    }
}