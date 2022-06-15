#ifndef YOLOXINFERENCE_H
#define YOLOXINFERENCE_H
#include <iostream>
#include <string.h>
#include <ncnn/net.h>
#include <ncnn/layer.h>
#include <objLibrary.h>

using namespace std;
class yoloxInference
{
public:
    yoloxInference();
    ncnn::Net yolox;
    ncnn::Mat in;
    ncnn::Mat in_pad;
    float scale;
    int imgWidth;
    int imgHeight;
    int w;
    int h;

    void loadModel(string modelBin,string modelParam,cv::Mat src);
    std::vector<Object>* startInference(cv::Mat src);
    //void startInference(cv::Mat src);
    void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                                                    std::vector<GridAndStride> &grid_strides);
    void generate_yolox_proposals(std::vector<GridAndStride> grid_strides,
                                                    const ncnn::Mat &feat_blob, float prob_threshold,
                                                        std::vector<Object> &objects1);
    void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object> &objects1);
    void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold);

    float intersection_area(const Object &a, const Object &b);
    cv::Mat draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects1);
};

#endif // YOLOXINFERENCE_H
