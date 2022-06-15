#ifndef OBJLIBRARY_H
#define OBJLIBRARY_H
#include <opencv2/opencv.hpp>

struct Object{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};



#endif // OBJLIBRARY_H
