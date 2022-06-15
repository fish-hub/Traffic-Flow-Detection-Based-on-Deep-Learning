﻿#include "yoloxinference.h"
#define YOLOX_TARGET_SIZE 640
#define YOLOX_CONF_THRESH 0.55
#define YOLOX_NMS_THRESH  0.25
#include<QDebug>
#include <ncnn/gpu.h>
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloV5Focus)



yoloxInference::yoloxInference()
{
//printf("init yolox inference\n");
}

void yoloxInference::loadModel(string modelBin, string modelParam, cv::Mat src)
{

    //ncnn::Net yolox;
    //ncnn::create_gpu_instance();
    yolox.opt.use_vulkan_compute = true;
    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolox.load_param(modelParam.c_str());
    yolox.load_model(modelBin.c_str());
    imgWidth = src.cols;
    imgHeight = src.rows;
    w = imgWidth;
    h = imgHeight;
    scale = 1.f;
    //将图像整体缩放至最长边小于640
    if (w > h)
    {
        scale = (float)YOLOX_TARGET_SIZE / w;
        w = YOLOX_TARGET_SIZE;
        h = h * scale;
    }
    else
    {
        scale = (float)YOLOX_TARGET_SIZE / h;
        h = YOLOX_TARGET_SIZE;
        w = w * scale;
    }
}

std::vector<Object>* yoloxInference::startInference(cv::Mat src)
{

    cv::Mat inputMat = src;

    in = ncnn::Mat::from_pixels_resize(inputMat.data, ncnn::Mat::PIXEL_BGR, src.cols, src.rows, w, h);

    ncnn::copy_make_border(in, in_pad, 0, 280, 0, 0, ncnn::BORDER_CONSTANT, 114.f);

    ncnn::Extractor ex = yolox.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(8);
    ex.input("images",in_pad);
    ncnn::Mat out;
    ex.extract("output",out);

    static const int stride_arr[3] = {8,16,32};
    std::vector<int>strides(stride_arr,stride_arr + sizeof(stride_arr) / sizeof(stride_arr[0]));
    std::vector<GridAndStride>grid_strides;
    generate_grids_and_stride(YOLOX_TARGET_SIZE, strides, grid_strides);
    //解析推理的输出，将每个置信度大于阈值x类别概率的候选框保存到proposals中
    std::vector<Object>proposals;
    generate_yolox_proposals(grid_strides,out,YOLOX_CONF_THRESH,proposals);
    //对筛选出的候选框进行排序
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    std::vector<Object> objects;
    nms_sorted_bboxes(proposals, picked, YOLOX_NMS_THRESH);
    int count = picked.size();
    objects.resize(count);
    for(int i=0;i<count;i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x)/scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        x0 = std::max(std::min(x0, (float)(imgWidth - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(imgHeight - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(imgWidth - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(imgHeight - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    std::vector<Object>* objs = new std::vector<Object>;

    *objs = objects;
    return objs;
}


void yoloxInference::generate_grids_and_stride(const int target_size, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
{
    for(int i=0;i<(int)strides.size();i++)
    {
        int stride = strides[i];
        //计算每个不同尺度的输出特征图对应的栅格数
        int num_grid = target_size/stride; //80,40,20
        //保存三个尺度下特征图上每个点的坐标和相对应的步长
        for(int g1=0;g1<num_grid;g1++)
        {
            for(int g0=0;g0<num_grid;g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void yoloxInference::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &feat_blob, float prob_threshold, std::vector<Object> &objects1)
{

    const int num_grid = feat_blob.h; //8400=80x80+40x40+20x20
    const int num_class = feat_blob.w-5; //coco有80各类别

    const int num_anchors = grid_strides.size(); //80x80+40x40+20x20
    //qDebug()<<num_grid;
    const float *feat_ptr = feat_blob.channel(0);
    //std::cout<<feat_blob.channel(0)<<std::endl;
    //遍历8400个anchor

    for(int anchor_idx=0;anchor_idx<num_anchors;anchor_idx++)
    {
        //每个anchor的位置，在20x20,40x40,80x80中的
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        //anchor所对应特征图是下载样多少倍的（8,16,32）
        const int stride = grid_strides[anchor_idx].stride;

        //每个anchor在原图中对应的中心 0,1,2,3是anchor的x,y,w,h相对偏移 anchorbased计算方式
        //预测结果是tx,ty,tw,th，与基于anchor回归的损失函数一样，相当于是不同尺寸下采样的特征图（8,16,32）的大小作为anchorbased 的pre anchor大小
        //x = wa*dx+xa-->wa=stride=8/16/32 xa=anchor=feat_ptr[0]*stride中心在原图的位置
        float x_center = (feat_ptr[0]+grid0)*stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2])*stride;
        float h = exp(feat_ptr[3]) * stride;

        //得到候选框的左上角坐标
        float x0 = x_center-w*0.5f;
        float y0 = y_center-h*0.5f;
        //获取预测框置信度
        float box_objectness = feat_ptr[4];
        for(int class_idx=0;class_idx<num_class;class_idx++)
        {
             if(class_idx!=2&&class_idx!=5&&class_idx!=7)
                 continue;
            //每个类别的概率
            float box_cls_score = feat_ptr[5+class_idx];
            float box_prob = box_objectness*box_cls_score;

            if(box_prob>prob_threshold)
            {

                Object obj;

                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;
                objects1.push_back(obj);

            }

        }
        feat_ptr += feat_blob.w;
    }

}

float yoloxInference::intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


void yoloxInference::nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    //当前图片检测到的目标个数（大于得分阈值）
    const int n = faceobjects.size();
    std::vector<float>areas(n);
    //得到每个候选框的面积
    for(int i=0;i<n;i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
    //便利每个候选框
    for(int i=0;i<n;i++)
    {
        const Object& a = faceobjects[i];
        int keep=1;
        //与已经存储的候选框进行nms比较抑制
        for(int j=0;j<(int)picked.size();j++)
        {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if(inter_area/union_area>nms_threshold)
                keep=0;
        }
        if(keep)
           picked.push_back(i);
    }
}


void yoloxInference::qsort_descent_inplace(std::vector<Object> &objects1)
{
    if(objects1.empty())
        return;
    qsort_descent_inplace(objects1,0,objects1.size()-1);


}

void yoloxInference::qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while(i<=j)
    {
        while(faceobjects[i].prob>p)
            i++;
        while(faceobjects[j].prob<p)
            j--;
        if(i<=j)
        {
            std::swap(faceobjects[i],faceobjects[j]);
            i++;
            j--;
        }
    }
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}



cv::Mat yoloxInference::draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects1)
{
    static const char* class_names[] = {
         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
         "hair drier", "toothbrush"
     };

     cv::Mat image = bgr.clone();

     for (size_t i = 0; i < objects1.size(); i++)
     {
         const Object& obj = objects1[i];

         //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
          //       obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

         cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0),2);

         char text[256];
         sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

         int baseLine = 0;
         cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

         int x = obj.rect.x;
         int y = obj.rect.y - label_size.height - baseLine;
         if (y < 0)
             y = 0;
         if (x + label_size.width > image.cols)
             x = image.cols - label_size.width;

         cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                       cv::Scalar(255, 255, 255), -1);
         float fontSize=0.5;
         if(image.cols*image.rows>1000000)
             float size = fontSize=0.7;
         cv::putText(image, text, cv::Point(x, y + label_size.height),
                     cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 0, 255),2);
     }
    return image;
}
