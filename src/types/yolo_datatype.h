

#ifndef RK3588_DEMO_NN_DATATYPE_H
#define RK3588_DEMO_NN_DATATYPE_H

#include <opencv2/opencv.hpp>
#include <string>

typedef struct _nn_object_s {
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} nn_object_s;

struct Detection
{
    int class_id{-1};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
    cv::Point point1{};
    cv::Point point2{};
    cv::Point point3{};
    cv::Point point4{};

    /// 用于obb跟踪算法
    // x,y是中心坐标，w,h是宽高
    float x;
    float y;
    float w;
    float h;
    float angle;
};

#endif //RK3588_DEMO_NN_DATATYPE_H
