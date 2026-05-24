#include <opencv2/opencv.hpp>

#include "draw/cv_draw.h"
#include "task/yolov11_custom.h"
#include "utils/logging.h"

int main(int argc, char **argv)
{
    // 最少需要传入模型路径和图片路径；其余参数都有默认值，方便快速验证模型。
    if (argc < 3)
    {
        printf("Usage: %s <yolo11.rknn> <image_path> [labels_path] [class_num] [box_thresh] [nms_thresh]\n", argv[0]);
        return -1;
    }

    const char *model_file = argv[1];
    const char *img_file = argv[2];
    const std::string labels_path = argc > 3 ? argv[3] : "coco_80_labels_list.txt";
    const int class_num = argc > 4 ? atoi(argv[4]) : 80;
    const float box_thresh = argc > 5 ? static_cast<float>(atof(argv[5])) : 0.25f;
    const float nms_thresh = argc > 6 ? static_cast<float>(atof(argv[6])) : 0.45f;

    cv::Mat img = cv::imread(img_file);
    if (img.empty())
    {
        NN_LOG_ERROR("failed to read image: %s", img_file);
        return -1;
    }

    // 单图 demo 只创建一个 RKNN 上下文，不绑定固定 NPU core。
    Yolov11Custom yolo;
    auto ret = yolo.LoadModel(model_file);
    if (ret != NN_SUCCESS)
    {
        return ret;
    }
    // 阈值和类别文件在后处理阶段使用，COCO 模型默认 80 类。
    yolo.setStaticParams(nms_thresh, box_thresh, labels_path, class_num);

    std::vector<Detection> objects;
    // Run 内部完成：letterbox 预处理 -> RKNN 推理 -> YOLO11 DFL 后处理 -> 坐标还原。
    ret = yolo.Run(img, objects);
    if (ret != NN_SUCCESS)
    {
        return ret;
    }

    DrawDetections(img, objects);
    cv::imwrite("result.jpg", img);
    NN_LOG_INFO("result saved to result.jpg, detect num: %ld", objects.size());

    return 0;
}
