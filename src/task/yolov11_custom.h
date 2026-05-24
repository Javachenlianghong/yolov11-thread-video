#ifndef RK3588_DEMO_YOLOV11_CUSTOM_H
#define RK3588_DEMO_YOLOV11_CUSTOM_H

#include "engine/engine.h"
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

#include <memory>
#include <opencv2/opencv.hpp>

class Yolov11Custom
{
public:
    // core_id 为 0/1/2 时绑定到指定 NPU core；-1 使用 RKNN 自动调度。
    explicit Yolov11Custom(int core_id = -1);
    ~Yolov11Custom();

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);
    int setStaticParams(float nms_threshold,
                        float box_threshold,
                        const std::string &model_labels_file_path,
                        int obj_class_num);

private:
    // 预处理和后处理拆开，便于图片 demo 和线程池复用同一套推理流程。
    nn_error_e Preprocess(const cv::Mat &img, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &letterbox_img, std::vector<Detection> &objects);

    // 记录 letterbox 补边信息，后处理后要用它把坐标还原到原图。
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool ready_;
    bool want_float_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV11_CUSTOM_H
