#include "yolov11_custom.h"

#include <algorithm>
#include <fstream>
#include <random>

#include "process/postprocess.h"
#include "utils/logging.h"

static std::vector<std::string> g_classes;

Yolov11Custom::Yolov11Custom(int core_id)
{
    engine_ = CreateRKNNEngine();
    // 线程池模式下会传入 0/1/2，让多个 RKNN 上下文分别跑在三个 NPU core 上。
    engine_->SetCoreId(core_id);
    input_tensor_.data = nullptr;
    ready_ = false;
    want_float_ = false;
}

Yolov11Custom::~Yolov11Custom()
{
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }

    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolov11Custom::LoadModel(const char *model_path)
{
    // LoadModelFile 会初始化 RKNN context，并查询输入输出张量属性。
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("YOLO11 load model file failed");
        return ret;
    }

    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("YOLO11 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }

    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);
    if (input_tensor_.data == nullptr)
    {
        NN_LOG_ERROR("malloc input tensor failed");
        return NN_RKNN_INPUT_ATTR_ERROR;
    }

    auto output_shapes = engine_->GetOutputShapes();
    // 官方优化版 YOLO11 通常是 9 输出：box/class/score_sum 各 3 个尺度。
    // 有些导出会去掉 score_sum，此时是 6 输出：box/class 各 3 个尺度。
    if (output_shapes.size() != 6 && output_shapes.size() != 9)
    {
        NN_LOG_ERROR("YOLO11 output tensor number should be 6 or 9, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    // float/float16 模型统一请求 float32 输出；int8 模型保留量化输出，后处理里按 zp/scale 反量化。
    want_float_ = output_shapes[0].type == NN_TENSOR_FLOAT16 || output_shapes[0].type == NN_TENSOR_FLOAT;
    if (want_float_)
    {
        NN_LOG_INFO("YOLO11 output requested as float32");
    }

    output_tensors_.clear();
    for (size_t i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr = output_shapes[i];
        tensor.attr.index = static_cast<uint32_t>(i);
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.size = tensor.attr.n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        if (tensor.data == nullptr)
        {
            NN_LOG_ERROR("malloc output tensor[%ld] failed", i);
            return NN_RKNN_OUTPUT_ATTR_ERROR;
        }
        output_tensors_.push_back(tensor);
    }

    ready_ = true;
    return NN_SUCCESS;
}

int Yolov11Custom::setStaticParams(float nms_threshold,
                                   float box_threshold,
                                   const std::string &model_labels_file_path,
                                   int obj_class_num)
{
    // 这些全局参数由后处理模块读取，线程池里每个实例使用同一套阈值和类别表。
    yolo::nmsThreshold = nms_threshold;
    yolo::objectThreshold = box_threshold;
    yolo::class_num = obj_class_num;

    std::ifstream labels(model_labels_file_path);
    if (!labels.is_open())
    {
        NN_LOG_ERROR("open label file failed: %s", model_labels_file_path.c_str());
        return -1;
    }

    g_classes.clear();
    std::string line;
    while (std::getline(labels, line) && static_cast<int>(g_classes.size()) < obj_class_num)
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        g_classes.push_back(line);
    }

    if (static_cast<int>(g_classes.size()) < obj_class_num)
    {
        NN_LOG_WARNING("label count %ld is less than class count %d", g_classes.size(), obj_class_num);
    }
    return 0;
}

nn_error_e Yolov11Custom::Preprocess(const cv::Mat &img, cv::Mat &image_letterbox)
{
    // 输入张量是 NHWC，dims[1]/dims[2] 对应模型高宽；当前模型为 640x640。
    float wh_ratio = static_cast<float>(input_tensor_.attr.dims[2]) /
                     static_cast<float>(input_tensor_.attr.dims[1]);
    letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
    cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    return NN_SUCCESS;
}

nn_error_e Yolov11Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

static cv::Rect DecodeLetterboxRect(const cv::Rect &box,
                                    const LetterBoxInfo &letterbox,
                                    int image_width,
                                    int image_height)
{
    // YOLO11 后处理得到的是 letterbox 图上的坐标，需要扣掉补边并裁剪到原图范围。
    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.width;
    int y2 = box.y + box.height;

    if (letterbox.hor)
    {
        x1 -= letterbox.pad;
        x2 -= letterbox.pad;
    }
    else
    {
        y1 -= letterbox.pad;
        y2 -= letterbox.pad;
    }

    x1 = std::max(0, std::min(x1, image_width - 1));
    y1 = std::max(0, std::min(y1, image_height - 1));
    x2 = std::max(0, std::min(x2, image_width));
    y2 = std::max(0, std::min(y2, image_height));

    return cv::Rect(cv::Point(x1, y1), cv::Point(std::max(x1 + 1, x2), std::max(y1 + 1, y2)));
}

nn_error_e Yolov11Custom::Postprocess(const cv::Mat &letterbox_img, std::vector<Detection> &objects)
{
    // GetYolo11DetectionResult 输出归一化坐标：class, score, xmin, ymin, xmax, ymax。
    std::vector<float> detection_rects;
    int ret = yolo::GetYolo11DetectionResult(output_tensors_, detection_rects);
    if (ret != 0)
    {
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    int img_width = letterbox_img.cols;
    int img_height = letterbox_img.rows;
    for (size_t i = 0; i + 5 < detection_rects.size(); i += 6)
    {
        int class_id = static_cast<int>(detection_rects[i + 0]);
        float conf = detection_rects[i + 1];
        int xmin = static_cast<int>(detection_rects[i + 2] * static_cast<float>(img_width) + 0.5f);
        int ymin = static_cast<int>(detection_rects[i + 3] * static_cast<float>(img_height) + 0.5f);
        int xmax = static_cast<int>(detection_rects[i + 4] * static_cast<float>(img_width) + 0.5f);
        int ymax = static_cast<int>(detection_rects[i + 5] * static_cast<float>(img_height) + 0.5f);

        if (xmax <= xmin || ymax <= ymin)
        {
            continue;
        }

        Detection result;
        result.class_id = class_id;
        result.confidence = conf;
        result.className = class_id >= 0 && class_id < static_cast<int>(g_classes.size())
                               ? g_classes[class_id]
                               : std::to_string(class_id);

        // 用 class_id 生成稳定颜色，同一类别每次运行颜色一致。
        std::mt19937 gen(class_id + 12345);
        std::uniform_int_distribution<int> dis(80, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        objects.push_back(result);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov11Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    if (!ready_)
    {
        NN_LOG_ERROR("YOLO11 model is not ready");
        return NN_RKNN_MODEL_NOT_LOAD;
    }

    cv::Mat image_letterbox;
    // 推理主流程：预处理 -> NPU 推理 -> 后处理 -> 原图坐标还原。
    auto ret = Preprocess(img, image_letterbox);
    if (ret != NN_SUCCESS)
    {
        return ret;
    }

    ret = Inference();
    if (ret != NN_SUCCESS)
    {
        return ret;
    }

    objects.clear();
    ret = Postprocess(image_letterbox, objects);
    if (ret != NN_SUCCESS)
    {
        return ret;
    }

    for (auto &obj : objects)
    {
        obj.box = DecodeLetterboxRect(obj.box, letterbox_info_, img.cols, img.rows);
    }

    return NN_SUCCESS;
}
