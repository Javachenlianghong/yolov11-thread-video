#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "utils/logging.h"

namespace yolo
{
    struct DetectRect
    {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float score;
        int classId;
    };

    float objectThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    int class_num = 80;

    static const int kHeadNum = 3;
    static const int kInputW = 640;
    static const int kInputH = 640;
    // 官方 YOLO11 优化模型固定三层输出，对应 80x80、40x40、20x20 三个尺度。
    static const int kStrides[kHeadNum] = {8, 16, 32};

    static inline float Dequant(const tensor_data_s &tensor, int offset)
    {
        switch (tensor.attr.type)
        {
        case NN_TENSOR_FLOAT:
            return reinterpret_cast<float *>(tensor.data)[offset];
        case NN_TENSOR_FLOAT16:
            NN_LOG_ERROR("float16 output should be requested as float32 by want_float");
            return 0.0f;
        case NN_TENSOR_UINT8:
            return (static_cast<float>(reinterpret_cast<uint8_t *>(tensor.data)[offset]) -
                    static_cast<float>(tensor.attr.zp)) *
                   tensor.attr.scale;
        case NN_TENSOR_INT8:
            return (static_cast<float>(reinterpret_cast<int8_t *>(tensor.data)[offset]) -
                    static_cast<float>(tensor.attr.zp)) *
                   tensor.attr.scale;
        default:
            NN_LOG_ERROR("unsupported output tensor type: %d", tensor.attr.type);
            return 0.0f;
        }
    }

    static inline int8_t QuantizeThresholdI8(float value, int32_t zp, float scale)
    {
        if (scale <= 0.0f)
        {
            return std::numeric_limits<int8_t>::max();
        }

        int q = static_cast<int>(std::round(value / scale + static_cast<float>(zp)));
        q = std::max(static_cast<int>(std::numeric_limits<int8_t>::min()),
                     std::min(static_cast<int>(std::numeric_limits<int8_t>::max()), q));
        return static_cast<int8_t>(q);
    }

    static inline uint8_t QuantizeThresholdU8(float value, int32_t zp, float scale)
    {
        if (scale <= 0.0f)
        {
            return std::numeric_limits<uint8_t>::max();
        }

        int q = static_cast<int>(std::round(value / scale + static_cast<float>(zp)));
        q = std::max(static_cast<int>(std::numeric_limits<uint8_t>::min()),
                     std::min(static_cast<int>(std::numeric_limits<uint8_t>::max()), q));
        return static_cast<uint8_t>(q);
    }

    static inline bool TensorValuePassThreshold(const tensor_data_s &tensor, int offset, float threshold)
    {
        // score_sum 只有 1 个通道，先用它过滤低分网格点，避免无效的 80 类扫描和 DFL 解码。
        switch (tensor.attr.type)
        {
        case NN_TENSOR_FLOAT:
            return reinterpret_cast<float *>(tensor.data)[offset] >= threshold;
        case NN_TENSOR_UINT8:
            return reinterpret_cast<uint8_t *>(tensor.data)[offset] >=
                   QuantizeThresholdU8(threshold, tensor.attr.zp, tensor.attr.scale);
        case NN_TENSOR_INT8:
            return reinterpret_cast<int8_t *>(tensor.data)[offset] >=
                   QuantizeThresholdI8(threshold, tensor.attr.zp, tensor.attr.scale);
        default:
            return Dequant(tensor, offset) >= threshold;
        }
    }

    static inline bool GetTensorShape(const tensor_data_s &tensor, int &channels, int &height, int &width, bool &nchw)
    {
        if (tensor.attr.n_dims != 4)
        {
            NN_LOG_ERROR("unsupported YOLO11 output dims: %d", tensor.attr.n_dims);
            return false;
        }

        if (tensor.attr.layout == NN_TENSOR_NCHW)
        {
            channels = tensor.attr.dims[1];
            height = tensor.attr.dims[2];
            width = tensor.attr.dims[3];
            nchw = true;
            return true;
        }

        if (tensor.attr.layout == NN_TENSOR_NHWC)
        {
            height = tensor.attr.dims[1];
            width = tensor.attr.dims[2];
            channels = tensor.attr.dims[3];
            nchw = false;
            return true;
        }

        if (tensor.attr.dims[2] == tensor.attr.dims[3])
        {
            channels = tensor.attr.dims[1];
            height = tensor.attr.dims[2];
            width = tensor.attr.dims[3];
            nchw = true;
        }
        else
        {
            height = tensor.attr.dims[1];
            width = tensor.attr.dims[2];
            channels = tensor.attr.dims[3];
            nchw = false;
        }
        return true;
    }

    static inline int TensorOffset(const tensor_data_s &tensor,
                                   bool nchw,
                                   int channels,
                                   int height,
                                   int width,
                                   int channel,
                                   int y,
                                   int x)
    {
        (void)tensor;
        if (nchw)
        {
            return channel * height * width + y * width + x;
        }
        return (y * width + x) * channels + channel;
    }

    static inline float IOU(const DetectRect &a, const DetectRect &b)
    {
        float xmin = std::max(a.xmin, b.xmin);
        float ymin = std::max(a.ymin, b.ymin);
        float xmax = std::min(a.xmax, b.xmax);
        float ymax = std::min(a.ymax, b.ymax);

        float inter_w = std::max(0.0f, xmax - xmin);
        float inter_h = std::max(0.0f, ymax - ymin);
        float inter = inter_w * inter_h;

        float area_a = std::max(0.0f, a.xmax - a.xmin) * std::max(0.0f, a.ymax - a.ymin);
        float area_b = std::max(0.0f, b.xmax - b.xmin) * std::max(0.0f, b.ymax - b.ymin);
        float total = area_a + area_b - inter;
        return total <= 0.0f ? 0.0f : inter / total;
    }

    static void ComputeDfl(const float *tensor, int dfl_len, float *box)
    {
        // YOLO11 的 box 分支是 4*dfl_len 通道，用 DFL softmax 求四个边距的期望值。
        for (int side = 0; side < 4; side++)
        {
            const float *side_ptr = tensor + side * dfl_len;
            float max_value = side_ptr[0];
            for (int i = 1; i < dfl_len; i++)
            {
                max_value = std::max(max_value, side_ptr[i]);
            }

            float exp_sum = 0.0f;
            float acc_sum = 0.0f;
            for (int i = 0; i < dfl_len; i++)
            {
                float exp_value = std::exp(side_ptr[i] - max_value);
                exp_sum += exp_value;
                acc_sum += exp_value * static_cast<float>(i);
            }
            box[side] = exp_sum <= 0.0f ? 0.0f : acc_sum / exp_sum;
        }
    }

    static bool FindMaxClassScore(const tensor_data_s &score_tensor,
                                  bool score_nchw,
                                  int score_channels,
                                  int score_h,
                                  int score_w,
                                  int y,
                                  int x,
                                  int cls_count,
                                  int &class_id,
                                  float &max_score)
    {
        // 量化模型先在原始 int8/uint8 值域比较，最后只反量化最大值，降低 CPU 后处理开销。
        class_id = -1;
        max_score = 0.0f;

        if (score_tensor.attr.type == NN_TENSOR_INT8)
        {
            const int8_t threshold = QuantizeThresholdI8(objectThreshold, score_tensor.attr.zp, score_tensor.attr.scale);
            const int8_t *scores = reinterpret_cast<int8_t *>(score_tensor.data);
            int8_t max_raw = std::numeric_limits<int8_t>::min();
            int max_offset = -1;

            for (int c = 0; c < cls_count; c++)
            {
                int offset = TensorOffset(score_tensor, score_nchw, score_channels, score_h, score_w, c, y, x);
                int8_t score = scores[offset];
                if (score > threshold && score > max_raw)
                {
                    max_raw = score;
                    max_offset = offset;
                    class_id = c;
                }
            }

            if (class_id >= 0)
            {
                max_score = Dequant(score_tensor, max_offset);
            }
            return class_id >= 0;
        }

        if (score_tensor.attr.type == NN_TENSOR_UINT8)
        {
            const uint8_t threshold = QuantizeThresholdU8(objectThreshold, score_tensor.attr.zp, score_tensor.attr.scale);
            const uint8_t *scores = reinterpret_cast<uint8_t *>(score_tensor.data);
            uint8_t max_raw = std::numeric_limits<uint8_t>::min();
            int max_offset = -1;

            for (int c = 0; c < cls_count; c++)
            {
                int offset = TensorOffset(score_tensor, score_nchw, score_channels, score_h, score_w, c, y, x);
                uint8_t score = scores[offset];
                if (score > threshold && score > max_raw)
                {
                    max_raw = score;
                    max_offset = offset;
                    class_id = c;
                }
            }

            if (class_id >= 0)
            {
                max_score = Dequant(score_tensor, max_offset);
            }
            return class_id >= 0;
        }

        float best = -1.0f;
        for (int c = 0; c < cls_count; c++)
        {
            int offset = TensorOffset(score_tensor, score_nchw, score_channels, score_h, score_w, c, y, x);
            float score = Dequant(score_tensor, offset);
            if (score > objectThreshold && score > best)
            {
                best = score;
                class_id = c;
            }
        }

        if (class_id >= 0)
        {
            max_score = best;
        }
        return class_id >= 0;
    }

    static int ProcessBranch(const tensor_data_s &box_tensor,
                             const tensor_data_s &score_tensor,
                             const tensor_data_s *score_sum_tensor,
                             int stride,
                             std::vector<DetectRect> &detectRects)
    {
        int box_channels = 0;
        int box_h = 0;
        int box_w = 0;
        bool box_nchw = true;
        if (!GetTensorShape(box_tensor, box_channels, box_h, box_w, box_nchw))
        {
            return -1;
        }

        int score_channels = 0;
        int score_h = 0;
        int score_w = 0;
        bool score_nchw = true;
        if (!GetTensorShape(score_tensor, score_channels, score_h, score_w, score_nchw))
        {
            return -1;
        }

        if (box_h != score_h || box_w != score_w)
        {
            NN_LOG_ERROR("YOLO11 box and score grid mismatch: box=%dx%d score=%dx%d",
                         box_w, box_h, score_w, score_h);
            return -1;
        }
        if (box_channels % 4 != 0)
        {
            NN_LOG_ERROR("YOLO11 box channel must be 4*dfl_len, got %d", box_channels);
            return -1;
        }

        int score_sum_channels = 0;
        int score_sum_h = 0;
        int score_sum_w = 0;
        bool score_sum_nchw = true;
        if (score_sum_tensor != nullptr)
        {
            if (!GetTensorShape(*score_sum_tensor, score_sum_channels, score_sum_h, score_sum_w, score_sum_nchw))
            {
                return -1;
            }
            if (score_sum_channels != 1 || score_sum_h != box_h || score_sum_w != box_w)
            {
                NN_LOG_ERROR("YOLO11 score_sum grid mismatch: score_sum=%dx%dx%d box=%dx%d",
                             score_sum_channels, score_sum_w, score_sum_h, box_w, box_h);
                return -1;
            }
        }

        const int dfl_len = box_channels / 4;
        const int cls_count = std::min(score_channels, class_num);
        const int grid_len = box_h * box_w;
        const size_t before_count = static_cast<size_t>(box_channels);
        if (before_count > 128)
        {
            NN_LOG_ERROR("YOLO11 dfl channel too large: %d", box_channels);
            return -1;
        }

        for (int y = 0; y < box_h; y++)
        {
            for (int x = 0; x < box_w; x++)
            {
                if (score_sum_tensor != nullptr)
                {
                    int score_sum_offset = TensorOffset(*score_sum_tensor,
                                                        score_sum_nchw,
                                                        score_sum_channels,
                                                        score_sum_h,
                                                        score_sum_w,
                                                        0,
                                                        y,
                                                        x);
                    if (!TensorValuePassThreshold(*score_sum_tensor, score_sum_offset, objectThreshold))
                    {
                        continue;
                    }
                }

                float max_score = 0.0f;
                int class_id = -1;
                if (!FindMaxClassScore(score_tensor,
                                       score_nchw,
                                       score_channels,
                                       score_h,
                                       score_w,
                                       y,
                                       x,
                                       cls_count,
                                       class_id,
                                       max_score))
                {
                    continue;
                }

                float before_dfl[128];
                for (int k = 0; k < box_channels; k++)
                {
                    int box_offset = TensorOffset(box_tensor, box_nchw, box_channels, box_h, box_w, k, y, x);
                    before_dfl[k] = Dequant(box_tensor, box_offset);
                }

                float box[4];
                ComputeDfl(before_dfl, dfl_len, box);

                float grid_x = static_cast<float>(x) + 0.5f;
                float grid_y = static_cast<float>(y) + 0.5f;

                float xmin = (grid_x - box[0]) * stride;
                float ymin = (grid_y - box[1]) * stride;
                float xmax = (grid_x + box[2]) * stride;
                float ymax = (grid_y + box[3]) * stride;

                xmin = std::max(0.0f, std::min(xmin, static_cast<float>(kInputW)));
                ymin = std::max(0.0f, std::min(ymin, static_cast<float>(kInputH)));
                xmax = std::max(0.0f, std::min(xmax, static_cast<float>(kInputW)));
                ymax = std::max(0.0f, std::min(ymax, static_cast<float>(kInputH)));

                if (xmax <= xmin || ymax <= ymin)
                {
                    continue;
                }

                DetectRect rect;
                rect.xmin = xmin / kInputW;
                rect.ymin = ymin / kInputH;
                rect.xmax = xmax / kInputW;
                rect.ymax = ymax / kInputH;
                rect.score = max_score;
                rect.classId = class_id;
                detectRects.push_back(rect);
            }
        }

        NN_LOG_DEBUG("YOLO11 branch stride=%d grid=%d candidates=%ld", stride, grid_len, detectRects.size());
        return 0;
    }

    int GetYolo11DetectionResult(const std::vector<tensor_data_s> &outputs,
                                 std::vector<float> &DetectiontRects)
    {
        // 9 输出模型使用 score_sum 快速过滤；6 输出模型没有 score_sum 时直接扫描类别分支。
        if (outputs.size() != 6 && outputs.size() != 9)
        {
            NN_LOG_ERROR("YOLO11 expects 6 outputs or 9 outputs, got %ld", outputs.size());
            return -1;
        }

        const int output_per_branch = static_cast<int>(outputs.size()) / kHeadNum;
        std::vector<DetectRect> detectRects;

        for (int i = 0; i < kHeadNum; i++)
        {
            const int box_idx = i * output_per_branch;
            const int score_idx = box_idx + 1;
            const tensor_data_s *score_sum_tensor = output_per_branch == 3 ? &outputs[box_idx + 2] : nullptr;
            if (ProcessBranch(outputs[box_idx], outputs[score_idx], score_sum_tensor, kStrides[i], detectRects) != 0)
            {
                return -1;
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](const DetectRect &a, const DetectRect &b)
                  { return a.score > b.score; });

        for (size_t i = 0; i < detectRects.size(); ++i)
        {
            if (detectRects[i].classId < 0)
            {
                continue;
            }

            for (size_t j = i + 1; j < detectRects.size(); ++j)
            {
                if (detectRects[j].classId == detectRects[i].classId &&
                    IOU(detectRects[i], detectRects[j]) > nmsThreshold)
                {
                    detectRects[j].classId = -1;
                }
            }

            DetectiontRects.push_back(static_cast<float>(detectRects[i].classId));
            DetectiontRects.push_back(detectRects[i].score);
            DetectiontRects.push_back(detectRects[i].xmin);
            DetectiontRects.push_back(detectRects[i].ymin);
            DetectiontRects.push_back(detectRects[i].xmax);
            DetectiontRects.push_back(detectRects[i].ymax);
        }

        return 0;
    }
}
