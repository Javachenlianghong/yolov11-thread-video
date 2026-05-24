#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <string>
#include <vector>

#include "types/datatype.h"

namespace yolo
{
    extern float objectThreshold;
    extern float nmsThreshold;
    extern int class_num;

    int GetYolo11DetectionResult(const std::vector<tensor_data_s> &outputs,
                                 std::vector<float> &DetectiontRects);
}

#endif // RK3588_DEMO_POSTPROCESS_H
