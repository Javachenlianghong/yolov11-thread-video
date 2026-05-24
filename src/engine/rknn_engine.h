#ifndef RK3588_DEMO_RKNN_ENGINE_H
#define RK3588_DEMO_RKNN_ENGINE_H

#include "engine.h"

#include <rknn_api.h>
#include <vector>

class RKEngine : public NNEngine
{
public:
    RKEngine();
    ~RKEngine() override;

    nn_error_e SetCoreId(int core_id) override;
    nn_error_e LoadModelFile(const char *model_file) override;
    const std::vector<tensor_attr_s> &GetInputShapes() override;
    const std::vector<tensor_attr_s> &GetOutputShapes() override;
    nn_error_e Run(std::vector<tensor_data_s> &inputs,
                   std::vector<tensor_data_s> &outputs,
                   bool want_float) override;

private:
    rknn_core_mask CoreMaskFromId(int core_id) const;

    rknn_context rknn_ctx_;
    bool ctx_created_;
    int core_id_;
    uint32_t input_num_;
    uint32_t output_num_;
    std::vector<tensor_attr_s> in_shapes_;
    std::vector<tensor_attr_s> out_shapes_;
};

#endif // RK3588_DEMO_RKNN_ENGINE_H
