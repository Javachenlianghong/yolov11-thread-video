#include "rknn_engine.h"

#include <string.h>

#include "utils/engine_helper.h"
#include "utils/logging.h"

static const int g_max_io_num = 16;

RKEngine::RKEngine()
    : rknn_ctx_(0),
      ctx_created_(false),
      core_id_(-1),
      input_num_(0),
      output_num_(0)
{
}

rknn_core_mask RKEngine::CoreMaskFromId(int core_id) const
{
    // 线程池会传入 0/1/2 绑定单独 NPU core；-1 或其他值交给 RKNN 自动调度。
    switch (core_id)
    {
    case 0:
        return RKNN_NPU_CORE_0;
    case 1:
        return RKNN_NPU_CORE_1;
    case 2:
        return RKNN_NPU_CORE_2;
    case 3:
        return RKNN_NPU_CORE_0_1_2;
    default:
        return RKNN_NPU_CORE_AUTO;
    }
}

nn_error_e RKEngine::SetCoreId(int core_id)
{
    core_id_ = core_id;
    if (!ctx_created_)
    {
        // 模型尚未加载时只记录 core_id，真正的 rknn_set_core_mask 在 rknn_init 后执行。
        return NN_SUCCESS;
    }

    int ret = rknn_set_core_mask(rknn_ctx_, CoreMaskFromId(core_id_));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_set_core_mask fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL;
    }
    return NN_SUCCESS;
}

nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    // RKNN C API 需要先把模型文件完整读入内存，再调用 rknn_init。
    int model_len = 0;
    auto model = load_model(model_file, &model_len);
    if (model == nullptr)
    {
        NN_LOG_ERROR("load model file %s fail!", model_file);
        return NN_LOAD_MODEL_FAIL;
    }

    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL;
    }

    ctx_created_ = true;
    NN_LOG_INFO("rknn_init success!");

    // context 创建成功后立即设置 NPU core，保证后续推理按线程池分配执行。
    ret = rknn_set_core_mask(rknn_ctx_, CoreMaskFromId(core_id_));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_set_core_mask fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL;
    }

    rknn_sdk_version version;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_query SDK version fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("RKNN API version: %s", version.api_version);
    NN_LOG_INFO("RKNN Driver version: %s", version.drv_version);

    // 查询输入输出数量和张量属性，后续分配输入输出缓存时要依赖这些信息。
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_query io num fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;
    in_shapes_.clear();
    out_shapes_.clear();

    NN_LOG_INFO("input tensors:");
    rknn_tensor_attr input_attrs[g_max_io_num];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query input attr fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(input_attrs[i]));
        in_shapes_.push_back(rknn_tensor_attr_convert(input_attrs[i]));
    }

    NN_LOG_INFO("output tensors:");
    rknn_tensor_attr output_attrs[g_max_io_num];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query output attr fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(output_attrs[i]));
        out_shapes_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }

    return NN_SUCCESS;
}

const std::vector<tensor_attr_s> &RKEngine::GetInputShapes()
{
    return in_shapes_;
}

const std::vector<tensor_attr_s> &RKEngine::GetOutputShapes()
{
    return out_shapes_;
}

nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs,
                         std::vector<tensor_data_s> &outputs,
                         bool want_float)
{
    // 每次推理流程：设置输入 -> rknn_run -> 获取输出 -> 拷贝到自管理缓存。
    if (inputs.size() != input_num_)
    {
        NN_LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputs.size(), input_num_);
        return NN_IO_NUM_NOT_MATCH;
    }
    if (outputs.size() != output_num_)
    {
        NN_LOG_ERROR("outputs num not match! outputs.size()=%ld, output_num_=%d", outputs.size(), output_num_);
        return NN_IO_NUM_NOT_MATCH;
    }

    rknn_input rknn_inputs[g_max_io_num];
    memset(rknn_inputs, 0, sizeof(rknn_inputs));
    for (size_t i = 0; i < inputs.size(); i++)
    {
        rknn_inputs[i] = tensor_data_to_rknn_input(inputs[i]);
    }

    int ret = rknn_inputs_set(rknn_ctx_, static_cast<uint32_t>(inputs.size()), rknn_inputs);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        return NN_RKNN_INPUT_SET_FAIL;
    }

    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        return NN_RKNN_RUNTIME_ERROR;
    }

    rknn_output rknn_outputs[g_max_io_num];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (uint32_t i = 0; i < output_num_; ++i)
    {
        // int8 模型保持量化输出，后处理按 zp/scale 反量化；float 模型请求 float32。
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    }

    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_outputs_get fail! ret=%d", ret);
        return NN_RKNN_OUTPUT_GET_FAIL;
    }

    for (uint32_t i = 0; i < output_num_; ++i)
    {
        if (rknn_outputs[i].size > outputs[i].attr.size)
        {
            NN_LOG_ERROR("output[%d] buffer too small: have=%d need=%d",
                         i, outputs[i].attr.size, rknn_outputs[i].size);
            rknn_outputs_release(rknn_ctx_, output_num_, rknn_outputs);
            return NN_RKNN_OUTPUT_GET_FAIL;
        }
        rknn_output_to_tensor_data(rknn_outputs[i], outputs[i]);
    }

    // rknn_outputs_get 分配的 buf 必须用 rknn_outputs_release 释放。
    rknn_outputs_release(rknn_ctx_, output_num_, rknn_outputs);
    return NN_SUCCESS;
}

RKEngine::~RKEngine()
{
    if (ctx_created_)
    {
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");
    }
}

std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKEngine>();
}
