#include "yolov11_thread_pool.h"

#include "draw/cv_draw.h"
#include "utils/logging.h"

#include <chrono>

Yolov11ThreadPool::Yolov11ThreadPool()
{
    stop_ = false;
}

Yolov11ThreadPool::~Yolov11ThreadPool()
{
    stopAll();
    for (auto &thread : threads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

nn_error_e Yolov11ThreadPool::setUp(std::string &model_path, int num_threads)
{
    for (int i = 0; i < num_threads; ++i)
    {
        // RK3588 有 3 个 NPU core，超过 3 个线程时按 0/1/2 轮询绑定。
        auto yolo = std::make_shared<Yolov11Custom>(i % 3);
        auto ret = yolo->LoadModel(model_path.c_str());
        if (ret != NN_SUCCESS)
        {
            return ret;
        }
        yolo_instances_.push_back(yolo);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        threads_.emplace_back(&Yolov11ThreadPool::worker, this, i);
    }
    return NN_SUCCESS;
}

nn_error_e Yolov11ThreadPool::setUp(std::string &model_path,
                                    int num_threads,
                                    float nms_threshold,
                                    float box_threshold,
                                    const std::string &model_labels_path,
                                    int obj_class_num)
{
    for (int i = 0; i < num_threads; ++i)
    {
        // 每个线程加载一份模型，避免多个线程共享同一个 rknn_context。
        auto yolo = std::make_shared<Yolov11Custom>(i % 3);
        auto ret = yolo->LoadModel(model_path.c_str());
        if (ret != NN_SUCCESS)
        {
            return ret;
        }
        yolo->setStaticParams(nms_threshold, box_threshold, model_labels_path, obj_class_num);
        yolo_instances_.push_back(yolo);
    }

    for (int i = 0; i < num_threads; ++i)
    {
        threads_.emplace_back(&Yolov11ThreadPool::worker, this, i);
    }
    return NN_SUCCESS;
}

void Yolov11ThreadPool::worker(int id)
{
    while (!stop_)
    {
        std::pair<int, cv::Mat> task;
        {
            // 等待读帧线程提交任务，stop_ 置位后唤醒并退出。
            std::unique_lock<std::mutex> lock(task_mutex_);
            cv_task_.wait(lock, [&]
                          { return !tasks_.empty() || stop_; });

            if (stop_)
            {
                return;
            }

            task = tasks_.front();
            tasks_.pop();
        }

        std::vector<Detection> detections;
        auto instance = yolo_instances_[id];
        // 推理耗时主要发生在这里，多个 worker 会并行调用各自的 RKNN context。
        instance->Run(task.second, detections);

        {
            // 结果按帧号存入 map，取结果线程可以按原始帧序阻塞等待。
            std::lock_guard<std::mutex> lock(result_mutex_);
            results_[task.first] = detections;
            img_source_[task.first] = task.second;
            DrawDetections(task.second, detections);
            img_results_[task.first] = task.second;
        }
        cv_task_.notify_all();
    }
}

nn_error_e Yolov11ThreadPool::submitTask(const cv::Mat &img, int id)
{
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            if (tasks_.size() <= 10)
            {
                tasks_.push({id, img});
                break;
            }
        }
        // 限制队列长度，防止视频读取过快时内存持续增长。
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cv_task_.notify_one();
    return NN_SUCCESS;
}

nn_error_e Yolov11ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    while (true)
    {
        {
            // 阻塞等待指定帧号的检测框结果。
            std::lock_guard<std::mutex> lock(result_mutex_);
            auto it = results_.find(id);
            if (it != results_.end())
            {
                objects = it->second;
                results_.erase(id);
                img_source_.erase(id);
                img_results_.erase(id);
                return NN_SUCCESS;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

nn_error_e Yolov11ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    while (true)
    {
        {
            // 视频 demo 用这个接口取已经画好框的图像。
            std::lock_guard<std::mutex> lock(result_mutex_);
            auto it = img_results_.find(id);
            if (it != img_results_.end())
            {
                img = it->second;
                img_results_.erase(id);
                results_.erase(id);
                img_source_.erase(id);
                return NN_SUCCESS;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            return NN_TIMEOUT;
        }
    }
}

nn_error_e Yolov11ThreadPool::getTargetResultNonBlock(std::vector<Detection> &objects, int id)
{
    std::lock_guard<std::mutex> lock(result_mutex_);
    auto it = results_.find(id);
    if (it == results_.end())
    {
        return NN_RESULT_NOT_READY;
    }

    objects = it->second;
    results_.erase(id);
    img_results_.erase(id);
    img_source_.erase(id);
    return NN_SUCCESS;
}

nn_error_e Yolov11ThreadPool::getTargetResultNonBlockAndSourceImg(std::vector<Detection> &objects, cv::Mat &img, int id)
{
    std::lock_guard<std::mutex> lock(result_mutex_);
    auto it = results_.find(id);
    if (it == results_.end())
    {
        return NN_RESULT_NOT_READY;
    }

    objects = it->second;
    img = img_source_[id];
    results_.erase(id);
    img_results_.erase(id);
    img_source_.erase(id);
    return NN_SUCCESS;
}

void Yolov11ThreadPool::stopAll()
{
    stop_ = true;
    cv_task_.notify_all();
}
