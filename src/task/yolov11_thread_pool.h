#ifndef RK3588_DEMO_YOLOV11_THREAD_POOL_H
#define RK3588_DEMO_YOLOV11_THREAD_POOL_H

#include "yolov11_custom.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class Yolov11ThreadPool
{
public:
    Yolov11ThreadPool();
    ~Yolov11ThreadPool();

    nn_error_e setUp(std::string &model_path, int num_threads = 3);
    nn_error_e setUp(std::string &model_path,
                     int num_threads,
                     float nms_threshold,
                     float box_threshold,
                     const std::string &model_labels_path,
                     int obj_class_num);
    nn_error_e submitTask(const cv::Mat &img, int id);
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);
    nn_error_e getTargetImgResult(cv::Mat &img, int id);
    nn_error_e getTargetResultNonBlock(std::vector<Detection> &objects, int id);
    nn_error_e getTargetResultNonBlockAndSourceImg(std::vector<Detection> &objects, cv::Mat &img, int id);
    void stopAll();

private:
    void worker(int id);

    // tasks_ 保存待推理帧，key 是递增帧号，用于结果按原始顺序输出。
    std::queue<std::pair<int, cv::Mat>> tasks_;
    // 每个工作线程独占一个 Yolov11Custom，也就是独占一个 RKNN context。
    std::vector<std::shared_ptr<Yolov11Custom>> yolo_instances_;
    // results_ 保存检测框，img_results_ 保存已绘制检测框的图像，img_source_ 保存原图。
    std::map<int, std::vector<Detection>> results_;
    std::map<int, cv::Mat> img_results_;
    std::map<int, cv::Mat> img_source_;
    std::vector<std::thread> threads_;
    std::mutex task_mutex_;
    std::mutex result_mutex_;
    std::condition_variable cv_task_;
    bool stop_;
};

#endif // RK3588_DEMO_YOLOV11_THREAD_POOL_H
