#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "task/yolov11_thread_pool.h"
#include "utils/logging.h"

static std::atomic<int> g_frame_start_id{0};
static std::atomic<int> g_frame_end_id{0};
static std::atomic<bool> g_read_end{false};
static std::atomic<bool> g_user_stop{false};
static std::unique_ptr<Yolov11ThreadPool> g_yolov11_thread_pool;

struct VideoDemoConfig
{
    std::string model_path{"../weights/yolo11n.rknn"};
    std::string video_source{"../medias/palace.mp4"};
    std::string labels_path{"../coco_80_labels_list.txt"};
    std::string output_path{"thread_pool_demo.mp4"};
    bool record{true};
    bool show_window{false};
    bool benchmark{false};
    bool loop_video{false};
    int benchmark_seconds{30};
    int threads{3};
    int class_num{80};
    float box_thresh{0.25f};
    float nms_thresh{0.45f};
};

static std::string trim(const std::string &value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
    {
        begin++;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        end--;
    }
    return value.substr(begin, end - begin);
}

static std::string lower_copy(std::string value)
{
    for (auto &ch : value)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static bool parse_bool(const std::string &value)
{
    std::string v = lower_copy(trim(value));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static bool file_exists(const std::string &path)
{
    std::ifstream file(path);
    return file.good();
}

static bool load_config_file(const std::string &config_path, VideoDemoConfig &config)
{
    std::ifstream file(config_path);
    if (!file.is_open())
    {
        NN_LOG_ERROR("failed to open config file: %s", config_path.c_str());
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(file, line))
    {
        line_no++;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';')
        {
            continue;
        }
        if (line.front() == '[' && line.back() == ']')
        {
            continue;
        }

        size_t pos = line.find('=');
        if (pos == std::string::npos)
        {
            NN_LOG_WARNING("ignore invalid config line %d: %s", line_no, line.c_str());
            continue;
        }

        std::string key = lower_copy(trim(line.substr(0, pos)));
        std::string value = trim(line.substr(pos + 1));
        if (key == "model_path")
        {
            config.model_path = value;
        }
        else if (key == "video_source" || key == "video_path" || key == "camera_id")
        {
            config.video_source = value;
        }
        else if (key == "labels_path")
        {
            config.labels_path = value;
        }
        else if (key == "output_path" || key == "output_video")
        {
            config.output_path = value;
        }
        else if (key == "record")
        {
            config.record = parse_bool(value);
        }
        else if (key == "show_window" || key == "display")
        {
            config.show_window = parse_bool(value);
        }
        else if (key == "benchmark")
        {
            config.benchmark = parse_bool(value);
        }
        else if (key == "loop_video")
        {
            config.loop_video = parse_bool(value);
        }
        else if (key == "benchmark_seconds" || key == "benchmark_duration")
        {
            config.benchmark_seconds = std::atoi(value.c_str());
        }
        else if (key == "threads")
        {
            config.threads = std::atoi(value.c_str());
        }
        else if (key == "class_num")
        {
            config.class_num = std::atoi(value.c_str());
        }
        else if (key == "box_thresh")
        {
            config.box_thresh = static_cast<float>(std::atof(value.c_str()));
        }
        else if (key == "nms_thresh")
        {
            config.nms_thresh = static_cast<float>(std::atof(value.c_str()));
        }
        else
        {
            NN_LOG_WARNING("ignore unknown config key: %s", key.c_str());
        }
    }
    return true;
}

static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s [config.ini]\n", program);
    printf("  %s <yolo11.rknn> <video_path|camera_id> [record 0/1] [threads 3] [labels_path] [class_num] [box_thresh] [nms_thresh] [show_window 0/1] [output_path] [benchmark 0/1] [benchmark_seconds 30]\n", program);
}

static bool is_camera_id(const char *value)
{
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }
    for (const char *p = value; *p != '\0'; ++p)
    {
        if (!std::isdigit(static_cast<unsigned char>(*p)))
        {
            return false;
        }
    }
    return true;
}

static cv::VideoCapture open_capture(const char *video_file)
{
    if (is_camera_id(video_file))
    {
        return cv::VideoCapture(atoi(video_file));
    }
    return cv::VideoCapture(video_file);
}

static bool benchmark_time_over(const VideoDemoConfig &config,
                                const std::chrono::steady_clock::time_point &start)
{
    if (!config.benchmark || config.benchmark_seconds <= 0)
    {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - start).count());
    return elapsed >= config.benchmark_seconds;
}

static void get_ordered_results(const VideoDemoConfig &config, double fps)
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    std::string fps_str;

    cv::VideoWriter writer;
    bool record = config.record;
    bool show_window = config.show_window;
    if (show_window)
    {
        try
        {
            // WINDOW_AUTOSIZE 会按当前帧原始尺寸显示，不允许窗口缩放图像。
            cv::namedWindow("YOLO11 Thread Pool Demo", cv::WINDOW_AUTOSIZE);
        }
        catch (const cv::Exception &e)
        {
            NN_LOG_ERROR("failed to create display window: %s", e.what());
            show_window = false;
        }
    }

    while (true)
    {
        if (g_user_stop.load())
        {
            break;
        }

        if (g_read_end.load() && g_frame_end_id.load() >= g_frame_start_id.load())
        {
            break;
        }

        cv::Mat img;
        const int frame_id = g_frame_end_id.load();
        auto ret = g_yolov11_thread_pool->getTargetImgResult(img, frame_id);
        if (ret != NN_SUCCESS)
        {
            if (g_read_end.load())
            {
                break;
            }
            continue;
        }
        g_frame_end_id++;
        frame_count++;

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - start_all).count() / 1000.0f;
        if (elapsed_ms > 1000.0f)
        {
            float fps_value = frame_count / (elapsed_ms / 1000.0f);
            NN_LOG_INFO("thread_pool_demo FPS:%f, Frame Count:%d", fps_value, frame_count);
            fps_str = std::to_string(static_cast<int>(fps_value)) + " FPS";
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }

        if (record && !writer.isOpened())
        {
            double output_fps = fps > 1.0 ? fps : 30.0;
            writer.open(config.output_path,
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        output_fps,
                        img.size());
            if (!writer.isOpened())
            {
                NN_LOG_ERROR("failed to open output video: %s", config.output_path.c_str());
                record = false;
            }
        }

        if (!fps_str.empty() && (record || show_window))
        {
            cv::putText(img, fps_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }

        if (show_window)
        {
            try
            {
                cv::imshow("YOLO11 Thread Pool Demo", img);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q')
                {
                    NN_LOG_INFO("display window requested stop.");
                    g_user_stop = true;
                    g_read_end = true;
                    break;
                }
            }
            catch (const cv::Exception &e)
            {
                NN_LOG_ERROR("display window failed: %s", e.what());
                show_window = false;
            }
        }

        if (record && writer.isOpened())
        {
            writer << img;
        }
    }

    g_yolov11_thread_pool->stopAll();
    if (writer.isOpened())
    {
        writer.release();
    }
    if (show_window)
    {
        cv::destroyWindow("YOLO11 Thread Pool Demo");
    }
    NN_LOG_INFO("get_results end.");
}

static void get_benchmark_results(const VideoDemoConfig &config)
{
    auto start = std::chrono::steady_clock::now();
    auto last = start;
    int frame_count = 0;
    int detect_count = 0;

    while (!g_user_stop.load())
    {
        if (benchmark_time_over(config, start))
        {
            g_user_stop = true;
            g_read_end = true;
            break;
        }

        std::vector<Detection> objects;
        int frame_id = -1;
        auto ret = g_yolov11_thread_pool->getAnyTargetResultNonBlock(objects, frame_id);
        if (ret == NN_SUCCESS)
        {
            g_frame_end_id++;
            frame_count++;
            detect_count += static_cast<int>(objects.size());
        }
        else
        {
            if (g_read_end.load() && g_frame_end_id.load() >= g_frame_start_id.load())
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count() / 1000.0f;
        if (elapsed_ms > 1000.0f)
        {
            float fps_value = frame_count / (elapsed_ms / 1000.0f);
            int submitted = g_frame_start_id.load();
            int completed = g_frame_end_id.load();
            NN_LOG_INFO("benchmark FPS:%f, Submitted:%d, Done:%d, Pending:%d, Detections:%d",
                        fps_value,
                        submitted,
                        completed,
                        submitted - completed,
                        detect_count);
            frame_count = 0;
            detect_count = 0;
            last = now;
        }
    }

    g_yolov11_thread_pool->stopAll();
    NN_LOG_INFO("benchmark end. submitted=%d, done=%d", g_frame_start_id.load(), g_frame_end_id.load());
}

static void read_video(const char *video_file, bool loop_video)
{
    cv::VideoCapture cap = open_capture(video_file);
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("failed to open video source: %s", video_file);
        g_read_end = true;
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    NN_LOG_INFO("video source: %s, size: %d x %d, fps: %.2f", video_file, width, height, fps);

    cv::Mat img;
    while (!g_user_stop.load())
    {
        cap >> img;
        if (img.empty())
        {
            if (loop_video && !is_camera_id(video_file))
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            }

            NN_LOG_INFO("video end.");
            g_read_end = true;
            break;
        }

        int id = g_frame_start_id++;
        auto ret = g_yolov11_thread_pool->submitTask(img.clone(), id);
        if (ret != NN_SUCCESS)
        {
            break;
        }
    }

    g_read_end = true;
    cap.release();
}

int main(int argc, char **argv)
{
    VideoDemoConfig config;

    if (argc == 1)
    {
        const std::string default_config = "../thread_pool_demo_config.ini";
        if (!file_exists(default_config) || !load_config_file(default_config, config))
        {
            print_usage(argv[0]);
            return -1;
        }
    }
    else if (argc == 2)
    {
        if (!load_config_file(argv[1], config))
        {
            return -1;
        }
    }
    else
    {
        config.model_path = argv[1];
        config.video_source = argv[2];
        config.record = argc > 3 ? atoi(argv[3]) != 0 : false;
        config.threads = argc > 4 ? atoi(argv[4]) : 3;
        config.labels_path = argc > 5 ? argv[5] : "coco_80_labels_list.txt";
        config.class_num = argc > 6 ? atoi(argv[6]) : 80;
        config.box_thresh = argc > 7 ? static_cast<float>(atof(argv[7])) : 0.25f;
        config.nms_thresh = argc > 8 ? static_cast<float>(atof(argv[8])) : 0.45f;
        config.show_window = argc > 9 ? atoi(argv[9]) != 0 : false;
        config.output_path = argc > 10 ? argv[10] : "thread_pool_demo.mp4";
        config.benchmark = argc > 11 ? atoi(argv[11]) != 0 : false;
        config.benchmark_seconds = argc > 12 ? atoi(argv[12]) : 30;
    }

    if (config.threads <= 0)
    {
        config.threads = 3;
    }
    if (config.benchmark)
    {
        // benchmark 模式只测推理流水线吞吐：不保存、不显示、不画框，并循环视频让 NPU 持续有输入。
        config.record = false;
        config.show_window = false;
        config.loop_video = !is_camera_id(config.video_source.c_str());
        if (config.benchmark_seconds <= 0)
        {
            config.benchmark_seconds = 30;
        }
    }

    NN_LOG_INFO("model_path: %s", config.model_path.c_str());
    NN_LOG_INFO("video_source: %s", config.video_source.c_str());
    NN_LOG_INFO("record: %d, show_window: %d, benchmark: %d, loop_video: %d, threads: %d, output_path: %s",
                config.record ? 1 : 0,
                config.show_window ? 1 : 0,
                config.benchmark ? 1 : 0,
                config.loop_video ? 1 : 0,
                config.threads,
                config.output_path.c_str());

    cv::VideoCapture probe = open_capture(config.video_source.c_str());
    double fps = probe.isOpened() ? probe.get(cv::CAP_PROP_FPS) : 30.0;
    probe.release();

    g_frame_start_id = 0;
    g_frame_end_id = 0;
    g_read_end = false;
    g_user_stop = false;

    g_yolov11_thread_pool.reset(new Yolov11ThreadPool());
    auto ret = g_yolov11_thread_pool->setUp(config.model_path,
                                            config.threads,
                                            config.nms_thresh,
                                            config.box_thresh,
                                            config.labels_path,
                                            config.class_num);
    if (ret != NN_SUCCESS)
    {
        return ret;
    }
    g_yolov11_thread_pool->setDrawResult(!config.benchmark);

    std::thread read_video_thread(read_video, config.video_source.c_str(), config.loop_video);
    std::thread result_thread;
    if (config.benchmark)
    {
        result_thread = std::thread(get_benchmark_results, config);
    }
    else
    {
        result_thread = std::thread(get_ordered_results, config, fps);
    }

    read_video_thread.join();
    result_thread.join();

    g_yolov11_thread_pool.reset();
    return 0;
}
