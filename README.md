# RK3588 YOLO11 多线程视频推理 Demo

这是一个精简后的 RK3588 YOLO11 推理工程，只保留本地图片推理和本地视频/USB 摄像头推理。

保留的 demo：

- `img_demo`：单张图片推理
- `thread_pool_demo`：本地视频或 USB 摄像头多线程推理，使用多个 RKNN context 并行跑 NPU

已移除内容：

- RTSP 推流服务相关代码
- MPP 编解码相关代码
- MediaKit 相关代码
- 原推流配置文件

YOLO11 后处理参考 Rockchip 官方样例：

https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11

## 板端编译

在 RK3588 板端进入工程目录后执行：

```sh
mkdir -p build
cd build
cmake ..
make -j4
```

运行前设置动态库路径：

```sh
export LD_LIBRARY_PATH=.:../librknn_api/aarch64:../3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH
```

## 图片推理

使用 `medias/bus.jpg` 测试：

```sh
./img_demo ../weights/yolo11n.rknn ../medias/bus.jpg ../coco_80_labels_list.txt 80 0.25 0.45
```

参数说明：

```text
img_demo <yolo11.rknn> <image_path> [labels_path] [class_num] [box_thresh] [nms_thresh]
```

输出文件：

```sh
result.jpg
```

## 多线程视频推理

推荐使用配置文件运行。配置文件在工程根目录：

```text
thread_pool_demo_config.ini
```

在 `build` 目录中直接运行：

```sh
./thread_pool_demo
```

等价于显式传入配置文件：

```sh
./thread_pool_demo ../thread_pool_demo_config.ini
```

默认配置使用 `medias/palace.mp4` 测试，并保存检测后的视频：

```ini
model_path=../weights/yolo11n.rknn
video_source=../medias/palace.mp4
labels_path=../coco_80_labels_list.txt
record=1
show_window=0
output_path=thread_pool_demo.mp4
benchmark=0
benchmark_seconds=30
loop_video=0
threads=3
class_num=80
box_thresh=0.25
nms_thresh=0.45
```

配置项说明：

- `model_path`：RKNN 模型路径
- `video_source`：视频文件路径或摄像头编号，例如 `../medias/palace.mp4` 或 `0`
- `labels_path`：类别标签文件路径
- `record`：是否保存结果视频，`1` 保存，`0` 不保存
- `show_window`：是否弹窗显示，`1` 显示，`0` 不显示
- `output_path`：结果视频保存路径，仅 `record=1` 时生效
- `benchmark`：非阻塞性能测试模式，`1` 开启，`0` 关闭
- `benchmark_seconds`：benchmark 持续时间，单位秒
- `loop_video`：普通模式下是否循环视频；benchmark 模式下文件视频会自动循环
- `threads`：推理线程数，RK3588 建议分别测试 `3`、`6`、`9`、`12`
- `class_num`：类别数量，COCO 为 `80`
- `box_thresh`：检测框置信度阈值；9 输出 YOLO11 的 `score_sum` 快速过滤也使用该阈值
- `nms_thresh`：NMS 阈值

## 弹窗显示

- `show_window=0`：不弹窗，只推理/保存视频，适合 SSH 或无桌面环境
- `show_window=1`：弹出 OpenCV 窗口，按原视频帧尺寸显示推理结果，窗口中按 `q` 或 `Esc` 退出
- 如果希望弹窗显示时仍尽量拉高 NPU 利用率，使用 `benchmark=1` 和 `show_window=1`。此时显示线程只展示最新完成帧，旧帧会被丢弃，不会按帧号阻塞推理线程。

## 非阻塞 benchmark 模式

开启：

```ini
benchmark=1
show_window=1
threads=12
benchmark_seconds=30
```

运行：

```sh
./thread_pool_demo ../thread_pool_demo_config.ini
```

benchmark 模式会自动：

- 关闭 `record`
- `show_window=0` 时关闭图像缓存，只统计纯推理吞吐
- `show_window=1` 时弹窗显示最新完成帧，并只给显示出来的帧画框
- 对文件视频自动循环
- 非阻塞回收结果，不再按帧号等待，减少结果线程对推理流水线的回压

日志示例：

```text
[NN_INFO] benchmark FPS:120.312500, Submitted:3600, Done:3588, Pending:12, Detections:42
[NN_INFO] benchmark FPS:118.500000, DisplayFPS:58.000000, Submitted:3600, Done:3588, Pending:12, Detections:42
```

这组日志适合配合下面命令观察 NPU：

```sh
watch -n 1 sudo cat /sys/kernel/debug/rknpu/load
```

仍然兼容旧的命令行方式：

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 1 3 ../coco_80_labels_list.txt 80 0.25 0.45 0 thread_pool_demo.mp4
```

新命令行参数也可以打开 benchmark：

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 0 12 ../coco_80_labels_list.txt 80 0.25 0.45 0 thread_pool_demo.mp4 1 30
```

打开带弹窗的非阻塞 benchmark：

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 0 12 ../coco_80_labels_list.txt 80 0.25 0.45 1 thread_pool_demo.mp4 1 30
```

参数说明：

```text
thread_pool_demo <yolo11.rknn> <video_path|camera_id> [record 0/1] [threads 3] [labels_path] [class_num] [box_thresh] [nms_thresh] [show_window 0/1] [output_path] [benchmark 0/1] [benchmark_seconds 30]
```

## 模型说明

本工程使用从官方优化版 YOLO11 ONNX 转换出来的 RKNN 模型。

官方优化版 YOLO11 输出格式通常是：

- `6` 个输出：三个尺度，每个尺度包含 `box, class`
- `9` 个输出：三个尺度，每个尺度包含 `box, class, score_sum`

本工程同时支持 `6` 输出和 `9` 输出。对于 `9` 输出模型，后处理会使用 `score_sum` 分支先过滤低分网格点，再扫描类别分支和做 DFL 解码，减少 CPU 后处理开销。对于 `6` 输出模型，没有 `score_sum` 时会退化为直接扫描类别分支。

不支持原始未优化的 YOLO11 单输出 RKNN 模型。

## 已验证环境

已在下面的 RK3588 板端环境验证通过：

```text
Linux lubancat 5.10.160 aarch64
OpenCV 4.5.1
RKNN API 1.5.3b6
RKNN Driver 0.9.8
```

使用模型：

```text
weights/yolo11n.rknn
```

验证结果：

- `img_demo` 跑通 `medias/bus.jpg`，生成 `result.jpg`
- `thread_pool_demo` 跑通 `medias/palace.mp4`，支持普通保存视频模式和非阻塞 benchmark 模式
