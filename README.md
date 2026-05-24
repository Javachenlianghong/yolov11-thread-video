# RK3588 YOLO11 多线程视频推理 Demo

这是一个精简后的 RK3588 YOLO11 推理工程，只保留本地图片和本地视频/USB 摄像头推理。

保留的 demo：

- `img_demo`：单张图片推理
- `thread_pool_demo`：本地视频或 USB 摄像头多线程推理，使用多个 RKNN 上下文并行跑 NPU

已移除内容：

- 网络推流服务相关代码
- MPP 编解码相关代码
- MediaKit 相关代码
- ini 推流配置文件

YOLO11 后处理参考瑞芯微官方样例：

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
export LD_LIBRARY_PATH=../librknn_api/aarch64:../3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH
```

如果在 `build` 目录运行，并且可执行文件需要加载当前目录下的工程动态库，也可以使用：

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
- `threads`：推理线程数，RK3588 建议先用 `3`
- `class_num`：类别数量，COCO 为 `80`
- `box_thresh`：检测框置信度阈值
- `nms_thresh`：NMS 阈值

弹窗显示说明：

- `show_window=0`：不弹窗，只推理/保存视频，适合 SSH 或无桌面环境
- `show_window=1`：弹出 OpenCV 窗口按原视频帧尺寸显示推理结果，窗口中按 `q` 或 `Esc` 退出

仍然兼容旧的命令行方式：

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 1 3 ../coco_80_labels_list.txt 80 0.25 0.45 0 thread_pool_demo.mp4
```

参数说明：

```text
thread_pool_demo <yolo11.rknn> <video_path|camera_id> [record 0/1] [threads 3] [labels_path] [class_num] [box_thresh] [nms_thresh] [show_window 0/1] [output_path]
```

示例：

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 1 3
./thread_pool_demo ../weights/yolo11n.rknn 0 0 3
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 0 3 ../coco_80_labels_list.txt 80 0.25 0.45 1
```

当 `record` 为 `1` 时，结果视频保存为：

```sh
thread_pool_demo.mp4
```

## 模型说明

本工程使用从官方优化版 YOLO11 ONNX 转换出来的 RKNN 模型。

官方优化版 YOLO11 输出格式通常是：

- `6` 个输出：三个尺度，每个尺度包含 `box, class`
- `9` 个输出：三个尺度，每个尺度包含 `box, class, score_sum`

本工程同时支持 `6` 输出和 `9` 输出。`score_sum` 分支在后处理中会被忽略，这一点和瑞芯微官方 Python 样例保持一致。

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
- `thread_pool_demo` 跑通 `medias/palace.mp4`，3 线程推理，生成 `thread_pool_demo.mp4`
