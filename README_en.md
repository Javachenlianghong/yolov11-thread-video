# RK3588 YOLO11 Multi-Thread Video Inference Demo

Language: [中文](README.md) | English

This is a trimmed RK3588 YOLO11 inference project. It keeps only local image inference and local video/USB camera inference.

Included demos:

- `img_demo`: single-image inference
- `thread_pool_demo`: multi-thread inference for local video files or USB cameras, using multiple RKNN contexts to run NPU inference in parallel

The YOLO11 post-processing implementation follows the Rockchip official example:

https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11

## Build On Board

Enter the project directory on the RK3588 board and run:

```sh
mkdir -p build
cd build
cmake ..
make -j4
```

Set the runtime library path before running:

```sh
export LD_LIBRARY_PATH=.:../librknn_api/aarch64:../3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH
```

## Image Inference

Test with `medias/bus.jpg`:

```sh
./img_demo ../weights/yolo11n.rknn ../medias/bus.jpg ../coco_80_labels_list.txt 80 0.25 0.45
```

Arguments:

```text
img_demo <yolo11.rknn> <image_path> [labels_path] [class_num] [box_thresh] [nms_thresh]
```

Output file:

```sh
result.jpg
```

## Multi-Thread Video Inference

Using the configuration file is recommended. The config file is located in the project root:

```text
thread_pool_demo_config.ini
```

Run directly inside the `build` directory:

```sh
./thread_pool_demo
```

This is equivalent to passing the config file explicitly:

```sh
./thread_pool_demo ../thread_pool_demo_config.ini
```

The default configuration tests `medias/palace.mp4` and saves the annotated output video:

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

Configuration options:

- `model_path`: RKNN model path
- `video_source`: video file path or camera index, for example `../medias/palace.mp4` or `0`
- `labels_path`: class label file path
- `record`: whether to save the output video, `1` to save and `0` to disable
- `show_window`: whether to show an OpenCV window, `1` to show and `0` to disable
- `output_path`: output video path, only used when `record=1`
- `benchmark`: non-blocking benchmark mode, `1` to enable and `0` to disable
- `benchmark_seconds`: benchmark duration in seconds
- `loop_video`: whether to loop a video file in normal mode; file videos are automatically looped in benchmark mode
- `threads`: inference thread count; on RK3588, test `3`, `6`, `9`, and `12` to find the best setting
- `class_num`: number of classes, `80` for COCO
- `box_thresh`: box confidence threshold; for 9-output YOLO11 models, the `score_sum` fast filter also uses this threshold
- `nms_thresh`: NMS threshold

## Window Display

- `show_window=0`: do not show a window; suitable for SSH or headless environments
- `show_window=1`: show the inference result in an OpenCV window at the original video frame size; press `q` or `Esc` to exit
- To keep NPU utilization high while showing a window, use `benchmark=1` and `show_window=1`. In this mode, the display thread only renders the latest completed frame and drops old frames, so it does not block inference by waiting for frame order.

## Non-Blocking Benchmark Mode

Enable benchmark mode:

```ini
benchmark=1
show_window=1
threads=12
benchmark_seconds=30
```

Run:

```sh
./thread_pool_demo ../thread_pool_demo_config.ini
```

Benchmark mode automatically:

- disables `record`
- disables image caching when `show_window=0`, measuring pure inference throughput
- shows the latest completed frame when `show_window=1`, drawing boxes only on displayed frames
- loops file videos automatically
- collects completed results non-blockingly instead of waiting for frame order, reducing result-thread backpressure on the inference pipeline

Example logs:

```text
[NN_INFO] benchmark FPS:120.312500, Submitted:3600, Done:3588, Pending:12, Detections:42
[NN_INFO] benchmark FPS:118.500000, DisplayFPS:58.000000, Submitted:3600, Done:3588, Pending:12, Detections:42
```

Use this command at the same time to monitor NPU load:

```sh
watch -n 1 sudo cat /sys/kernel/debug/rknpu/load
```

The old command-line style is still supported:

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 1 3 ../coco_80_labels_list.txt 80 0.25 0.45 0 thread_pool_demo.mp4
```

Benchmark can also be enabled from the command line:

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 0 12 ../coco_80_labels_list.txt 80 0.25 0.45 0 thread_pool_demo.mp4 1 30
```

Non-blocking benchmark with window display:

```sh
./thread_pool_demo ../weights/yolo11n.rknn ../medias/palace.mp4 0 12 ../coco_80_labels_list.txt 80 0.25 0.45 1 thread_pool_demo.mp4 1 30
```

Arguments:

```text
thread_pool_demo <yolo11.rknn> <video_path|camera_id> [record 0/1] [threads 3] [labels_path] [class_num] [box_thresh] [nms_thresh] [show_window 0/1] [output_path] [benchmark 0/1] [benchmark_seconds 30]
```

## Model Notes

This project uses an RKNN model converted from the official optimized YOLO11 ONNX model.

The official optimized YOLO11 output format is usually:

- `6` outputs: three scales, each containing `box, class`
- `9` outputs: three scales, each containing `box, class, score_sum`

This project supports both `6` and `9` outputs. For `9`-output models, post-processing uses the `score_sum` branch to filter low-score grid points before scanning class scores and decoding DFL boxes, reducing CPU post-processing overhead. For `6`-output models, where `score_sum` is not available, the code falls back to directly scanning the class branch.

Raw unoptimized single-output YOLO11 RKNN models are not supported.

## Verified Environment

Verified on the following RK3588 board environment:

```text
Linux lubancat 5.10.160 aarch64
OpenCV 4.5.1
RKNN API 1.5.3b6
RKNN Driver 0.9.8
```

Model used:

```text
weights/yolo11n.rknn
```

Verification results:

- `img_demo` runs successfully on `medias/bus.jpg` and generates `result.jpg`
- `thread_pool_demo` runs successfully on `medias/palace.mp4`, supporting normal output-video mode and non-blocking benchmark mode
