# YOLO11 RKNN 模型目录

把 RK3588 上使用的 YOLO11 RKNN 模型放在这个目录，例如：

```sh
weights/yolo11n.rknn
```

本工程参考瑞芯微官方 YOLO11 优化模型输出格式：

- `6` 输出：三个尺度，每个尺度包含 `box, class`
- `9` 输出：三个尺度，每个尺度包含 `box, class, score_sum`

后处理会忽略 `score_sum` 分支，和官方 Python 样例保持一致。
