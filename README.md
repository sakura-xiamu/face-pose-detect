# 低头与转头检测

本地演示服务：用 YuNet 找出画面里最大的人脸，再用 MobileNet 估计头部姿态，判断是否低头、是否转头。推理走 ONNX Runtime，在 Windows 上优先使用 DirectML，没有 DirectML 时回退到 CPU。

页面支持上传图片和摄像头连续检测。低头与转头分开判定，可以同时成立。

## 工作流程

1. 初始化时加载两份 ONNX：人脸检测 `face_detection_yunet_2023mar.onnx`，以及所选姿态模型。
2. YuNet 把图像等比缩放到 640×640 后检测人脸，取得分最高且面积最大的一张脸。
3. 人脸框每边向外扩 20%，裁出脸部区域。
4. 姿态模型输出 3×3 旋转矩阵，换算成 pitch、yaw、roll（单位：度）。
5. 用页面上的阈值给出 `has_pitch`（低头）和 `has_yaw`（转头）。

姿态模型可选：

| 名称 | 文件 | 说明 |
| --- | --- | --- |
| `mobilenetv3_small` | `weights/mobilenetv3_small.onnx` | 默认，MobileNet V3-Small |
| `mobilenetv2` | `weights/mobilenetv2.onnx` | MobileNet V2 |

## 判定规则

角度来自当前这套旋转约定：

- **pitch**：正值表示鼻尖朝画面上方（抬头），负值表示朝下方（低头）。
- **yaw**：水平转头角。页面默认正视区间是 `-30,30`。

默认阈值：

- 低头：`pitch < -15`
- 转头：`yaw` 小于 `-30` 或大于 `30`

两项互不影响。例如低头且大幅侧转时，两个标记都会为真。

## 环境

面向 Windows 11，conda 环境名 `face-pose`，Python 3.11。依赖写在 `environment.yml`：

- `onnxruntime-directml`
- `opencv-python-headless`
- `fastapi`、`uvicorn`、`python-multipart`
- `numpy`、`pillow`

## 安装

环境已存在时可跳过。

```powershell
conda env create -f environment.yml
conda activate face-pose
```

模型已包含在 `weights/`：

- `face_detection_yunet_2023mar.onnx`
- `mobilenetv3_small.onnx`
- `mobilenetv2.onnx`

克隆仓库后可直接初始化。若文件被删掉，初始化接口会返回 400，并说明缺的是哪一个模型。

## 启动

在项目目录执行：

```powershell
conda activate face-pose
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000 。

1. 选择姿态模型，按需要改低头阈值和转头区间。
2. 点「初始化模型」，等状态变为已加载。页面会显示当前执行器（DML 或 CPU）。
3. 在「文件」里上传图片，或切到「摄像头」开启实时检测。
4. 结果区显示 pitch、yaw、是否低头、是否转头，并在预览上画出人脸框。
5. 用完可点「释放资源」卸下会话。

摄像头画面会镜像后再送去检测，和预览方向一致。单张图片上限 20MB。

## HTTP 接口

服务前缀是 `/api/detect/`。页面通过 SSE 订阅状态，检测请求走表单或 JSON。

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/` | 检测页面 |
| `GET` | `/api/detect/status` | 当前状态快照 |
| `GET` | `/api/detect/events` | SSE，状态变化时推送 |
| `POST` | `/api/detect/initialize` | 加载模型 |
| `POST` | `/api/detect/release` | 释放模型 |
| `POST` | `/api/detect/image` | 上传图片检测 |
| `POST` | `/api/detect/base64` | Base64 图片检测 |

初始化请求体：

```json
{ "model_name": "mobilenetv3_small" }
```

`model_name` 只能是 `mobilenetv3_small` 或 `mobilenetv2`。

图片检测使用 `multipart/form-data`：

- `file`：图片
- `lower_head_pc`：低头阈值，默认 `-15`
- `turnhead_pc`：转头正视区间，默认 `-30,30`，左边必须小于右边

Base64 请求体字段为 `image_data`、`lower_head_pc`、`turnhead_pc`。`image_data` 可以是纯 Base64，也可以是 `data:image/...;base64,` 开头的数据 URL。

检测成功时，`result` 大致如下：

```json
{
  "has_pitch": true,
  "has_yaw": false,
  "pitch": -22.4,
  "yaw": 3.1,
  "face_box": [120.0, 80.0, 180.0, 200.0],
  "image_width": 1280,
  "image_height": 720,
  "message": "检测完成"
}
```

未检测到人脸时，`has_pitch` 和 `has_yaw` 为 `false`，角度为 `null`，`message` 为「未检测到人脸」。`face_box` 是原图上的 `[x, y, w, h]`。

未初始化就检测会返回 400。转头区间格式不对也会返回 400。

## 项目结构

```
app/
  main.py            FastAPI 入口、状态与接口
  engine.py          加载模型并完成单帧检测
  face_detector.py   YuNet 解码与 NMS
  pose_estimator.py  MobileNet 姿态推理
  geometry.py        欧拉角、阈值、人脸框外扩
  runtime.py         ONNX Runtime 会话（优先 DirectML）
static/index.html    检测页面
tests/test_geometry.py
environment.yml
weights/             YuNet 与 MobileNet ONNX
```

根目录的 `detect.html` 是同一套接口的另一份页面，服务实际挂载的是 `static/index.html`。

## 测试

不加载模型即可跑几何与解码单测：

```powershell
conda activate face-pose
python -m unittest tests.test_geometry
```
