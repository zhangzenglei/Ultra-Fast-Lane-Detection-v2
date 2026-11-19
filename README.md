# Ultra-Fast-Lane-Detection-V2
PyTorch implementation of the paper "[Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)".


![](ufldv2.png "vis")

# Demo 
<a href="https://youtu.be/VkvpoHlaMe0
" target="_blank"><img src="http://img.youtube.com/vi/VkvpoHlaMe0/0.jpg" 
alt="Demo" width="240" height="180" border="10" /></a>


# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
Please modify the `data_root` in any configs you would like to run. We will use `configs/culane_res18.py` as an example.

To train the model, you can run:
```
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
It should be noted that if you use different number of GPUs, the learning rate should be adjusted accordingly. The configs' learning rates correspond to 8-GPU training on CULane and CurveLanes datasets. **If you want to train on CULane or CurveLanes with single GPU, please decrease the learning rate by a factor of 1/8.** On the Tusimple, the learning rate corresponds to single GPU training.
# Trained models
We provide trained models on CULane, Tusimple, and CurveLanes.

| Dataset    | Backbone | F1   | Link |
|------------|----------|-------|------|
| CULane     | ResNet18 | 75.0  |  [Google](https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs)    |
| CULane     | ResNet34 | 76.0  |   [Google](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f)   |
| Tusimple   | ResNet18 | 96.11 |   [Google](https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1)   |
| Tusimple   | ResNet34 | 96.24 |   [Google](https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p)   |
| CurveLanes | ResNet18 | 80.42 |   [Google](https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe)   |
| CurveLanes | ResNet34 | 81.34 |   [Google](https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3)   |

For evaluation, run
```Shell
mkdir tmp

python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

Same as training, multi-gpu evaluation is also supported.
```Shell
mkdir tmp

python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

# Visualization
We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane.
```
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth
```

# Tensorrt Deploy
We also provide a python script to do tensorrt inference on videos.

1. Convert to onnx model
    ```
    python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
    ```
    Or you can download the onnx model using the following script: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh. And copy `ufldv2_culane_res34_320x1600.onnx` to `weights/ufldv2_culane_res34_320x1600.onnx`

2. Convert to tensorrt model

    Use trtexec to convert engine model

    `trtexec --onnx=weights/culane_res34.onnx --saveEngine=weights/culane_res34.engine`

3. Do inference
    ```
    python deploy/trt_infer.py --config_path  configs/culane_res34.py --engine_path weights/culane_res34.engine --video_path example.mp4
    ```

# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author={Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3182097}
}
```

# 前视图像去畸变及车道线结果对比

在工程根目录运行（需要 CUDA GPU）：

```bash
source .venv/bin/activate

python batch_infer_ufldv2.py \
    --config configs/culane_res34.py \
    --weights weights/culane_res34.pth \
    --input /home/zenglei/Data/seg_j6/seg/sensor_camera_front_mono_pym_image \
    --camera-param /home/zenglei/Data/seg_j6/seg/sensor_camera_front_mono_param/sensor_camera_front_mono_param_0.json \
    --recorded-contours /home/zenglei/Data/seg_j6/seg/perception_calib_contour_front_mono \
    --match-tolerance-ms 1 \
    --output ./results/seg_j6_comparison
```

- 内参固定，只需指定一个内参 JSON。畸变系数采用 `[k1, k2, p1, p2, k3, k4, k5, k6]`，允许尾部零填充。
- 图像必须是完整标定画面的缩放图，不能有额外裁剪或旋转。按实际分辨率缩放 K，在内存中去畸变，输出仍使用该 K，保持尺寸、不裁剪；不保存去畸变中间图。
- 本 Topic 的录制轮廓已恢复到原图尺度，并已逐点去畸变，对应原始 K。对于 3840×2160 标定、1920×1080 输入，录制点坐标乘以 0.5 后直接叠加，不重复去畸变。
- 图像文件名末尾为微秒时间戳，轮廓 JSON 的 `timestamp` 为毫秒。按最近时间匹配，默认容差 1 ms；未匹配帧仍输出 UFLDv2 结果，并显示 `NO MATCH`。
- 最终 `*_lanes.jpg` 为去畸变底图，橙色表示 UFLDv2，青色采样点表示录制轮廓。录制数据的分组连接语义尚未确认，因此不跨点组连线；两套结果不按索引强制配对。
- 同名 JSON 的 `lanes` 保存 UFLDv2 坐标，`recorded.lanes` 保存缩放后的录制坐标，两者属于同一输出图像坐标系。还保存内参、匹配状态及时间差；录制点在画面外的坐标保留在 JSON 中，绘图时跳过。
- 录制字节解析采用已对当前目录全部文件验证的布局：小端 topic header `0x00010005`、记录数，以及每条记录的三个 `uint32` 字段和随后 `(字段 1 + 字段 2)` 对 `float32` 坐标（字段从 0 编号）。原始字段保存在 `record_header`，不推断其语义；结构异常会报告失败。
- 可以加 `--limit 10` 先预览前 10 帧；`--input` 也支持单张图片。省略 `--camera-param` 和 `--recorded-contours` 时保留普通图像推理方式。
