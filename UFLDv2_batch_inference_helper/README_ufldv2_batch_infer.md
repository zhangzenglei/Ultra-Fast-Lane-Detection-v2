# UFLDv2 本地图库推理（Ubuntu 22.04 / NVIDIA GPU）

脚本适配 https://github.com/zhangzenglei/Ultra-Fast-Lane-Detection-v2 ，针对原版 `demo.py` 不支持任意图片文件夹的问题，提供目录递归处理、可视化 JPG 和像素坐标 JSON。

## 1. 检查显卡

```bash
nvidia-smi
python3 --version
```

## 2. 安装环境并克隆

```bash
sudo apt update
sudo apt install -y git python3.10-venv

git clone https://github.com/zhangzenglei/Ultra-Fast-Lane-Detection-v2.git
cd Ultra-Fast-Lane-Detection-v2
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==1.26.4'
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'opencv-python==4.10.0.84' pillow tqdm tensorboard addict pathspec
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")'
```

> 上述 PyTorch 组合适合常见 Ubuntu 22.04 + NVIDIA GPU；若驱动或 GPU 架构不兼容，需要按本机硬件选择其他 PyTorch/CUDA 组合。

## 3. 推理专用源码微调

原版 `utils/common.py` 在导入时立即加载用于训练的 NVIDIA DALI 与 `my_interp`，即使只做推理也会受影响；把导入改为仅在进入训练加载器时执行。

原版 `model/model_culane.py`/`model/model_tusimple.py` 创建模型时会下载 ImageNet 预训练 Backbone；推理已有完整 `.pth` 不需要重复下载。

在项目根目录运行：

```bash
python - <<'PY'
from pathlib import Path
p = Path('utils/common.py')
s = p.read_text()
lines = s.splitlines(keepends=True)
lines = [line for line in lines if line.strip() != 'from data.dali_data import TrainCollect']
s = ''.join(lines)
s = s.replace('def get_train_loader(cfg):\n', 'def get_train_loader(cfg):\n    from data.dali_data import TrainCollect\n')
p.write_text(s)
for name in ('model/model_culane.py', 'model/model_tusimple.py'):
    p = Path(name)
    s = p.read_text().replace('parsingNet(pretrained = True,', 'parsingNet(pretrained = False,')
    p.write_text(s)
print('Patched for inference-only use')
PY
```

只要模型是对应训练好的完整 checkpoint，禁用 ImageNet 初始权重不会影响最终推理；完整模型参数会从 checkpoint 加载。

## 4. 下载权重

选择官方 CULane ResNet34 预训练模型：
https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing

保存成 `weights/culane_res34.pth`：

```bash
mkdir -p weights
ls -lh weights/culane_res34.pth
```

## 5. 放入批量推理脚本并运行

把本文件配套的 `batch_infer_ufldv2.py` 复制到仓库根目录。

```bash
python batch_infer_ufldv2.py \
  --config configs/culane_res34.py \
  --weights weights/culane_res34.pth \
  --input /home/$USER/Data/road_images \
  --output /home/$USER/Data/road_results
```

输入可以是单张图片，也可以是目录。目录模式递归处理 `.jpg/.jpeg/.png/.bmp/.webp`，保留子目录结构。

每张图片输出 `xxx_lanes.jpg`（检测点/曲线可视化）和 `xxx_lanes.json`（每条车道线的像素坐标）。JSON 中空的 `points` 数组代表该车道槽位未通过存在性筛选。

## 6. 注意事项

- 本脚本按原版 `demo.py` 的预处理、车道坐标解码和存在性筛选逻辑实现；不是新的训练模型，不能自动保证鱼骨线场景准确率。
- CULane 模型默认输入网络尺寸为 1600x320（预处理先缩放到 1600x533，再裁剪底部 320），对非 CULane 相机长宽比和鱼眼畸变，效果需要实测。
- 模型 F1 需要标注真值评估；保存图片上的绿色等彩色线不能当作 Ground Truth。
- 如果后续要训练，安装 DALI 和 `my_interp`，即可使用被延迟的导入；要恢复仓库原始代码可执行 `git restore utils/common.py model/model_culane.py model/model_tusimple.py`。
