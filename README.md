## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.

## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'

./tools/dist_train.sh configs/dcfa/dcfa_r50_fpn_1x.py 4
```

## Inference

```python
./tools/dist_test.sh configs/dcfa/dcfa_r50_fpn_1x.py work_dirs/dcfa_r50_fpn_1x/latest.pth  4 --eval bbox
```

## Speed Test (FPS)

```python
CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/dcfa/dcfa_r50_fpn_1x.py work_dirs/dcfa_r50_fpn_1x/latest.pth
```

## Pretrained Models

Baidu cloud drive: https://pan.baidu.com/s/1n95GWt3Ek5J-FP9MCGLTlA

Download password: ydby
