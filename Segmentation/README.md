# ADE20k Semantic Segmentation with DeepMIM

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning with DeepMIM-CLIP
Command:
```
bash tools/dist_train.sh \
configs/mae/upernet_mae_base_12_512_slide_160k_ade20k.py 8 --seed 0 --work-dir ./ckpt/ \
--options model.pretrained="/path/to/DeepMIM-CLIP-PT.pth"
```
Expected results [log](./log/DeepMIM-Seg.log) :
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 53.05 | 64.18 | 84.73 |
+--------+-------+-------+-------+
``` 

## Checkpoint
The checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/1VLJX93RTnCLvIThLxmp71eBsm41HP0sw?usp=sharing)

## Acknowledgement
This repository is built using [mae segmentation](https://github.com/implus/mae_segmentation), [mmseg](https://github.com/open-mmlab/mmsegmentation)
