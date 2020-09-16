# Notes

## step1: 稀疏训练
1. `-sr`用于开启稀疏训练
2. `--prune 0`适用于`prune.py`, `--prune 1`适用于其他剪枝策略
3. 训练保存的pt权重包含epoch信息, 可通过`python -c "from models import *; convert('cfg/yolov3-tiny-warship-quant-large-anchors.cfg', 'weights/last-16,16_16.pt')"`
转换为darknet weights去除掉epoch信息, 使用darknet weights从epoch 0开始稀疏训练
4. 稀疏训练
   ```
   python train.py --cfg cfg/yolov3-tiny-1branch-256-quant-thick128.cfg --data data/warship.data --weights weights/last-1branch-256-quant-thick128.weights --epochs 300 --batch-size 32 -sr --s 0.001 --prune 1 --multi-scale
   ```
5. 稀疏训练有3种策略, 默认的是network slimming论文里的策略, 具体参考README

## step2: 通道剪枝
1. 三种策略: `prune.py`, `shortcut_prune.py`, `slim_prune.py`
2. slim_prune用于yolov3-tiny剪枝
   ```
   python slim_prune.py --cfg cfg/yolov3-tiny-1branch-256-quant-thick.cfg --data data/warship.data --weights weights/best.pt --global_percent 0.5 --layer_keep 0.2
   ```

## step3: finetune
1. 若需要对通道剪枝后的模型进行微调, 使用如下命令
   ```
   python train.py --cfg cfg/prune_0.2_keep_0.1_yolov3-tiny-1branch-256-quant-thick.cfg --data data/warship.data --weights weights/prune_0.2_keep_0.1_last-16bit-256i-256d-1branch-256c-sr.weights --multi-scale
   ```
2. 微调时加入知识蒸馏
   ```
   python train.py --cfg cfg/prune_0.5_keep_0.01_yolov3-tiny-warship-quant-large-anchors.cfg --data data/warship.data --weights weights/prune_0.5_keep_0.01_last-16,16_16-sr.weights --epochs 100 --batch-size 32 --t_cfg cfg/yolov3-tiny-warship-large-anchors.cfg --t_weights weights/last-fp32-large-anchors.pt
   ```   

## Commands

- Pytorch转Darknet
  ```
  python -c "from models import *; convert('cfg/prune_0.5_keep_0.01_yolov3-tiny-warship-quant-large-anchors.cfg', 'weights/last.pt')"
  ```