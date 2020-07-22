# Textsnake.pytorch
A TextSnake implementation based on https://github.com/princewang1994/TextSnake.pytorch

### Training

#### Train from scratch
```shell script
python train.py thai_name --dataset thai_name --max_epoch 400 --batch_size 8 --num_workers 0 --save_freq 30
```

#### Train with pre-trained model
```shell script
python train.py thai_name --dataset thai_name --max_epoch 400 --batch_size 8 --num_workers 0 --save_freq 30 --resume save/synthtext_pretrain/textsnake_vgg_0.pth
```

### Inference

#### Run inference with CPU
```shell script
python inference.py thai_name --checkepoch 60 --img_root data/thai_name_test/images --cuda False
```

#### Run inference with GPU
```shell script
python inference.py thai_name --checkepoch 60 --img_root data/thai_name_test/images --cuda False
```