# Textsnake.pytorch
A TextSnake implementation based on https://github.com/princewang1994/TextSnake.pytorch

### Training

#### Train from scratch
```
python train.py thai_name --dataset thai_name --max_epoch 400 --batch_size 8 --num_workers 0 --save_freq 30
```

#### Train with pre-trained model
```
python train.py thai_name --dataset thai_name --max_epoch 400 --batch_size 8 --num_workers 0 --save_freq 30 --resume save/synthtext_pretrain/textsnake_vgg_0.pth
```

### Inference
