# Textsnake.pytorch
A TextSnake implementation based on https://github.com/princewang1994/TextSnake.pytorch

### Training

#### Annotation format: [vgg](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)
Example:
```json
{"thai_nid_0001_1.jpg215470": {"filename": "thai_nid_0001_1.jpg", "size": 215470, "regions": [{"shape_attributes": {"name": "polygon", "all_points_x": [198, 199, 469, 466, 361, 321], "all_points_y": [217, 259, 253, 203, 204, 213]}, "region_attributes": {"transcript": "temp", "orientation": "c"}}], "file_attributes": {}}}
```

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
python inference.py thai_name --checkepoch 60 --img_root data/thai_name_test/images --cuda True
```