import scipy.io as io
import numpy as np
import os
import re
import json

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class CustomText(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None):

        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'images', 'train' if is_training else 'val')
        self.annotation_root = os.path.join(data_root, 'gt', 'train' if is_training else 'val')
        self.image_list = os.listdir(self.image_root)
        # self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []

        key = 'polygt'
        if 'gt' in annot:
            key = 'gt'

        for cell in annot[key]:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def load_json(self, json_path):
        '''
        Load and parse vgg json
        :param json_path: json file path
        :return: list of TextInstance
        '''

        polygons = []

        with open(json_path, encoding="utf-8") as json_file:

            data = json.load(json_file)

            for id, value in data.items():

                file_name = value['filename']
                regions = value['regions']

                for region in regions:

                    orientation = region['region_attributes']['orientation']
                    transcript = region['region_attributes']['transcript']

                    orientation = orientation if orientation != '' else 'c'
                    transcript = transcript if transcript != '' else '#'

                    xs = region['shape_attributes']['all_points_x']
                    ys = region['shape_attributes']['all_points_y']

                    pts = np.stack([xs, ys]).T.astype(np.int32)
                    polygons.append(TextInstance(pts, orientation, transcript))

        return polygons

    def parse_txt(self, txt_path):

        with open(txt_path, 'r') as txt_file:

            lines = txt_file.readlines()

        polygons = []
        for line in lines:
            print('line:' + line)
            x_seg, y_seg, ori_seg, transcript_seg = re.findall(r'\[[\-\d\w\s\'\,\.\#\&\$\:]*\]', line)

            xs = re.findall(r'[\d]+', x_seg)
            ys = re.findall(r'[\d]+', y_seg)

            ori = re.findall(r'\'[\w\#]*\'', ori_seg)
            if not ori:
                ori = 'c'
            else:
                ori = ori[0]

            text = re.findall(r'\'[\w\d\#]*\'', transcript_seg)
            text = text if len(text) > 0 else '#'

            if len(xs) < 4:  # too few points
                continue
            pts = np.stack([xs, ys]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        polygons = self.load_json(os.path.join(self.annotation_root, image_id.replace('jpg', 'json')))

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()
        try:
            return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

        except Exception as e:

            print(e)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = CustomText(
        data_root='data/custom-text',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[idx]
        print(idx, img.shape)