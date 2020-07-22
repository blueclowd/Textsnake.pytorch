
import os
import json


def convert(image_folder, txt_folder, json_path):

    image_names = os.listdir(image_folder)
    image_names.sort()

    vgg_json_dict = {}
    for image_name in image_names:

        print(image_name)

        file_size = os.path.getsize(os.path.join(image_folder, image_name))

        regions = []

        with open(os.path.join(txt_folder, image_name.replace('jpg', 'txt'))) as txt_file:

            lines = txt_file.read().splitlines()

            for line_idx in range(1, len(lines)):

                region = {}

                line = lines[line_idx]

                splited = line.split('"')

                transcript = splited[-2]

                coordinates = splited[0].rstrip(",").split(',')

                xs = coordinates[0::2]
                ys = coordinates[1::2]

                region['shape_attributes'] = {'name': 'polygon',
                                              'all_points_x': [int(x) for x in xs],
                                              'all_points_y': [int(y) for y in ys]}

                region['region_attributes'] = {'transcript': transcript.replace("###", "#"),
                                               'orientation': 'm'}

                regions.append(region)

        vgg_json_dict[image_name + str(file_size)] = {'size': file_size,
                                                      'filename': image_name,
                                                      'regions': regions}

    with open(json_path, 'w') as json_file:

        json.dump(vgg_json_dict, json_file)







if __name__=="__main__":

    convert('../data/ctw1500/images/train', '../data/ctw1500/gt/ctw1500_e2e_train', '../data/ctw1500/gt/train/train.json')







