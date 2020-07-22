import scipy.io as io
import os
import json


def convert_mat_to_json(mat_folder, image_folder, json_path):

    mat_names = os.listdir(mat_folder)

    json_data: dict = {}

    for mat_name in mat_names:

        abs_mat_path = os.path.join(mat_folder, mat_name)

        mat_cotent = io.loadmat(abs_mat_path)

        key = 'polygt'

        regions = []
        for cell in mat_cotent[key]:
            xs = cell[1][0]
            ys = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            region = {}
            shape_attributes: dict = {}
            shape_attributes['name'] = 'polygon'
            xs = [x.item() for x in xs]
            ys = [y.item() for y in ys]

            if len(xs) < 4:

                print(mat_name)

            shape_attributes['all_points_x'] = list(xs)
            shape_attributes['all_points_y'] = list(ys)

            region['shape_attributes'] = shape_attributes

            region_attributes: dict = {}
            region_attributes['transcript'] = text
            region_attributes['orientation'] = ori

            region['region_attributes'] = region_attributes

            regions.append(region)

        file_name = mat_name.replace('mat', 'jpg').replace('poly_gt_', '')
        file_size = os.path.getsize(os.path.join(image_folder, file_name))

        json_data[file_name + str(file_size)] = {'filename': file_name,
                                                 'size': file_size,
                                                 'regions': regions}

    with open(json_path, 'w') as json_file:

        json.dump(json_data, json_file)


if __name__=="__main__":

    convert_mat_to_json('../data/total-text/gt/Train_2',
                        '../data/total-text/Images/Train',
                        '../data/total-text/gt/json/train_2.json')
