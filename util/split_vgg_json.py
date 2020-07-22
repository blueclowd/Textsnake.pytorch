
import json
import os

def split(vgg_json_path, output_json_folder):

    with open(vgg_json_path) as vgg_json_file:

        data = json.load(vgg_json_file)

        for key, value in data.items():

            file_name = value['filename']

            # Check if missing field
            regions = value['regions']
            for region in regions:
                if 'orientation' not in region['region_attributes'] or 'transcript' not in region['region_attributes']:

                    raise Exception(file_name)

            with open(os.path.join(output_json_folder, file_name.replace('jpg', 'json')), 'w') as output_json_file:

                json.dump({key: value}, output_json_file)


if __name__ == "__main__":

    split("../data/thai_name/gt/train_name_address.json", "../data/thai_name/gt/train_name_address")