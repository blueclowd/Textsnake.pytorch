import scipy.io as io
import os
import json

def read_mat(mat_path):
    mat_content = io.loadmat(mat_path)

    print(mat_content)

if __name__=="__main__":

    read_mat('/Users/hungting/PycharmProjects/TextSnake.pytorch/data/icdar_art/gt/train/poly_gt_img11.mat')