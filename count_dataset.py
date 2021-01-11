import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

path_list = ['msra10k', 'ECSSD', 'DUT-OMROM', 'DUTS-TR', 'DUTS-TE', 'HKU-IS', 'PASCAL-S', 'SED2', 'SOC', 'SOD', 'THUR-15K']

def main():
    Dataset, Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, Class10, Total = [], [], [], [], [], [], [], [], [], [], [], [], []
    for data_path in path_list:
        test_path = './SOD_label/label_' + data_path + '.csv'
        print('Evalute for ' + test_path)
        test_data = pd.read_csv(test_path)
        imgs = []
        num, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for index, row in test_data.iterrows():
            imgs.append((row['img_path'], row['gt_path'], row['label']))
            img_path, gt_path, label = imgs[index]

            if label == 0:
                c0 += 1
            elif label == 1:
                c1 += 1
            elif label == 2:
                c2 += 1
            elif label == 3:
                c3 += 1
            elif label == 4:
                c4 += 1
            elif label == 5:
                c5 += 1
            elif label == 6:
                c6 += 1
            elif label == 7:
                c7 += 1
            elif label == 8:
                c8 += 1
            elif label == 9:
                c9 += 1
            elif label == 10:
                c10 += 1
            num += 1
        print('[Class0 %.f], [Class1 %.f], [Class2 %.f], [Class3 %.f]\n'\
              '[Class4 %.f], [Class5 %.f], [Class6 %.f], [Class7 %.f]\n'\
              '[Class8 %.f], [Class9 %.f], [Class10 %.f], [Total %.f]\n'%\
              (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, num)
              )
        Dataset.append(data_path)
        Class0.append(c0)
        Class1.append(c1)
        Class2.append(c2)
        Class3.append(c3)
        Class4.append(c4)
        Class5.append(c5)
        Class6.append(c6)
        Class7.append(c7)
        Class8.append(c8)
        Class9.append(c9)
        Class10.append(c10)
        Total.append(num)

    label_file = pd.DataFrame({'Datasets': Dataset, 'Class 0': Class0, 'Class 1': Class1, 'Class 2': Class2, 'Class 3': Class3, 'Class 4': Class4, 'Class 5': Class5, 'Class 6': Class6, 'Class 7': Class7, 'Class 8': Class8, 'Class 9': Class9, 'Class 10': Class10, 'Num of Pic': Total})
    label_file = label_file[['Datasets', 'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 'Num of Pic']]

    label_file.to_csv('./Dataset_statistics.csv', index=False)

if __name__ == '__main__':
    main()
