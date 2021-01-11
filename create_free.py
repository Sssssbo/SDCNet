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
import cv2
import numpy as np

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, MTDD_test_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from datasets import TestFolder_joint
import joint_transforms
from model import HSNet_single1, HSNet_single1_ASPP, HSNet_single1_NR, HSNet_single2, SDMS_A, SDMS_C

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

ckpt_path = './ckpt' 
test_path = './test_ECSSD.csv'


def main():
    img = np.zeros((512, 512),dtype = np.uint8)
    img2 = cv2.imread('./0595.PNG', 0)
    cv2.imshow('img',img2)
    #cv2.waitKey(0)
    print(img, img2)
    Image.fromarray(img).save('./free.png')
            


if __name__ == '__main__':
    main()
