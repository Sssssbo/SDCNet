import numpy as np
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure, cal_sizec, cal_sc
from datasets import TestFolder_joint
import joint_transforms
from model import R3Net, SDCNet
torch.manual_seed(2021)

# set which gpu to use
torch.cuda.set_device(6)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'SDCNet' 

msra10k_path = './SOD_label/label_msra10k.csv'
ecssd_path = './SOD_label/label_ECSSD.csv'
dutomrom_path = './SOD_label/label_DUT-OMROM.csv'
dutste_path = './SOD_label/label_DUTS-TE.csv'
hkuis_path = './SOD_label/label_HKU-IS.csv'
pascals_path = './SOD_label/label_PASCAL-S.csv'
sed2_path = './SOD_label/label_SED2.csv'
socval_path = './SOD_label/label_SOC-Val.csv'
sod_path = './SOD_label/label_SOD.csv'
thur15k_path = './SOD_label/label_THUR-15K.csv'

args = {
    'snapshot': '30000',  # your snapshot filename (exclude extension name)
    'save_results': True,  # whether to save the resulting masks
    'test_mode': 1
}
joint_transform = joint_transforms.Compose([
    #joint_transforms.RandomCrop(300),
    #joint_transforms.RandomHorizontallyFlip(),
    #joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test ={'ECSSD': ecssd_path,'SOD': sod_path, 'DUTS-TE': dutste_path} #{'DUTS-TE': dutste_path,'ECSSD': ecssd_path,'SOD': sod_path, 'SED2': sed2_path, 'PASCAL-S': pascals_path, 'HKU-IS': hkuis_path, 'DUT-OMROM': dutomrom_path}

def main():
    net = SDCNet(num_classes = 5).cuda()

    print('load snapshot \'%s\' for testing, mode:\'%s\'' % (args['snapshot'], args['test_mode']))
    print(exp_name)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()

    results = {}

    with torch.no_grad():

        for name, root in to_test.items():
            print('load snapshot \'%s\' for testing %s' %(args['snapshot'], name))

            test_data = pd.read_csv(root)
            test_set = TestFolder_joint(test_data, joint_transform, img_transform, target_transform)
            test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
            
            precision0_record, recall0_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 
            precision1_record, recall1_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 
            precision2_record, recall2_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            precision3_record, recall3_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 
            precision4_record, recall4_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 
            precision5_record, recall5_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 
            precision6_record, recall6_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)] 

            mae0_record = AvgMeter()
            mae1_record = AvgMeter()
            mae2_record = AvgMeter()
            mae3_record = AvgMeter()
            mae4_record = AvgMeter()
            mae5_record = AvgMeter()
            mae6_record = AvgMeter()

            n0, n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0, 0

            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s' % (name, args['snapshot'])))

            for i, (inputs, gt, labels, img_path) in enumerate(tqdm(test_loader)): 

                shape = gt.size()[2:]

                img_var = Variable(inputs).cuda()
                img = np.array(to_pil(img_var.data.squeeze(0).cpu()))

                gt = np.array(to_pil(gt.data.squeeze(0).cpu()))
                sizec = labels.numpy()
                pred2021 = net(img_var, sizec)

                pred2021 = F.interpolate(pred2021, size=shape, mode='bilinear', align_corners=True)
                pred2021 = np.array(to_pil(pred2021.data.squeeze(0).cpu()))

                if labels == 0:
                    precision1, recall1, mae1 = cal_precision_recall_mae(pred2021, gt)
                    for pidx, pdata in enumerate(zip(precision1, recall1)):
                        p, r = pdata
                        precision1_record[pidx].update(p)
                        #print('Presicion:', p, 'Recall:', r)
                        recall1_record[pidx].update(r)
                    mae1_record.update(mae1)
                    n1 += 1

                elif labels == 1:
                    precision2, recall2, mae2 = cal_precision_recall_mae(pred2021, gt)
                    for pidx, pdata in enumerate(zip(precision2, recall2)):
                        p, r = pdata
                        precision2_record[pidx].update(p)
                        #print('Presicion:', p, 'Recall:', r)
                        recall2_record[pidx].update(r)
                    mae2_record.update(mae2)
                    n2 += 1

                elif labels == 2:
                    precision3, recall3, mae3 = cal_precision_recall_mae(pred2021, gt)
                    for pidx, pdata in enumerate(zip(precision3, recall3)):
                        p, r = pdata
                        precision3_record[pidx].update(p)
                        #print('Presicion:', p, 'Recall:', r)
                        recall3_record[pidx].update(r)
                    mae3_record.update(mae3)
                    n3 += 1

                elif labels == 3:
                    precision4, recall4, mae4 = cal_precision_recall_mae(pred2021, gt)
                    for pidx, pdata in enumerate(zip(precision4, recall4)):
                        p, r = pdata
                        precision4_record[pidx].update(p)
                        #print('Presicion:', p, 'Recall:', r)
                        recall4_record[pidx].update(r)
                    mae4_record.update(mae4)
                    n4 += 1

                elif labels == 4:
                    precision5, recall5, mae5 = cal_precision_recall_mae(pred2021, gt)
                    for pidx, pdata in enumerate(zip(precision5, recall5)):
                        p, r = pdata
                        precision5_record[pidx].update(p)
                        #print('Presicion:', p, 'Recall:', r)
                        recall5_record[pidx].update(r)
                    mae5_record.update(mae5)
                    n5 += 1

                precision6, recall6, mae6 = cal_precision_recall_mae(pred2021, gt)
                for pidx, pdata in enumerate(zip(precision6, recall6)):
                    p, r = pdata
                    precision6_record[pidx].update(p)
                    recall6_record[pidx].update(r)
                mae6_record.update(mae6)

                img_name = os.path.split(str(img_path))[1]
                img_name = os.path.splitext(img_name)[0]
                n0 += 1

                
                if args['save_results']:
                    Image.fromarray(pred2021).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (
                        name, args['snapshot']), img_name + '_2021.png'))
                    
            fmeasure1 = cal_fmeasure([precord.avg for precord in precision1_record],
                                    [rrecord.avg for rrecord in recall1_record])
            fmeasure2 = cal_fmeasure([precord.avg for precord in precision2_record],
                                    [rrecord.avg for rrecord in recall2_record])
            fmeasure3 = cal_fmeasure([precord.avg for precord in precision3_record],
                                    [rrecord.avg for rrecord in recall3_record])
            fmeasure4 = cal_fmeasure([precord.avg for precord in precision4_record],
                                    [rrecord.avg for rrecord in recall4_record])
            fmeasure5 = cal_fmeasure([precord.avg for precord in precision5_record],
                                    [rrecord.avg for rrecord in recall5_record])
            fmeasure6 = cal_fmeasure([precord.avg for precord in precision6_record],
                                    [rrecord.avg for rrecord in recall6_record])
            results[name] = {'fmeasure1': fmeasure1, 'mae1': mae1_record.avg,'fmeasure2': fmeasure2, 
                                    'mae2': mae2_record.avg, 'fmeasure3': fmeasure3, 'mae3': mae3_record.avg, 
                                    'fmeasure4': fmeasure4, 'mae4': mae4_record.avg, 'fmeasure5': fmeasure5, 
                                    'mae5': mae5_record.avg, 'fmeasure6': fmeasure6, 'mae6': mae6_record.avg}

            print('test results:')
            print('[fmeasure1 %.3f], [mae1 %.4f], [class1 %.0f]\n'\
                  '[fmeasure2 %.3f], [mae2 %.4f], [class2 %.0f]\n'\
                  '[fmeasure3 %.3f], [mae3 %.4f], [class3 %.0f]\n'\
                  '[fmeasure4 %.3f], [mae4 %.4f], [class4 %.0f]\n'\
                  '[fmeasure5 %.3f], [mae5 %.4f], [class5 %.0f]\n'\
                  '[fmeasure6 %.3f], [mae6 %.4f], [all %.0f]\n'%\
                  (fmeasure1, mae1_record.avg, n1, fmeasure2, mae2_record.avg, n2, fmeasure3, mae3_record.avg, n3, fmeasure4, mae4_record.avg, n4, fmeasure5, mae5_record.avg, n5, fmeasure6, mae6_record.avg, n0))


def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = float(PRED_CORRECT_COUNT / PRED_COUNT)
        return final_acc * 100, PRED_COUNT


if __name__ == '__main__':
    main()
