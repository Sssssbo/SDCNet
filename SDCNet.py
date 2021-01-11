import datetime
import os
import time

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

import joint_transforms
from config import msra10k_path, MTDD_train_path
from datasets import ImageFolder_joint
from misc import AvgMeter, check_mkdir, cal_sc
from model import R3Net, SDCNet
from torch.backends import cudnn

cudnn.benchmark = True

torch.manual_seed(2021)
torch.cuda.set_device(6)

csv_path = './label_DUTS-TR.csv'
ckpt_path = './ckpt'
exp_name ='SDCNet'

args = {
    'iter_num': 30000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

all_data = pd.read_csv(csv_path)
train_set = ImageFolder_joint(all_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True, drop_last=True)#

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = SDCNet(num_classes = 5).cuda().train() # 
    
    print('training in ' + exp_name)
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    start_time = time.time()
    curr_iter = args['last_iter']
    num_class = [0, 0, 0, 0, 0]
    while True:
        total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        batch_time = AvgMeter()
        end = time.time()
        print('-----begining the first stage, train_mode==0-----')
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gt, labels = data
            print(labels)
            # depends on the num of classes
            cweight = torch.tensor([0.5, 0.75, 1, 1.25, 1.5])
            #weight = torch.ones(size=gt.shape)
            weight = gt.clone().detach()
            sizec = labels.numpy()
            #ta = np.zeros(shape=gt.shape)
            '''
            np.zeros(shape=labels.shape)
            sc = gt.clone().detach()
            for i in range(len(sizec)):
                gta = np.array(to_pil(sc[i,:].data.squeeze(0).cpu()))#
                #print(gta.shape)
                labels[i] = cal_sc(gta)
                sizec[i] = labels[i]
            print(labels)
            '''
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gt = Variable(gt).cuda()
            labels = Variable(labels).cuda()

            #print(sizec.shape)

            optimizer.zero_grad()
            p5, p4, p3, p2, p1, predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9, predict10, predict11 = net(inputs, sizec) # mode=1

            criterion = nn.BCEWithLogitsLoss().cuda()
            criterion2 = nn.CrossEntropyLoss().cuda()

            gt2 = gt.long()
            gt2 = gt2.squeeze(1)

            l5 = criterion2(p5, gt2)
            l4 = criterion2(p4, gt2)
            l3 = criterion2(p3, gt2)
            l2 = criterion2(p2, gt2)
            l1 = criterion2(p1, gt2)

            loss0 = criterion(predict11, gt)
            loss10 = criterion(predict10, gt)
            loss9 = criterion(predict9, gt)
            loss8 = criterion(predict8, gt)
            loss7 = criterion(predict7, gt)
            loss6 = criterion(predict6, gt)
            loss5 = criterion(predict5, gt)
            loss4 = criterion(predict4, gt)
            loss3 = criterion(predict3, gt)
            loss2 = criterion(predict2, gt)
            loss1 = criterion(predict1, gt)

            total_loss = l1 + l2 + l3 + l4 + l5 + loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(l5.item(), batch_size)
            loss0_record.update(loss0.item(), batch_size)

            curr_iter += 1.0
            batch_time.update(time.time() - end)
            end = time.time()

            log = '[iter %d], [R1/Mode0], [total loss %.5f]\n' \
                  '[l5 %.5f], [loss0 %.5f]\n' \
                  '[lr %.13f], [time %.4f]' % \
                  (curr_iter, total_loss_record.avg, loss1_record.avg, loss0_record.avg, optimizer.param_groups[1]['lr'],
                   batch_time.avg)
            print(log)
            print('Num of class:', num_class)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                total_time = time.time() - start_time
                print(total_time)
                return


if __name__ == '__main__':
    main()
