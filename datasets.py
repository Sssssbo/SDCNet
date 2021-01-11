import os
import os.path

import torch.utils.data as data
from PIL import Image


class ImageFolder_joint(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, label_list, joint_transform=None, transform=None, target_transform=None):
        imgs = []
        self.label_list = label_list
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['gt_path'], row['label']))
        self.imgs = imgs
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_path, gt_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

class ImageFolder_joint_for_edge(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, label_list, joint_transform=None, transform=None, target_transform=None):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['gt_path'], row['label']))
        self.imgs = imgs
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, label = self.imgs[index]
        edge_path = "."+gt_path.split(".")[1]+"_edge."+gt_path.split(".")[2]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        target_edge = Image.open(edge_path).convert('L')
        if self.joint_transform is not None:
            if img.size != target.size or img.size != target_edge.size:
                print("error path:", img_path, gt_path)
                print("size:", img.size, target.size, target_edge.size)
            img, target, target_edge = self.joint_transform(img, target, target_edge)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target_edge = self.target_transform(target_edge)

        return img, target, target_edge, label

    def __len__(self):
        return len(self.imgs)

class TestFolder_joint(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, label_list, joint_transform=None, transform=None, target_transform=None):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['gt_path'], row['label']))
        self.imgs = imgs
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label, img_path

    def __len__(self):
        return len(self.imgs)


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, img_name + '.jpg'), os.path.join(root, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
