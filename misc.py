import numpy as np
import os
import pylab as pl

#import pydensecrf.densecrf as dcrf


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt) #t is sum of 1

    precision, recall, TPR, FP = [], [], [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1
        #false_pred = np.zeros(prediction.shape)
        #false_prediction[prediction < threshold] = 1
        a = prediction.shape
        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        #for roc
        #fp = np.sum(false_pred * hard_gt)
        #tpr = (tp + eps)/(a + eps)
        fp = p - tp
        #TPR.append(tpr)
        FP.append(fp)
        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae#, TPR, FP


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def cal_sizec(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4
    #print(gt.shape)

    prediction = prediction / 255.
    gt = gt / 255.

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt) #t is sum of 1

    precision, recall, TPR, FP = [], [], [], []
    # calculating precision and recall at 255 different binarizing thresholds
    best_threshold = 0
    best_F = 0
    for threshold in range(256):
        threshold = threshold / 255.

        gt_size = np.ones(prediction.shape)
        a = np.sum(gt_size)
        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        #print(a, p)
        precision = (tp + eps) / (p + eps)
        recall = (tp + eps) / (t + eps)

        beta_square = 0.3
        fmeasure = (1 + beta_square) * precision * recall / (beta_square * precision + recall)
        if fmeasure > best_F:
            best_threshold = threshold*255
            best_F = fmeasure
            sm_size = p / a
    if 0 <= sm_size < 0.1:
        sizec = 0
    elif 0.1 <= sm_size < 0.2:
        sizec = 1
    elif 0.2 <= sm_size < 0.3:
        sizec = 2
    elif 0.3 <= sm_size < 0.4:
        sizec = 3
    elif 0.4 <= sm_size <= 1.0:
        sizec = 4

    return sizec, best_threshold#, TPR, FP


def cal_sc(gt):
    # input should be np array with data type uint8
    assert gt.dtype == np.uint8

    eps = 1e-4

    gt = gt / 255.
    #print(gt.shape)
    img_size = np.ones(gt.shape)
    a = np.sum(img_size)

    hard_gt = np.zeros(gt.shape)
    hard_gt[gt > 0.5] = 1
    p = np.sum(hard_gt)
    b = np.sum(gt)
    sm_size = float(p) / float(a)
    #print(p, a, sm_size, b)
    #print(gt)
    if 0 <= sm_size < 0.1:
        sizec = 0
    elif 0.1 <= sm_size < 0.2:
        sizec = 1
    elif 0.2 <= sm_size < 0.3:
        sizec = 2
    elif 0.3 <= sm_size < 0.4:
        sizec = 3
    elif 0.4 <= sm_size <= 1.0:
        sizec = 4

    return sizec

def pr_cruve(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    r = [a[1] for a in zip(precision, recall)]
    p = [a[0] for a in zip(precision, recall)]
    pl.title('PR curve')
    pl.xlabel('Recall')
    pl.xlabel('Precision')
    pl.plot(r, p)
    pl.show()

# for define the size type of the salient object
def size_aware(gt):
    assert gt.dtype == np.uint8
    eps = 1e-4
    gt = gt / 255.

    hard_gt = np.zeros(gt.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)
    pic = np.size(hard_gt)
    rate = t/pic
    return rate

# # codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
# def crf_refine(img, annos):
#     def _sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     assert img.dtype == np.uint8
#     assert annos.dtype == np.uint8
#     assert img.shape[:2] == annos.shape

#     # img and annos should be np array with data type uint8

#     EPSILON = 1e-8

#     M = 2  # salient or not
#     tau = 1.05
#     # Setup the CRF model
#     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

#     anno_norm = annos / 255.

#     n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
#     p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

#     U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
#     U[0, :] = n_energy.flatten()
#     U[1, :] = p_energy.flatten()

#     d.setUnaryEnergy(U)

#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

#     # Do the inference
#     infer = np.array(d.inference(1)).astype('float32')
#     res = infer[1, :]

#     res = res * 255
#     res = res.reshape(img.shape[:2])
#     return res.astype('uint8')
