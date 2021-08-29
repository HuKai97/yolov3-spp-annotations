import torch
import torch.nn as nn
import math
import time
import numpy as np
import torchvision


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def compute_loss(p, targets, model):
    """
    Args:
        p: predictions 预测框  [4, 3, 23, 23, 25] + [4, 3, 46, 46, 25] + [4, 3, 96, 96, 25]
           [batch_size, anchor_num, grid, grid, xywh + obj + classes]
        targets: 真实框 [22, 6]
        model:
    Returns:

    """
    device = p[0].device
    lcls = torch.zeros(1, device=device)  # Tensor(0)
    lbox = torch.zeros(1, device=device)  # Tensor(0)
    lobj = torch.zeros(1, device=device)  # Tensor(0)
    # tcls: 得到筛选后的target(正样本)对应的class
    # tbox: 得到gt box相对anchor的x,y偏移量以及w,h
    # indices: (b,a,gj,ji)  b: 一个batch中的下标    a: 代表所选中的正样本的anchor的下标    gj, gi: 代表所选中的栅格的左上角坐标
    # anchors: 得到每一个yolo层筛选后target(正样本)对应的anchors列表
    # 用groudtruth（target）与anchor IOU的阈值来决定某个框是否在这层featuremap上进行预测,
    # 会出现groundtruth在多层feature map上预测，这样子多层有输出，最后通过NMS进行抑制.
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets  得到正样本
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    # BCEWithLogitsLoss = Sigmoid-BCELoss合成一步  先用对output进行sigmoid再对output和target进行交叉熵
    # pos_weight: 用于设置损失的class权重，用于缓解样本的不均衡问题
    # reduction: 设为"sum"表示对样本进行求损失和；设为"mean"表示对样本进行求损失的平均值；而设为"none"表示对样本逐个求损失，输出与输入的shape一样。
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 标签平滑  cp: positive label smoothing BCE targets     cn: negative label smoothing BCE targets
    cp, cn = smooth_BCE(eps=0) # 平滑系数eps=0说明不采用标签平滑  要用通常为0.1

    # focal loss  这里暂时没用到 一般是用到分类损失中
    g = h['fl_gamma']  # focal loss gamma  hpy中g默认=0
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    # layer index(0,1,2), layer predictions [[batch_size, anchor_num, grid, grid, xywh + obj + classes]]
    for i, pi in enumerate(p):  # pi为第i层yolo层输出 [4,3,23,23,25]
        # 获得每一层的正样本:  b: image_index, a: anchor_index, gj和gi:代表所选中真实框的栅格的左上角坐标y,x
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj [4,3,23,23]

        nb = b.shape[0]  # number of targets 真实框的个数
        if nb:
            # 对应匹配到正样本的预测信息
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  [38,25]

            # lbox: 位置损失  GIoU Loss
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False)  # iou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # obj model.gr=1  置为giou(有物体的地方，置信度为giou)   model.gr在train.py中定义
            # model.gr: giou loss ratio (obj_loss = 1.0 or giou)
            # model.gr=1 obj_loss=giou;  model.gr=0, obj_loss=1
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # lcls: 类别损失 BCELoss
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE class loss  如果打开focal loss ，这里会调用focal loss

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # lobj: 置信度损失（有无物体） lobj是对所有prediction区域计算的
        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    # 乘上每种损失的对应权重
    lbox *= h['giou']   # 3.54
    lobj *= h['obj']    # 102.88
    lcls *= h['cls']    # 9.35

    # loss = lbox + lobj + lcls
    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}

def build_targets(p, targets, model):
    """
    在每个yolo层将预设的每一个anchor和ground truth进行匹配，得到每个anchor的正样本
    规则:
        1.如果一个预测框与所有的GroundTruth的最大 IoU < ignore_thresh时，那这个预测框就是负样本
        2.如果Ground Truth的中心点落在一个区域中，该区域就负责检测该物体。将与该物体有最大IoU的预测框
          作为正样本（注意这里没有用到ignore thresh,即使该最大IoU<ignore thresh也不会影响该预测框为正样本）
    Args:
        p: predictions 预测框  [4, 3, 23, 23, 25] + [4, 3, 46, 46, 25] + [4, 3, 96, 96, 25]
           [batch_size, anchor_num, grid, grid, xywh + obj + classes]
        targets: 真实框 [22, 6] 22: 真实框的数量  6: batch中第几张图（0,1,2,3）,类别,x,y,w,h  xywh都是相对grid坐标 <1
        model:
    Returns:
        tcls: 每个yolo层得到筛选后的target(正样本)对应的class
              len=3  [1,38] [1,30] [1,8]
        tbox: tbox.append(torch.cat((gxy - gij, gwh), 1))
              得到每个yolo层对gt box相对anchor的x,y偏移量以及w,h
              len=3 [38,4] [30,4] [8,4]
        indices: indices.append(b,a,gj,ji)
                 b: 一个batch中的下标    a: 代表所选中的正样本的anchor的下标    gj, gi: 代表所选中的栅格的左上角坐标
                 len=3  3个tensor 每个tensor里又有4个tensor 分别是b:[1,38] a:[1,38] gj:[1,38] gi:[1,38]
                                                               b:[1,30] a:[1,30] gj:[1,30] gi:[1,30]
                                                               b:[1,8] a:[1,8] gj:[1,8] gi:[1,8]
        anch: 每个yolo层得到筛选后target(正样本)对应的anchors列表
              len=3 [38,2] [30,2] [8,2]
    """
    nt = targets.shape[0]  # 真实框的数量
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) # False
    for i, j in enumerate(model.yolo_layers):  # [89, 101, 113]   i,j = 0, 89   1, 101   2, 113
        # 获取该yolo predictor对应的anchors的大小，即anchors大小缩放到预测特征层(23, 23)上的尺度  shape=[3, 2]
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain   gain=[1, 1, 23, 23, 23, 23]  torch.Size([6])
        na = anchors.shape[0]  # number of anchors  3个
        # [3] -> [3, 1] -> [3, nt]
        # anchor tensor, same as .repeat_interleave(nt) [3, 22]   22个 0, 1, 2
        at = torch.arange(na).view(na, 1).repeat(1, nt)

        # Match targets to anchors
        # t存放真实框在特征图上的box信息 包括中心点坐标 宽 高坐标(相对于特征图的坐标)
        # 广播原理 targets=[22,6]  gain=[6] => [6,6]    => t=[22,6]
        a, t, offsets = [], targets * gain, 0

        # 先根据model.hyp['iou_t']= 0.20对target（真实框）进行筛选  筛选出符合该yolo层对应的正样本？
        # 没层有三个anchor,每个anchor都要对target进行筛选  不同anchor可以筛选同一个target
        if nt:  # 如果存在target的话
            # j: [3, nt]
            # 筛选出大于iou_t的iou(相对feature map 的真实框t与anchors)
            # 把yolo层的anchor在该feature map上对应的wh（anchors）和所有预测真实框在该feature map
            # 上对应的wh(t[4:6])做iou, 若大于model.hyp['iou_t']=0.2, 则为正样本保留，否则则为负样本舍弃
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
            # 获取iou大于阈值的anchor与target对应信息
            # a=[1,38]: anchor_index  0表示第一个anchor(包含4张图片)的正样本  同理第二。。。
            # t=[38,6]: 第一个0、1、2、3(4张图片)的部分是第一个anchor的正样本  同理第二。。。
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        # Define
        # long等于to(torch.int64), 数值向下取整
        b, c = t[:, :2].long().T  # b: image_index[38], c: class[38]
        gxy = t[:, 2:4]  # grid xy  [38,2]
        gwh = t[:, 4:6]  # grid wh  [38,2]
        gij = (gxy - offsets).long()  # 匹配targets所在的grid cell左上角坐标
        gi, gj = gij.T  # grid xy indices  gi=x  gj=y

        # Append
        indices.append((b, a, gj, gi))  # image_index, anchor_index, grid左上角y, grid左上角x
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box(真实框)相对anchor的x,y偏移量（<1）以及w,h
        anch.append(anchors[a])  # 每个yolo层正样本对应的anchor框size
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            # 目标的标签数值不能大于给定的目标类别数
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch

def smooth_BCE(eps=0.1):
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # eps 平滑系数  [0, 1]  =>  [0.95, 0.05]
    # return positive, negative label smoothing BCE targets
    # positive label= y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # y_true=1  label_smoothing=eps=0.1
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma   # 参数gamma
        self.alpha = alpha   # 参数alpha
        # reduction: 控制损失输出模式 sum/mean/none 这里定义的交叉熵损失BCE都是mean
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 不知道这句有什么用?  required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # 普通BCE Loss
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作 prob from logits
        # ture=0,p_t=1-p; true=1, p_t=p
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # ture=0, alpha_factor=1-alpha; true=1,alpha_factor=alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        # loss = focus loss(代入公式即可)
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean': # 一般是mean
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 就是普通iou
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# iou giou diou ciou
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """
    Args:
        box1: 预测框
        box2: 真实框
        x1y1x2y2: False
    Returns:
        box1和box2的IoU/GIoU/DIoU/CIoU
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()  # 转置 ？？？

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2  # b1左上角和右下角的x坐标
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2  # b1左下角和右下角的y坐标
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2  # b2左上角和右下角的x坐标
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2  # b2左下角和右下角的y坐标

    # Intersection area  tensor.clamp(0): 将矩阵中小于0的元数变成0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter  # 1e-16: 防止分母为0

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # return GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou

# 筛选正样本
def wh_iou(wh1, wh2):
    """
    把yolo层的anchor在该feature map上对应的wh（anchors）和所有预测真实框在该feature map上对应的wh(t[4:6])做iou,
    若大于model.hyp['iou_t']=0.2, 则为正样本保留，否则则为负样本舍弃  筛选出符合该yolo层对应的正样本
    Args:
        wh1: anchors  [3, 2]
        wh2: target   [22,2]
    Returns:
        wh1 和 wh2 的iou [3, 22]
    """
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]  [3, 1, 2]
    wh2 = wh2[None]  # [1,M,2]     [1, 22, 2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]  [3, 22]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2