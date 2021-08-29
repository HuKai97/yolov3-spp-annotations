import math
import random
import sys
import time

import torch
from torch.cuda import amp
import torch.nn.functional as F
from train_val_utils import distributed_utils as utils
from train_val_utils.coco_eval import CocoEvaluator
from train_val_utils.coco_utils import get_coco_api_from_dataset
from train_val_utils.utils import compute_loss, non_max_suppression, scale_coords


def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs,
                    print_freq, accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=True):
    """
    Args:
        data_loader: len = 1430 1430个batch_size=1个epochs分成一块块的batch_size
        print_freq: 每50个batch在logger中更新
        accumulate: 1、多尺度训练时accumulate个batch改变一次图片的大小
                    2、每训练accumulate*batch_size张图片更新一次权重和学习率
                    第一个epoch  accumulate=1
        img_size: 训练图像的大小
        grid_min, grid_max: 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
        gs: grid_size
        warmup: 用在训练第一个epoch时，这个时候的训练学习率要调小点，慢慢训练

    Returns:
        mloss: 每个epch计算的mloss [box_mean_loss, obj_mean_loss, class_mean_loss, total_mean_loss]
        now_lr: 每个epoch之后的学习率
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, epochs)

    # 模型训练开始第一轮采用warmup训练 慢慢训练
    lr_scheduler = None
    if epoch == 1 and warmup is True:  # 当训练第一轮（epoch=1）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1  # 慢慢训练，每个batch都改变img大小，每个batch都改变权重

    # amp.GradScaler: 混合精度训练
    # GradScaler: 在反向传播前给 loss 乘一个 scale factor,所以之后反向传播得到的梯度都乘了相同的 scale factor
    # scaler: GradScaler对象用来自动做梯度缩放
    # https://blog.csdn.net/l7H9JA4/article/details/114324414?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161944073216780357273770%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161944073216780357273770&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-1-114324414.pc_search_result_cache&utm_term=amp.GradScaler%28enabled%3Denable_amp%29
    enable_amp = True if "cuda" in device.type else False
    scaler = amp.GradScaler(enabled=enable_amp)


    # mean losses [box_mean_loss, obj_mean_loss, class_mean_loss, total_mean_loss]
    mloss = torch.zeros(4).to(device)
    now_lr = 0.  # 本batch的lr
    nb = len(data_loader)  # number of batches
    # imgs: [batch_size, 3, img_size, img_size]
    # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
    # paths: list of img path
    # 这里调用执行batch_size次datasets.__getitem__再执行1次collate_fn
    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)

        # imgs: [4, 3, 736, 736]一个batch的图片
        # targets(真实框): [22, 6] 22: num_object  6: batch中第几张图（0,1,2,3）,类别,x,y,w,h
        imgs = imgs.to(device).float() / 255.0  # 对imgs进行归一化 uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Multi-Scale
        if multi_scale:
            # 每训练accumulate个batch(batch_size*accumulate张图片)，就随机修改一次输入图片大小
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            if ni % accumulate == 0:  #  adjust img_size (67% - 150%) every 1 batch
                # 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
                img_size = random.randrange(grid_min, grid_max + 1) * gs  # img_size = 320~736
            sf = img_size / max(imgs.shape[2:])  # scale factor

            # 如果图片最大边长不等于img_size, 则缩放一个batch图片，并将长和宽调整到32的整数倍
            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with amp.autocast(enabled=enable_amp):
            pred = model(imgs)  # [4, 3, 23, 23, 25] + [4, 3, 46, 46, 25] + [4, 3, 96, 96, 25]

            # loss ['box_loss', 'class_loss', 'obj_loss']
            loss_dict = compute_loss(pred, targets, model)

            losses = sum(loss for loss in loss_dict.values()) # 三个相加

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                    loss_dict_reduced["obj_loss"],
                                    loss_dict_reduced["class_loss"],
                                    losses_reduced)).detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

            # 如果losses_reduced无效，则输出对应图片信息
            if not torch.isfinite(losses_reduced):
                print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
                print("training image path: {}".format(",".join(paths)))
                sys.exit(1)

            losses *= 1. / accumulate  # scale loss

        # 1、backward 反向传播 scale loss 先将梯度放大 防止梯度消失
        scaler.scale(losses).backward()
        # optimize
        # 每训练accumulate*batch_size张图片更新一次权重
        if ni % accumulate == 0:
            # 2、scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)
            # 3、准备着，看是否要增大scaler 不一定更新  看scaler.step(optimizer)的结果，需要更新就更新
            scaler.update()

            # 正常更新权重
            optimizer.zero_grad()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        # 每训练accumulate*batch_size张图片更新一次学习率（只在第一个epoch） warmup=True 才会执行
        if ni % accumulate == 0 and lr_scheduler is not None:
            lr_scheduler.step()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    """
    Args:
        coco: coco api
    Returns:

    """
    n_threads = torch.get_num_threads()  # 8线程
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)  # ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # targets = targets.to(device)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred = model(imgs)[0]  # only get inference result   [4, 5040, 25]
        # [4, 5040, 25] => len=4  [57,6], [5,6], [14,6], [1,6]  6: batch中第几张图（0,1,2,3）,类别,x,y,w,h
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)
        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

            # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)
        model_time = time.time() - model_time

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return result_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
