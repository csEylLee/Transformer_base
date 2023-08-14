#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python implementation recognition AP for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
(c) icube, unistra
"""
# %%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from sklearn.metrics import average_precision_score
import warnings
import sys
from ivtmetrics.disentangle import Disentangle


# %%%%%%%%%% RECOGNITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Recognition(Disentangle):
    """
    Class: compute (mean) Average Precision
    @args
    ----
        num_class: int, optional. The number of class of the classification task (default = 100)
    @attributes
    ----------
    predictions:    2D array
        holds the accumulated predictions before a reset()
    targets:        2D array
        holds the accumulated groundtruths before a reset()
    @methods
    -------
    GENERIC
    -------
    reset():
        call at the beginning of new experiment or epoch to reset all accumulators.
    update(targets, predictions):
        call per iteration to update the class accumulators for predictions and corresponding groundtruths.
    video_end():
        call at the end of every video during inference to log performance per video.

    RESULTS
    ----------
    compute_AP():
        call at any point to check the performance of all seen examples after the last reset() call.
    compute_video_AP():
        call at any time, usually at the end of experiment or inference, to obtain the performance of all tested videos.
    compute_global_AP():
        call at any point, compute the framewise AP for all frames across all videos and mAP
    compute_per_video_mAP(self):
        show mAP per video (not very useful)
    topk(k):
        obtain top k=[5,10,20, etc] performance
    topClass(k):
        obtain top-k correctly detected classes
    """

    def __init__(self, num_class=100, ignore_null=False):
        super(Recognition, self).__init__()
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.ignore_null = ignore_null
        self.reset_global()

        ##%%%%%%%%%%%%%%%%%%% RESET OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def reset(self):
        "call at the beginning of new experiment or epoch to reset the accumulators for preditions and groundtruths."
        self.predictions = np.empty(shape=[0, self.num_class], dtype=np.float)  # 存储的是一个测试文件夹的预测结果, 最终形成[该视频帧数, 类别数]的数组
        self.targets = np.empty(shape=[0, self.num_class], dtype=np.int)  # 存储的是一个测试文件夹的GT

    def reset_global(self):
        "call at the beginning of new experiment"
        self.global_predictions = []
        self.global_targets = []
        self.reset()  # 要调用reset方法

        ##%%%%%%%%%%%%%%%%%%% UPDATE OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def update(self, targets, predictions):
        """
        update prediction function
        @args
        -----
        targets: 2D array, float
            groundtruth of shape (F, C) where F = number of frames, C = number of class
        predictions: 2D array, int
            model prediction of the shape as the groundtruth
        """
        self.predictions = np.append(self.predictions, predictions, axis=0)  # 每一行是一个预测向量，共有class列
        self.targets = np.append(self.targets, targets, axis=0)

    def video_end(self):
        "call to signal the end of current video. Needed during inference to log performance per video"
        self.global_predictions.append(self.predictions)
        self.global_targets.append(self.targets)
        self.reset()  # 注意有个reset

    ##%%%%%%%%%%%%%%%%%%% COMPUTE OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def compute_AP(self, component="ivt", ignore_null=False):
        """
        compute performance for all seen examples after a reset()
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class
        mean: float
            mean AP performance
        """
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets = self.extract(self.targets, component)
            predicts = self.extract(self.predictions, component)
        else:
            sys.exit("Function filtering {} not yet supported!".format(component))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore')  # , message='[info] triplet classes not represented in this test sample will be reported as nan values.')
            classwise = average_precision_score(targets, predicts, average=None)
            if (ignore_null and component == "ivt"): classwise = classwise[:-6]
            mean = np.nanmean(classwise)
        return {"AP": classwise, "mAP": mean}

    def compute_global_AP(self, component="ivt", ignore_null=False):
        """
        compute performance for all seen examples after a reset_global()
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class
        mean: float
            mean AP performance
        """
        global_targets = self.global_targets
        global_predictions = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not yet supported!".format(component))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore')  # , message='[info] triplet classes not represented in this test sample will be reported as nan values.')
            classwise = average_precision_score(targets, predicts, average=None)
            if (ignore_null and component == "ivt"): classwise = classwise[:-6]
            mean = np.nanmean(classwise)
        return {"AP": classwise, "mAP": mean}

    def compute_video_AP(self, component="ivt", ignore_null=False):
        # 注意看if test里, set_chlg_eval永远为False, 因此component不可能传入的是'i', 'v', 't'
        # 虽然名叫video_AP, 在验证时就是整个验证集的AP, 在测试时也是整个测试集的AP
        """
        compute performance video-wise AP
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class for all videos
        mean: float
            mean AP performance for all videos
        """
        global_targets = self.global_targets
        global_predictions = self.global_predictions
        if len(self.targets) > 0:  # 这个好像不起作用吧, len(self.targets)是0
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        video_log = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for targets, predicts in zip(global_targets, global_predictions):  # 打包成多个元组, 每个元组是一个视频的GT和PR
                if component in ["ivt", "it", "iv", "t", "v", "i"]:
                    # 如果component="ivt", extract其实不起作用, 注意由于不可能传入"i","v","t", 因此将其看作传入的是"ivt"
                    targets = self.extract(targets, component)  # 在计算mAP、mAPi、mAPv、mAPt时，extract不起作用
                    predicts = self.extract(predicts, component)
                else:
                    sys.exit("Function filtering {} not yet supported!".format(component))
                # NOTE：下面average_precision_score的average=None, 这样传回来的才是一个1D数组, 否则会直接将各个类别的AP平均返回1个数
                classwise = average_precision_score(targets, predicts, average=None)  # 对一个视频计算AP, 以I举例, 返回的是I各个类别的AP, classwise是维度为(I_class, )的1D数组
                video_log.append(classwise.reshape([1, -1]))  # reshape成[1, I_class]然后append到video_log, video_log在整个for循环结束后会形成一个长度为"测试视频个数"的列表, 列表中每个元素是一个[1, I_class]的数组
            video_log = np.concatenate(video_log, axis=0)  # concatenate后video_log变成一个维度为[测试视频个数, I_class]的数组
            videowise = np.nanmean(video_log, axis=0)  # 计算每一列的平均值, videowise是一个(I_class, )的1D数组
            if (ignore_null and component == "ivt"): videowise = videowise[:-6]  # if不成立
            mean = np.nanmean(videowise)  # 在videowise上算平均, 得到1个数
        return {"AP": videowise, "mAP": mean}  # AP是在整个测试集上I的各个类别的的AP值, 而mAP是I的AP值(即将I各个类别的AP值加起来平均)

    ##%%%%%%%%%%%%%%%%%%% TOP OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def topK(self, k=5, component="ivt"):
        """
        compute topK performance for all seen examples after a reset_global()
        @args
        ----
        k: int
            number of chances of correct prediction
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target.
        @return
        ----
        mean: float
            mean top-k performance
        """
        global_targets = self.global_targets
        global_predictions = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not supported yet!".format(component))
        correct = 0.0
        total = 0
        for gt, pd in zip(targets, predicts):
            gt_pos = np.nonzero(gt)[0]
            pd_idx = (-pd).argsort()[:k]
            correct += len(set(gt_pos).intersection(set(pd_idx)))
            total += len(gt_pos)
        if total == 0: total = 1
        return correct / total

    def topClass(self, k=10, component="ivt"):
        """
        compute top K recognize classes for all seen examples after a reset_global()
        @args
        ----
        k: int
            number of chances of correct prediction
        @return
        ----
        mean: float
            mean top-k recognized classes
        """
        global_targets = self.global_targets
        global_predictions = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not supported yet!".format(component))
        classwise = average_precision_score(targets, predicts, average=None)
        pd_idx = (-classwise).argsort()[:k]
        output = {x: classwise[x] for x in pd_idx}
        return output
