# -*- coding: utf-8 -*-
# License: MIT License
"""
SSAVEP Feedback on NeuroScan.

"""
import socket
import sys
import time
import numpy as np
import torch

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker, Neuracle
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.csp import FBCSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from scipy import signal
from metabci.brainda.algorithms.deep_learning.models import model_initialize
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

# 预测标签
def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    # X = resample(X, up=256, down=srate)
    X = resample(X, down=5)
    # 滤波
    X = bandpass(X, 8, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    logit_prob = model(X)
    logit_prob = torch.squeeze(logit_prob)
    logit_prob = logit_prob.detach().cpu().numpy()
    print(logit_prob)
    return logit_prob

class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 pick_chs,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name):
        super().__init__(timeout=timeout, name=worker_name)

        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.labels = None

    def pre(self):
        config = {"encoder":"eegnet",
                  "n_channels":30,
                  "n_samples":200*4,
                  "n_classes":3}
        self.estimator = model_initialize(**config)
        self.ch_ind = np.arange(len(self.pick_chs))
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')
        sys.stdout.flush()

    def consume(self, data, beta=0.6):
        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        logit_prob = model_predict(data, srate=self.srate, model=self.estimator)
        if self.labels is None:
            self.labels = logit_prob
        else:
            self.labels = logit_prob*beta + self.labels*(1-beta)
        p_labels = np.argmax(logit_prob, axis=-1)
        p_labels = int(p_labels)
        p_labels = p_labels + 1
        p_labels = [p_labels]
        # p_labels = p_labels.tolist()
        print(p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

    def post(self):
        emotion_dict = {0: "sad", 1: "neutral", 2: "happy"}
        emotion_key = np.argmax(self.labels)
        emotion = emotion_dict[emotion_key]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('127.0.0.1', 4023)
        messages = emotion
        try:
            print(f'Sending "{messages}" to {server_address}')
            sent = sock.sendto(messages.encode(), server_address)
        finally:
            print('Closing socket')
            sock.close()


def post_test():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.bind((local_ip, local_port))
    server_address = ('127.0.0.1', 4023)
    messages = "我今天被领导骂了一顿，好想哭。"
    try:
        print(f'Sending "{messages}" to {server_address}')
        sent = sock.sendto(messages.encode(), server_address)

    finally:
        print('Closing socket')
        sock.close()

if __name__ == '__main__':
    # post_test()

    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_interval = [0, 4]
    # 事件标签
    stim_labels = None
    pick_chs = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',
                'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2',
                'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
                'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
                'O1', 'OZ', 'O2', 'CB2']

    indexs_32 = [0, 2, 9, 7, 11, 5,
                 13, 17, 19, 15, 21, 27,
                 25, 29, 23, 31, 35, 37,
                 33, 39, 45, 43, 47, 41,
                 49, 52, 54, 59, 58, 60]
    pick_chs = np.array(pick_chs)[indexs_32].tolist()

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(pick_chs=pick_chs,
                            stim_interval=stim_interval,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            timeout=0.1,
                            worker_name=feedback_worker_name)  # 在线处理
    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)        # 打标签全为1
    # worker.pre()

    nc = Neuracle(
        device_address=('192.168.31.46', 8712),
        srate=srate,
        num_chans=33)  # Neuracle parameter

    # 与nc建立tcp连接
    nc.connect_tcp()

    # register worker来实现在线处理
    nc.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    nc.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)

    # nc开始截取数据线程，并把数据传递数据给处理进程
    nc.start_trans()

    # 任意键关闭处理进程
    input('press any key to close\n')
    # 关闭处理进程
    nc.down_worker('feedback_worker')
    # 等待 1s
    time.sleep(1)

    # nc停止在线截取线程
    nc.stop_trans()

    nc.close_connection()  # 与nc断开连接
    nc.clear()
    print('bye')
