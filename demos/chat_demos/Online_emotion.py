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
from einops import rearrange
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from scipy import signal

from metabci.brainda.algorithms.deep_learning.models import model_initialize, model_pretrained
from metabci.brainda.algorithms.deep_learning.utils import get_input_chans
from metabci.brainflow.amplifiers import Marker, Neuracle
from metabci.brainflow.workers import ProcessWorker


def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

# for eegnet
def model_predict(X, model=None):

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
    if not type(X) == torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32)
    logit_prob = model(X)
    logit_prob = torch.squeeze(logit_prob)
    logit_prob = logit_prob.detach().cpu().numpy()
    print(logit_prob)
    return logit_prob

# for labram
def labram_model_predict(X, pick_channels, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, down=5)
    # 滤波
    X = bandpass(X, 13, 75, 200)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    if not type(X) == torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32)
    X = rearrange(X, 'b n (a t) -> b n a t', t=200)
    logit_prob = model((X, pick_channels))
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
        ##### get eegnet model #####
        # config = {"encoder":"eegnet",
        #           "n_channels":32,
        #           "n_samples":200*4,
        #           "n_classes":3}
        # self.estimator = model_initialize(**config)
        config = {"encoder": "labram",
                  "n_channels": 32,
                  "n_samples": 200,
                  "n_classes": 3,
                  "pretrained_path": "E:/emotion_metabci/emotion_metabci/checkpoints/LaBraM/labram-base.pth",
                  "yaml_path": "E:/emotion_metabci/emotion_metabci/metabci/brainda/algorithms/deep_learning/encoders/LaBraM/config.yaml"}
        self.estimator = model_pretrained(**config)
        self.ch_idx = np.arange(len(self.pick_chs))
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
        data = data[self.ch_idx]  # drop trigger channel
        idx_chs = get_input_chans(self.pick_chs)  # add CLS token and remap channel names to 10-20 index
        ##### use eegnet model #####
        # logit_prob = model_predict(data, model=self.estimator)
        logit_prob = labram_model_predict(data, idx_chs, model=self.estimator)
        if self.labels is None:
            self.labels = logit_prob
        else:
            self.labels = logit_prob*beta + self.labels*(1-beta)
        p_labels = np.argmax(logit_prob, axis=-1)
        p_labels = int(p_labels)
        p_labels = p_labels
        p_labels = [p_labels]
        print(p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

    def post(self):
        emotion_dict = {0: "sad", 1: "neutral", 2: "happy"}
        emotion_key = np.argmax(self.labels)
        emotion = emotion_dict[emotion_key]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # server_address = ('127.0.0.1', 4023)
        server_address = ('192.168.31.10', 4023)
        messages = emotion
        try:
            while True:
                # if time.time() % 1 == 0:
                print(f'Sending "{messages}" to {server_address}')
                sock.sendto(messages.encode(), server_address)
                response, server_address = sock.recvfrom(1024)
                if response.decode() == "got it":
                    print("Send successfully")
                    break
        finally:
            print('Closing socket')
            sock.close()



def post():
    labels = np.random.random((1, 3))
    emotion_dict = {0: "sad", 1: "neutral", 2: "happy"}
    emotion_key = np.argmax(labels, axis=-1)[0]
    emotion_key = 0
    emotion = emotion_dict[emotion_key]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('192.168.31.10', 4023)
    messages = emotion
    try:
        while True:
            # if time.time() % 1 == 0:
            print(f'Sending "{messages}" to {server_address}')
            sock.sendto(messages.encode(), server_address)
            response, server_address = sock.recvfrom(1024)
            if response.decode() == "got it":
                print("Send successfully")
                break
    finally:
        print('Closing socket')
        sock.close()


if __name__ == '__main__':
    # post()

    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段，考虑进视觉刺激延迟140ms
    stim_interval = [0, 4]
    # 事件标签
    stim_labels = None
    pick_chs = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
                'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ',
                'P3', 'P4', 'P7', 'P8', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'OZ',
                'O1', 'O2']

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
        # device_address=('192.168.31.46', 8712),
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
