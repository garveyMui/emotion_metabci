
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/21 15:22
@Author ： FANG Junying
@Email ： fangjunying@neuracle.cn
@File ：Step1_DetectImp.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
#Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/
"""


# Versions:
# 	v0.1: 2018-08-14, orignal
# 	v1.0: 2020-06-04，update demo
# Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

# from neuracle_lib.dataServer import DataServerThread
from W3_SDK_V1.NeuraclePython import *
import time,sys
import numpy as np
from os import path
import json

### 用户修改
ChannelCount = 32
SampleRate = 1000
# deviceAddress = "192.168.10.107"
deviceAddress = "192.168.31.54"
#triggerBoxAddress = "192.168.10.145"

### 用户最好不要修改
DelayMilliSeconds = 100  ## 最大1000
Gain = 12
channels = ["T6", "P4", "Pz","PG2","F8","F4","Fp1","Cz","PG1","F7","F3","C3","T3","A1","Oz","O1","O2","Fz","C4","T4","Fp2","A2","T5","P3",
     "EKG","X1","X2","X3","X4","X5","X6",'X7']



def printByteArray(title, bArray, length):
    tt = create_string_buffer(length + 1)
    memmove(tt, byref(bArray), length)
    temp = tt.value.__str__()[0:-1]
    print(title + temp)

def TestForNeuracleTCPConnect(hasTriggerBox: bool):
    print("")
    print("==============Test For TCP=============")
    print("NeuracleControllerInitialize")
    controller = c_void_p(0)
    if hasTriggerBox:
        ret = NeuracleControllerInitializeWithTriggerBox(controller, deviceAddress, triggerBoxAddress, ChannelCount,
                                                         SampleRate,
                                                         Gain, DelayMilliSeconds, True)
    else:
        ret = NeuracleControllerInitialize(controller, deviceAddress, ChannelCount, SampleRate, Gain, DelayMilliSeconds,
                                           True)

    print(ret)
    print(controller)
    bufferSize = 1
    pDataBlock = pointer(NeuracleDataBlock())

    print("NeuracleControllerStart")
    ret = NeuracleControllerStart(controller)
    if ret != NeuracleResultCode.NeuracleResult_Succeed:
        print("NeuracleControllerStart Failed with code:" + str(ret))

        print("NeuracleControllerStop")
        NeuracleControllerStop(controller)

        print("NeuracleControllerFinialize")
        NeuracleControllerFinialize(controller)

        return "NeuracleResultCode.NeuracleResult_Error_StartAmplifyFailed",[]

    dataLenthPerChannel = int(DelayMilliSeconds * SampleRate / 1000)
    pointCountBeforeTrigger = 50
    pointCountAfterTrigger = 200 - 1
    buffer = [0.0] * dataLenthPerChannel
    dataIndex = 0
    triggerFounded = False
    segData = []
    npoints = 0
    # timestampList = []
    for iii in range(1000):
        time.sleep(0.1)
        ret = NeuracleControllerReadData(controller, pDataBlock)
        if iii % 500 == 99:
            deviceStatus = NeuracleDeviceStatus()
            NeuracleControllerGetStatus(controller, deviceStatus)
            print("BatteryLevel:" + str(deviceStatus.BatteryLevel) + " SyncStatus:" + str(
                deviceStatus.SyncStatus) + " DeviceConnectStatus:" + str(deviceStatus.DeviceConnectStatus))
        # import pdb; pdb.set_trace()
        if ret == 0:  # NeuracleResultCode.NeuracleResult_Succeed:
            # print(str(ret))
            data_block = [0.0] * int(dataLenthPerChannel * (ChannelCount + 1))
            # print('timstemp = {0}'.format(pDataBlock[0].Timestamp))
            # timestampList.append(pDataBlock[0].Timestamp)
            for kkk in range(0, int(dataLenthPerChannel * ChannelCount)):
                data_block[kkk] = pDataBlock[0].Datas[kkk]

            if (pDataBlock[0].Markers):
                for jjj in range(0, dataLenthPerChannel):
                    if (pDataBlock[0].Markers[jjj] > 0):
                        print('trigger={0}'.format(pDataBlock[0].Markers[jjj]))
                    data_block[jjj + dataLenthPerChannel * ChannelCount] = pDataBlock[0].Markers[jjj]
            x = np.reshape(data_block, (ChannelCount+1, dataLenthPerChannel))
            segData.append(x)
            NeuracleFreeDataBlock(pDataBlock)
            npoints = npoints + dataLenthPerChannel

    print("NeuracleControllerStop")
    NeuracleControllerStop(controller)
    print("NeuracleControllerFinialize")
    NeuracleControllerFinialize(controller)
    return 'ok',segData,

if __name__ == '__main__':
    for i in range(10):
        ret,segData =TestForNeuracleTCPConnect(False)
        if str(ret) == 'NeuracleResultCode.NeuracleResult_Error_StartAmplifyFailed':
            time.sleep(1)
        elif str(ret) == 'NeuracleResultCode.NeuracleResult_Error_StartTriggerFailed':
            time.sleep(1)
        else:
            break
    if segData:
        x = np.hstack(segData)
        print(x.shape)
        index = np.argwhere(x[-1,:] >0)
        print(index)
        # print(x[-1,index])
