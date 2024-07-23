#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/21 9:44
@Author ： FANG Junying
@Email ： fangjunying@neuracle.cn
@File ：Step1_DetectImp.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
#Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/
"""
import os
from queue import Queue
import time
from W3_SDK_V1.NeuraclePython import NeuracleDeviceDiscovery, c_void_p
from W3_SDK_V1.NeuraclePython import *
import numpy as np

ChannelCount = 32
SampleRate = 1000
DelayMilliSeconds = 100  ## 最大1000
Gain = 12

deviceAddress = "192.168.31.54"
triggerBoxAddress = "192.168.31.7"


def printByteArray(title, bArray, length):
    tt = create_string_buffer(length + 1)
    memmove(tt, byref(bArray), length)
    temp = tt.value.__str__()[0:-1]
    print(title + temp)

def TestForImpedance(inputDeviceAddress):
    os.chdir("E:/PycharmProjects/flow/Meta_BCI-master/W3_SDK_V1")
    print("")
    print("==============Test For Impedance=============")
    print("NeuracleControllerInitialize")
    deviceAddress = inputDeviceAddress
    controller = c_void_p(0)
    ret = NeuracleControllerInitialize(controller, deviceAddress, ChannelCount, SampleRate, Gain, DelayMilliSeconds,
                                       True)
    print(controller)

    print("NeuracleControllerStartImpedance")
    ret = NeuracleControllerStartImpedance(controller)
    print(str(ret))
    if ret != NeuracleResultCode.NeuracleResult_Succeed:
        print("NeuracleControllerStartImpedance Failed with code:" + str(ret))

        print("NeuracleControllerStop")
        NeuracleControllerStop(controller)

        print("NeuracleControllerFinialize")
        NeuracleControllerFinialize(controller)

        return "NeuracleResultCode.NeuracleResult_Error_StartAmplifyFailed"

    bufferCount = 1
    bufferSize = 1
    impedances = (c_float * ChannelCount)()
    pBufferCount = pointer(c_int(bufferCount))

    for iii in range(int(3 * 60)):
        time.sleep(1)
        pBufferCount[0] = bufferSize
        ret = NeuracleControllerReadImpedance(controller, impedances, pBufferCount)
        if ret == NeuracleResultCode.NeuracleResult_Warning_NeedMoreSpace:
            print(str(ret))
            bufferSize = pBufferCount[0] + 10
            pBufferCount[0] = bufferSize
            impedances = (c_float * bufferSize)()
            continue

        if ret != NeuracleResultCode.NeuracleResult_Succeed:
            print("NeuracleControllerReadImpedance Failed with:" + str(ret))
            continue
        print("Impedances:")
        for jjj in range(ChannelCount):
            print("Channel " + str(jjj) + ":" + str(impedances[jjj]))
    try:
        print("NeuracleControllerStop")
        NeuracleControllerStop(controller)

        print("NeuracleControllerFinialize")
        NeuracleControllerFinialize(controller)
        return 'ok'
    except:
        return 'failedToStop'

def TestForUDP():
    print("NeuracleDeviceDiscovery")
    controller = c_void_p(0)
    NeuracleDeviceDiscovery(controller)

    bufferSize = 1
    bufferCount = 1
    pBufferCount = pointer(c_int(bufferCount))
    pDevicesInformations = (DeviceInformation * bufferCount)()

    print("NeuracleStartDeviceDiscovery")
    NeuracleStartDeviceDiscovery(controller)
    for iii in range(0, 5):  ## origin range(0,20)
        time.sleep(0.5)
        pBufferCount[0] = bufferSize
        ret = NeuracleGetFoundedDevices(controller, pDevicesInformations, pBufferCount)
        if ret == NeuracleResultCode.NeuracleResult_Warning_NeedMoreSpace:
            bufferSize = pBufferCount[0] + 10
            pBufferCount[0] = bufferSize
            pDevicesInformations = (DeviceInformation * bufferSize)()
            continue

        if ret == NeuracleResultCode.NeuracleResult_Succeed and pBufferCount[0] > 0:
            print("=========New Device Founded========" + str(pBufferCount[0]))
            for i in range(0, pBufferCount[0]):
                obj = pDevicesInformations.__getitem__(i)
                print("----------------")
                print("Index:" + str(i))
                printByteArray("DeviceType  :", obj.DeviceType, 20)
                printByteArray("SerialNumber:", obj.SerialNumber, 20)
                printByteArray("IPAddress   :", obj.IPAddress, 20)
                print("IsTrigger   :" + str(obj.IsTrigger))

    print("NeuracleStopDeviceDiscovery")
    NeuracleStopDeviceDiscovery(controller)


if __name__ == '__main__':
    ### 搜索设备
    TestForUDP()
    print("Going to check impedance....")
    ipaddress = deviceAddress
    # ipaddress = '192.168.1.88'
    ### 阻抗检测
    for i in range(1000):
        ret = TestForImpedance(ipaddress)
        if str(ret) == 'NeuracleResultCode.NeuracleResult_Error_StartAmplifyFailed':
            time.sleep(1)

