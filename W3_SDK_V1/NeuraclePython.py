from  ctypes import *
import ctypes
import os
from enum import IntEnum



class  NeuracleResultCode(IntEnum):
    NeuracleResult_Succeed = 0,
    NeuracleResult_Information_SyncSucceeded		= 0x01000001,
    NeuracleResult_Information_TriggerArrived		= 0x01000002,
    NeuracleResult_Information_ReconnectSucceeded	= 0x01000003
    NeuracleResult_Information_ValueNoChanging		= 0x01000004

    NeuracleResult_Warning_NeedMoreSpace		= 0x04000001
    NeuracleResult_Warning_NoEnoughData			= 0x04000002
    NeuracleResult_Warning_DoggleLost			= 0x04000003
    NeuracleResult_Warning_DeviceDisconnected	= 0x04000004

    NeuracleResult_Error_InvalidHandle			= 0x08000001
    NeuracleResult_Error_InvalidParameter		= 0x08000002
    NeuracleResult_Error_DoggleVerifyFailed		= 0x08000003
    NeuracleResult_Error_StartAmplifyFailed		= 0x08000004
    NeuracleResult_Error_StartTriggerFailed		= 0x08000005

    NeuracleResult_System_Error_NoEnoughMemory	= 0x0C000001


# 定义回调函数参数的结构体
class DeviceInformation(Structure):
    _fields_ = [("DeviceType", c_byte *20),
                ("SerialNumber", c_byte *20),
                ("IPAddress", c_byte *20),
                ("IsTrigger", c_bool)]

class  NeuracleDataBlock(Structure):
    _fields_ = [("Datas", POINTER(c_float)),
                ("Markers", POINTER(c_int)),
                ("Timestamp", c_int),
                ("ChannelCount", c_int),
                ("DataCountPerChannel", c_int)]

class  NeuracleDeviceStatus(Structure):
    _fields_ = [("BatteryLevel", c_int),
                ("SyncStatus", c_bool),
               ("DeviceConnectStatus", c_bool)
    ]

# import pdb; pdb.set_trace()
NeuracleSDKDll = windll.LoadLibrary('E:/PycharmProjects/flow/Meta_BCI-master/W3_SDK_V1/NeusenW3SDKDll.dll')


def NeuracleControllerInitialize(controller:c_void_p, deviceAddress:str, channelCount:int, sampleRate:int, gain:int, delayMilliseconds:int, autoReconnect:bool):
    func = NeuracleSDKDll.NeuracleControllerInitialize
    func.argtypes = [POINTER(c_void_p),c_char_p,c_int,c_int,c_int,c_int,c_bool]
    func.restype = c_int
    nRet = func(controller, create_string_buffer(bytes(deviceAddress,encoding='utf-8')), c_int(channelCount), c_int(sampleRate),c_int( gain),c_int( delayMilliseconds),c_bool(autoReconnect))
    return nRet

def NeuracleControllerInitializeWithTriggerBox(controller:c_void_p, deviceAddress:str, TriggerBoxAddress:str, channelCount:int, sampleRate:int, gain:int, delayMilliseconds:int, autoReconnect:bool):
    func = NeuracleSDKDll.NeuracleControllerInitializeWithTriggerBox
    func.argtypes = [POINTER(c_void_p),c_char_p,c_char_p,c_int,c_int,c_int,c_int,c_bool]
    func.restype = NeuracleResultCode
    nRet = func(controller, create_string_buffer(bytes(deviceAddress,encoding='utf-8')), create_string_buffer(bytes(TriggerBoxAddress,encoding='utf-8')), c_int(channelCount), c_int(sampleRate),c_int( gain),c_int( delayMilliseconds),c_bool(autoReconnect))
    return nRet

def NeuracleControllerStart(controller:c_void_p):
    func = NeuracleSDKDll.NeuracleControllerStart
    func.argtypes = [c_void_p]
    func.restype = NeuracleResultCode
    ret = func(controller)
    return ret

def NeuracleControllerStartImpedance(controller:c_void_p):
    func = NeuracleSDKDll.NeuracleControllerStartImpedance
    func.argtypes = [c_void_p]
    func.restype = NeuracleResultCode
    ret = func(controller)
    return ret

#__declspec(dllexport)  NeuracleResultCode NeuracleControllerReadData(NeuracleController* controller, DataBlock** dataBlock);
def  NeuracleControllerReadData(controller:c_void_p, dataBlock:POINTER(POINTER(NeuracleDataBlock))):
    func = NeuracleSDKDll.NeuracleControllerReadData
    func.argtypes = [c_void_p, POINTER(POINTER(NeuracleDataBlock))]
    func.restype = NeuracleResultCode
    ret = func(controller, pointer(dataBlock))
    return ret

#__declspec(dllexport)  NeuracleResultCode NeuracleControllerGetStatus(NeuracleController* controller, NeuracleDeviceStatus *deviceStatus);
def  NeuracleControllerGetStatus(controller:c_void_p, deviceStatus:POINTER(NeuracleDeviceStatus)):
    func = NeuracleSDKDll.NeuracleControllerGetStatus
    func.argtypes = [c_void_p, POINTER(NeuracleDeviceStatus)]
    func.restype = NeuracleResultCode
    ret = func(controller, pointer(deviceStatus))
    return ret

#__declspec(dllexport)  NeuracleResultCode NeuracleControllerReadImpedance(NeuracleController* controller, float* impedance, int* count);
def  NeuracleControllerReadImpedance(controller:c_void_p, impedances:byref, count:c_void_p):
    func = NeuracleSDKDll.NeuracleControllerReadImpedance
    #func.argtypes = [c_void_p, POINTER(c_float)]
    func.restype = NeuracleResultCode
    ret = func(controller, ctypes.byref(impedances),count)
    return ret

#__declspec(dllexport)  void NeuracleFreeDataBlock(DataBlock* dataBlock);
def NeuracleFreeDataBlock(dataBlock):
    func = NeuracleSDKDll.NeuracleFreeDataBlock
    func.argtypes = [POINTER(NeuracleDataBlock)]
    func(dataBlock)
    return

def NeuracleControllerStop(controller):
    func = NeuracleSDKDll.NeuracleControllerStop
    func.argtypes = [c_void_p]
    func(controller)

def NeuracleControllerFinialize(controller):
    func = NeuracleSDKDll.NeuracleControllerFinialize
    func.argtypes = [c_void_p]
    func(controller)
    return


#########################


def NeuracleDeviceDiscovery(controller:c_void_p):
    func = NeuracleSDKDll.NeuracleDeviceDiscovery
    func.argtypes = [POINTER(c_void_p)]
    func.restype = NeuracleResultCode
    ret = func(controller)
    return ret


def NeuracleStartDeviceDiscovery(controller:c_void_p):
    func = NeuracleSDKDll.NeuracleStartDeviceDiscovery
    func.argtypes = [c_void_p]
    func.restype = NeuracleResultCode
    ret = func(controller)
    return ret

def NeuracleGetFoundedDevices(controller:c_void_p, devices:byref , count:c_void_p):
    func = NeuracleSDKDll.NeuracleGetFoundedDevices
    #func.argtypes = [c_void_p, type((DeviceInformation*(count[0]))()), c_void_p]
    func.restype = NeuracleResultCode
    ret = func(controller, ctypes.byref(devices), (count))
    return ret

def NeuracleStopDeviceDiscovery(controller:c_void_p):
    func = NeuracleSDKDll.NeuracleStopDeviceDiscovery
    func.argtypes = [c_void_p]
    ret = func(controller)
    return None

if __name__ == '__main__':
    print(base_path)