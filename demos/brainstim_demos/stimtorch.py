import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (paradigm,scene,Text,Movie,VisualStim,emotion,CountDown)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
class pro_emotion(VisualStim):
    def __init__(self,win):
        super().__init__(win=win)
        self.scene = scene()
        self.scene.add_module(Text(self.win,"start",time=5))
        self.scene.add_module(CountDown(self.win,time=5,start=0,label=-1,pos=[0.0,200.0]))
        self.scene.add_module(Movie(self.win,"C:\\Users\\405\\PycharmProjects\\emotion_metabci\\metabci\\brainstim\\textures\\1-1.mkv",time=5,label=1))
        self.scene.add_module(Text(self.win,"rest", time=5))
    def forward(self,win):
        self.scene.run(win,bg_color=np.array([-1, -1, -1]),device_type="Txt",port_addr="out.txt")
if __name__ == "__main__":
    mon = monitors.Monitor(
        name="secondary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1600, 900])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([-1, -1, -1])
    win_size = np.array([1600, 900])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()
    """
        emotion
        """
    basic_emotion = emotion(win)
    basic_emotion.config_movie()
    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 5  # 提示时长，转移视线
    rest_time = 30  # 提示后的休息时长
    image_time = 120  # 想象时长
    response_time = 2  # 在线反馈
    # port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    nrep = 5  # block数目
    lsl_source_id = "meta_online_worker"  # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic emotion",
        paradigm,
        VSObject=basic_emotion,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        # port_addr=port_addr,
        port_addr=None,
        nrep=nrep,
        image_time=image_time,
        pdim="emotion",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    emotion pro
    """
    pro_emotion = pro_emotion(win=win)
    #base_emotion = base_emotion(win=win, film="demos/brainstim_demos/emotion.xlsx")


                                 # 在线实验的标志
    ex.register_paradigm_new("pro emotion",pro_emotion)

    ex.run()
