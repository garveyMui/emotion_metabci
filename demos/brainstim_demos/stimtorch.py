import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (paradigm,VisualStim,emotion)
from metabci.brainstim import pystim as ps
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
class pro_emotion(VisualStim):
    def __init__(self,win):
        super().__init__(win=win)
        movie1=ps.Movie(self.win,"metabci/brainstim/textures/1-3.mkv",time=5,label=1)
        movie2=ps.Movie(self.win,"metabci/brainstim/textures/1-4.mkv",time=5,label=2)
        pic1=ps.Image(self.win,"metabci/brainstim/textures/left_hand.png",time=5,label=3)
        self.scene = ps.scene()
        self.scene.add_module("start",ps.Text(self.win,"start",time=1))
        self.scene.add_module("start count down",ps.CountDown(self.win,time=1,start=0,label=-1,pos=[0.0,200.0]))
        self.scene.add_module("test",ps.StimArray([movie1,movie2,pic1],method="sequential"))
        self.scene.add_module("rest",ps.Text(self.win,"rest", time=3))
    def forward(self,win):
        for i in range(3):
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
    emotion pro
    """
    pro_emotion = pro_emotion(win=win)
    #base_emotion = base_emotion(win=win, film="demos/brainstim_demos/emotion.xlsx")


                                 # 在线实验的标志
    ex.register_paradigm_new("pro emotion",pro_emotion)

    ex.run()
