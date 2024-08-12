import math
import pandas as pd
from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (VisualStim)
from metabci.brainstim import pystim as ps
from metabci.brainstim.framework import Experiment
class FileStim(VisualStim):
    def __init__(self,win,file):
        super().__init__(self,win)
        self.scene=ps.scene()
        f = pd.read_excel(file)
        for index, row in f.iterrows():
            if(row['type']=='movie'):
                self.scene.add_module(row['name'],ps.Movie(win=self.win,
                                                           file=row['file'],
                                                           time=row['time'],
                                                           start=row['start'],
                                                           end=row['end'],
                                                           label=row['label'],
                                                           pos=[row['pos_x'],row['pos_y']]))
            elif(row['type']=='text'):
                self.scene.add_module(row['name'], ps.Text(win=self.win,
                                                            text=row['text'],
                                                            time=row['time'],
                                                            start=row['start'],
                                                            end=row['end'],
                                                            label=row['label'],
                                                            pos=[row['pos_x'], row['pos_y']]))
            elif (row['type'] == 'countdown'):
                self.scene.add_module(row['name'], ps.CountDown(win=self.win,
                                                            time=row['time'],
                                                            start=row['start'],
                                                            end=row['end'],
                                                            label=row['label'],
                                                            pos=[row['pos_x'], row['pos_y']]))
            elif (row['type'] == 'image'):
                self.scene.add_module(row['name'], ps.Image(win=self.win,
                                                            file=row['file'],
                                                            time=row['time'],
                                                            start=row['start'],
                                                            end=row['end'],
                                                            label=row['label'],
                                                            pos=[row['pos_x'], row['pos_y']]))
    def forward(self):
        self.scene.run(win,bg_color=np.array([-1, -1, -1]),device_type="Txt",port_addr="out.txt")



class pro_emotion(VisualStim):
    def __init__(self,win):
        super().__init__(win=win)
        movie1=ps.Movie(self.win,"metabci/brainstim/textures/1-3.mkv",time=5,label=1)
        movie2=ps.Movie(self.win,"metabci/brainstim/textures/1-4.mkv",time=5,label=2)
        self.scene = ps.scene()
        self.scene.add_module("start",ps.Text(self.win,"start",time=1))
        self.scene.add_module("start count down",ps.CountDown(self.win,time=1,start=0,label=-1,pos=[0.0,200.0]))
        self.scene.add_module("test",ps.StimArray([movie1,movie2],method="random"))
        self.scene.add_module("rest",ps.Text(self.win,"rest", time=3))
    def forward(self,win):
        for i in range(3):
            self.scene.run(win,bg_color=np.array([-1, -1, -1]),device_type="Txt",port_addr="out.txt")

class pro_MI(VisualStim):
    def __init__(self,win):
        super().__init__(win=win)
        left=ps.StimFlash(ps.Image(self.win,"metabci/brainstim/textures/left_hand.png",pos=(-480.,0.),size=(288,288)),freq=1.,label=1)
        right = ps.StimFlash(ps.Image(self.win, "metabci/brainstim/textures/right_hand.png",pos=(480.,0.),size=(288,288)), freq=1., label=2)
        self.scene = ps.scene()
        self.scene.add_module("start",ps.Text(self.win,"prepare",time=3))
        self.scene.add_module("start count down",ps.CountDown(self.win,time=3,start=0,label=-1,pos=[0.0,200.0]))
        self.scene.add_module('back_left',ps.Image(self.win,"metabci/brainstim/textures/left_hand.png",
                                                   color=(-0.2,-0.2,-0.2),
                                                   pos=(-480.,0.),
                                                   size=(288,288),
                                                   label=-1,
                                                   time=10))
        self.scene.add_module('back_right', ps.Image(self.win, "metabci/brainstim/textures/right_hand.png",
                                                    color=(-0.2, -0.2, -0.2),
                                                    pos=(480., 0.),
                                                    size=(288, 288),
                                                    label=-1,
                                                    start=3,
                                                    time=10))
        self.scene.add_module("test",ps.StimArray([left,right],method="random",start=5,time=4))
        self.scene.add_module("imagine", ps.Text(self.win, "imagine", time=4))
        self.scene.add_module("rest",ps.Text(self.win,"rest", time=5))
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
    pro_MI=pro_MI(win=win)
    #base_emotion = base_emotion(win=win, film="demos/brainstim_demos/emotion.xlsx")


                                 # 在线实验的标志
    ex.register_paradigm_new("pro emotion",pro_emotion)
    ex.register_paradigm_new("pro_mi",pro_MI)

    ex.run()
