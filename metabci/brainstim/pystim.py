import random

import numpy as np
import datetime
from .utils import NeuroScanPort, NeuraclePort, TxtPort, _check_array_like
from psychopy import visual, event
from psychopy.visual.movie3 import MovieStim3
class item:
    def __init__(self):
        self.st=-1
        self.en=-1
        self.label = 0
        self.time=0


    def start(self):
        return

    def end(self):
        return


class scene:
    def __init__(self):
        self.time = 0
        self.modules = []
        self.names=[]

    def __getitem__(self, item):
        for key,value in zip(self.names,self.modules):
            if(key==item):
                return value
        else:
            raise RuntimeError('\''+item+'\''+' does not exist.')

    def add_module(self,name, mod):
        if name in self.names:
            raise RuntimeError('\''+name+'\''+' already exists.')
        else:
            self.names.append(name)
        if mod.st < 0:
            if(not self.modules):
                mod.st=0
            else:
                mod.st=self.modules[-1].en
        if mod.en < 0:
            mod.en=mod.st + mod.time
        if mod.en > self.time:
            self.time = mod.en
        self.modules.append(mod)
    def run(self, win=None,
            bg_color=np.array([-1, -1, -1]),
            port_addr=9045,
            device_type="Txt", ):
        if not _check_array_like(bg_color, 3):
            raise ValueError("bg_color should be 3 elements array-like object.")

        win.color = bg_color

        if device_type == "NeuroScan":
            port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
        elif device_type == "Neuracle":
            port = NeuraclePort(port_addr) if port_addr else None
        elif device_type == "Txt":
            port = TxtPort(port_addr) if port_addr else None
        else:
            raise KeyError(
                "Unknown device type: {}, please check your input".format(device_type)
            )

        # start routine
        # episode 1: display speller interface
        flags=[True for i in self.names]
        start = datetime.datetime.now()
        while (datetime.datetime.now() - start).total_seconds() < self.time:
            if event.getKeys(keyList=['escape']):
                break
            now=(datetime.datetime.now() - start).total_seconds()
            win.flip()
            for item,flag in zip(self.modules,range(len(flags))):
                if now > item.st and now < item.en:
                    if flags[flag]==True:
                        flags[flag]=False
                        item.start()
                        if port and item.label != -1:
                            port.setData(item.label)
                    item.draw()
                if flags[flag]==False and now > item.en:
                    flags[flag]=True
                    item.end()
        win.flip()



class Text(visual.TextStim, item):
    def __init__(self, win,
                 text="Text",
                 font="Times New Roman",
                 pos=(0.0, 0.0),
                 color=(1, -1, -1),
                 height=100,
                 opacity=1.0,
                 contrast=1.0,
                 ori=0.0,
                 antialias=True,
                 bold=False,
                 italic=False,
                 time=5,
                 label=0,
                 start=-1,
                 end=-1):
        super().__init__(win=win,
                         text=text,
                         font=font,
                         pos=pos,
                         color=color,
                         units="pix",
                         height=height,
                         opacity=opacity,
                         contrast=contrast,
                         ori=ori,
                         antialias=antialias,
                         bold=bold,
                         italic=italic)
        self.st = start
        self.en = end
        self.label = label
        self.time = time

    def start(self):
        return

    def end(self):
        return

class CountDown(visual.TextStim, item):
    def __init__(self, win, font="Times New Roman", pos=(0.0, 0.0), color=(1, -1, -1), size=100, time=5,
                 label=0,start=-1,end=-1):
        super().__init__(text=str(time),
                         win=win,
                         font=font,
                         pos=pos,
                         color=color,
                         units="pix",
                         height=size,
                         bold=True, )
        self.st = start
        self.en = end
        self.label = label
        self.time = time
        self.timestamp=0

    def start(self):
        self.timestamp=datetime.datetime.now()
        return
    def draw(self,win=None):
        self.text=str(1+int(self.time-(datetime.datetime.now()-self.timestamp).total_seconds()))
        super().draw(win)
    def end(self):
        return

class Movie(MovieStim3, item):
    def __init__(self, win, file, pos=[0.0, 0.0], size=[1024, 768],start=-1,end=-1, time=5, label=0):
        super().__init__(
            win=win,
            units="pix",
            filename=file,
            size=size,
            pos=np.array(pos),
            ori=0.0,
            opacity=1.0,
        )
        self.st = start
        self.en = end
        self.label = label
        self.time = time

    def start(self):
        self.play()

    def end(self):
        self.pause()
        self.seek(0.0)


class Image(visual.ImageStim,item):
    def __init__(self,
                 win,
                 file,
                 units='pix',
                 mask=None,
                 texRes=2,
                 pos=[0.0, 0.0],
                 size=[1024, 768],
                 color=(1.,1.,1.),
                 opacity=1,
                 contrast=-1,
                 start=-1,
                 end=-1,
                 time=5,
                 label=0):
        super().__init__(win=win,
                         image=file,
                         mask=mask,
                         units=units,
                         pos=pos,
                         size=size,
                         contrast=contrast,
                         opacity=opacity,
                         texRes=texRes,
                         color=color,
                         )
        self.st = start
        self.en = end
        self.label = label
        self.time = time

    def start(self):
        return
    def end(self):
        return

class StimArray(item):
    def __init__(self,list=[],
                 start=-1,
                 end=-1,
                 time=-1,
                 method='random'):
        super().__init__()
        self.st=start
        self.en=end
        self.time=time
        self.label=-1
        self.stim_list=list
        self.method="random"
        if self.time<0:
            for item in self.stim_list:
                self.time=self.time if self.time>item.time else item.time
        if method=="sequential":
            self.method='sequential'
        elif method=="random":
            self.method=="random"
            random.seed(datetime.datetime.now().microsecond)
        else:
            raise RuntimeError('\''+method+'\'+' "method is illegal")
        self.iter=0


    def reset(self,n=-1):
        if n==-1:
            if self.method=="sequential":
                self.iter=0
            else:
                random.seed(datetime.datetime.now().microsecond)
        else:
            if self.method == "sequential":
                self.iter = n
            else:
                random.seed(n)

    def start(self):
        if len(self.stim_list)==0:
            raise ValueError("Stim List is empty.")
        if self.method=="random":
            self.iter=random.randint(0,len(self.stim_list)-1)
        self.label=self.stim_list[self.iter].label

        self.stim_list[self.iter].start()

    def draw(self):
        self.stim_list[self.iter].draw()

    def end(self):
        self.stim_list[self.iter].end()
        self.label=-1
        if self.method=="sequential":
            self.iter=(self.iter+1) % len(self.stim_list)

class StimFlash(item):
    def __init__(self,
                 stim,
                 color=[(-1.,-1.,-1.),(-0.2,-0.2,-0.2)],
                 freq=1.0,
                 start=-1,
                 end=-1,
                 time=5,
                 label=0):
        self.stim=stim
        self.color=color
        self.time_interval=0.5/freq
        self.st = start
        self.en = end
        self.label = label
        self.time = time
        self.timestamp = 0

    def start(self):
        self.timestamp = datetime.datetime.now()
        return
    def draw(self):
        self.stim.setColor(self.color[int((datetime.datetime.now()-self.timestamp).total_seconds()/self.time_interval)%2])
        self.stim.draw()
        return
    def end(self):
        return





