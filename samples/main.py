import os
import os.path as osp
import sys

cur_d = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(cur_d, '..'))

os.environ['GREEN_BACKEND'] = 'gevent'
#import matplotlib.pyplot as plt
import gevent
from greenthread.monkey import monkey_patch; monkey_patch()

from mqsrv.server import make_server, run_server
from mqsrv.client import make_client
from greenthread.green import green_spawn, GreenQueue as Queue

from gevent.threading import Thread

from loguru import logger
import numpy as np
import numpy.linalg as npl

np.set_printoptions(suppress=True)
import cv2
import math as M
import shutil
import time
import json
import click

from glob import glob
from tools.utils import rigid_transform_3D
from tools.control_xianfa import RobotControl

BASE_LEN = 1.4
SCALE_BASE = 0.3

class MainControl:
    def __init__(self):
        self.simu = RobotControl(0.05, "configs/map.toml")
        self.simu.daemon = True
        self.simu.start()
       
        T_c_car = np.array([
           [0, 0, 1, 0],
           [0, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 1]
        ])

        self.w = self.simu.pp.w
        self.l = self.simu.pp.l

        task  = {
                "movement": {
                    "points": [
                        None,
                        [0, -BASE_LEN],
                        [self.l + BASE_LEN, -BASE_LEN],
                        [self.l + BASE_LEN, self.w+BASE_LEN],
                        [- BASE_LEN, self.w+BASE_LEN],
                        [-BASE_LEN, 0],
                    ],
                    "vel":100,
                    "end":[-BASE_LEN, 0],
                }
            }
	
#self.verify_points(tasks)
#for task in tasks:
        self.simu.update_task(task)
        self.simu.resume_task()
        self.simu.start = True

    def verify_points(self, tasks):
        xs = []
        ys = []
        cs = [i for i in range(8)]
        for i, task in enumerate(tasks):
            xs.append(task['movement']['end'][0])
            ys.append(task['movement']['end'][1])
            cs.append([(30 * i)/255, (255-i*30)/255,0])
        plt.scatter(xs, ys)
        for i in range(8):
            plt.annotate(cs[i], xy=(xs[i], ys[i]), xytext=(xs[i]+0.1, ys[i]+0.1))
        plt.show()


@click.command()
@click.option('--restart/--no-restart', default=False)
def main(restart):
    nav = MainControl()

if __name__ == "__main__":
    main()

