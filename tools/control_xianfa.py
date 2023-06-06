import os
import os.path as osp
import sys
from typing import Any

os.environ['GREEN_BACKEND'] = 'gevent'
from greenthread.monkey import monkey_patch; monkey_patch()

from greenthread.green import *
from gevent.threading import Thread

cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '.'))

from mqsrv.client import make_client

import math as M
import random
import time
import toml
import cv2
import numpy as np
import numpy.linalg as npl
import socket
from utils import now,today

from tools.path import PathManage
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

base_dir = "/home/vision/data/lenovo"


class TargetCourse:
    def __init__(self, Lfc):
        self.old_nearest_point_index = None
        self.Lfc = Lfc

    def update(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                if (ind + 1) >= len(self.cx):
                    break  # not exceed goal
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        while self.Lfc > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1
        self.old_nearest_point_index = ind
        return ind

def calc_direc(delta):
    if abs(delta) < 180:
        return delta
    else:
        if delta < -180:
            return 360 + delta
        else:
            return delta - 360

def pure_pursuit_steer_control(state, trajectory, pind):
    ind  = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    logger.debug(f'see: {tx}, {ty}')
    
    logger.debug(f'rear: {state.rear_x}, {state.rear_y}')
    
    alpha = M.degrees(M.atan2(ty - state.rear_y, tx - state.rear_x))
    logger.debug(f'alpha: {alpha}')
    alpha = (alpha + 360) % 360

    logger.debug(f'alpha: {alpha}')
    delta = alpha - M.degrees(state.yaw)
    logger.debug(f'curr_yaw: {M.degrees(state.yaw)}')
    delta = calc_direc(delta)
    logger.debug(f'jiuzheng: {delta}')
    return delta

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.2):
        self.yaw = yaw
        
        self.v = v
        
        self.rear_x, self.rear_y = x, y

        self.L = 0.90

        self.socket_config = self.load_socket_params("configs/hdl.toml")
        self.trigger_server = TriggerSocketServer(0.05, self.socket_config["cam_trigger_ip"], self.socket_config['cam_trigger_port'])
        self.trigger_server.daemon = True
        self.trigger_server.start()

    def load_socket_params(self,path):
        with open(path, 'r') as f:
            config = toml.load(f)
        return config    

 
    def get_cam_pos(self, lidar_x, lidar_y, yaw):
        cam_x, cam_y = lidar_x - (self.L*M.cos(yaw)), lidar_y - (self.L*M.sin(yaw)) 
        return cam_x, cam_y

    def update_detect(self, x, y, yaw, roadway_name=None, roadway_width=None, is_pub=True):
        self.rear_x, self.rear_y = x, y
        self.yaw = yaw
        cur_pos = [self.rear_x,self.rear_y]
        self.trigger_server.update_pos(cur_pos)

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return M.hypot(dx, dy)
    
    
class TriggerSocketServer(Thread):
    def __init__(self,interval, host,port):
        Thread.__init__(self)
        self.interval = interval
        self.buffer_size = 128
        self.ADDR = (host, port)
        self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.pos = None
        
    def run(self):
        while True:
            time.sleep(self.interval)
            if self.pos is None:
                continue
            else:
                logger.error("send pos")
                self.server.sendto(str(self.pos).encode(),self.ADDR)

    def update_pos(self, pos):
        self.pos = pos
                

class HikCapture(Thread):
    def __init__(self, interval, w, l):
        Thread.__init__(self)

        self.hik_capture_pos = []
        self.plane_name_list = ['A', 'B', 'C', 'D']
        self.road_w, self.road_l = w, l
        self.set_hik_capture_info()   
        
        self.interval = interval
        client = make_client()

        self.hik_capture_caller = client.get_caller("hik_camera_rpc_queue")
        # self.hik_light_caller = client.get_caller("light_rpc_queue")
        # self.hik_light_caller.control("set_ctrl_mode", mode=0)
#self.hik_light_caller.control("set_strobe_time", time=990, unit="us")

        # self.hik_light_caller.control("open")
        # self.hik_light_caller.control("close")

        self.hik_capture_caller.open(["camera1","camera2","camera3","camera4","camera5"])
        self.hik_capture_caller.enable_trigger()
        self.hik_capture_caller.start()
        self.tx = None
        logger.error("hik start suc")
        
    def set_hik_capture_info(self):
        for idx, name in enumerate(self.plane_name_list):
            trigger_pos_list = []
            for j in range(4):
                trigger_info = {
                    "plane_name": name,
                    "trigger_name": j+1,
                    "trigger_pos": self.compute_trigger_pos(self.road_w,self.road_l)[idx][j],
                    "acc_control": 0 if idx % 2 == 0 else 1,
                    "is_success": False                    
                    }
                
                trigger_pos_list.append(trigger_info)
        
            self.hik_capture_pos.append(trigger_pos_list)
                
    def compute_trigger_pos(self,w,l):
        re_list = []
        m = 8
        trigger_plane = ["A", "B"]
        for t in range(2):
            for name in trigger_plane:
                pos_list = []

                if t == 0:
                    for i in [2, 4, 6, 8]:
                        pos_list.append([(i * l) / m,None] if name == "A" else [None, (i * w) / m])
                    re_list.append(pos_list)
                else:
                    for i in [8, 6, 4, 2]:
                        pos_list.append([(i * l) / m,None] if name == "A" else [None,(i * w) / m])
                    re_list.append(pos_list)
        return re_list
    
    
    def camera_trigger(self,path):
        os.makedirs(path, exist_ok=True)
        
        # green_spawn(self.hik_light_caller.control, "open")
        # time.sleep(0.2)
        self.hik_capture_caller.trigger(["camera1","camera2","camera3","camera4","camera5"])
        # green_spawn(self.hik_light_caller.control, "close")
        self.hik_capture_caller.save_images(path=path)

    def send_hik_result(self,plane_name,trigger_name,image_path):
        return {
            "location": f"hik_{plane_name + trigger_name}",
            "images": [files for files in os.listdir(image_path)]
        }
    
    def update(self, tx, ty):
        self.tx = tx
        self.ty = ty
        
    def run(self):
        while True:
            time.sleep(self.interval)
            logger.info('hik checking')
            if self.tx is None:
                continue
            cur_pose = [self.tx, self.ty]
            
            names = []
            for idx,name in enumerate(self.hik_capture_pos):
                names += name
            for i, info in enumerate(names):
                is_success = info["is_success"]
                if i == 0:
                    flag = True
                else:
                    flag = names[i-1]['is_success'] is True

                if not is_success and flag:
                    gap = cur_pose[info["acc_control"]] - info["trigger_pos"][info["acc_control"]]
                    if abs(gap) < 0.1:
                        dir_name = info["plane_name"] + str(info["trigger_name"])
                        whole_path = osp.join(base_dir,dir_name)
                        
                        info["is_success"] = True
                        
                        self.camera_trigger(whole_path)

                        # self.control_rpc.send_res(self.send_hik_result(info["plane_name"],str(info["trigger_name"]),whole_path))
                        
                    else:
                        print("not trigger pos") 

class RobotControl(Thread):
    def __init__(self, interval, config_f):
        Thread.__init__(self)
        self.turn_spawn_queue = []
        self.curr_task = {
            'is_pub': False
        }

        self.pp = PathManage()

        self.road_w, self.road_l = self.pp.w, self.pp.l
        
        with open(config_f, 'r') as f:
            self.config = toml.load(f)
        self.interval = interval

        client = make_client()
        self.truck_caller = client.get_caller(routing_key="server_rpc_xianfa")

        self.Lfc = 1.2

        # initial state
        self.state = None

        self.target_course = TargetCourse(Lfc=self.Lfc)

        self.is_updated = False

        self.force_stop = False

        self.vel = 100
        self.vel = self.get_curr_v()

        self.Twc = None
        self.T_c_car = None

        self.hik_capture_thread = HikCapture(0.05, self.road_w, self.road_l)
        self.hik_capture_thread.daemon = True
        self.hik_capture_thread.start()

    def update_path(self, path):
        p = np.array([tmp[:2] for tmp in path])
        
        cx, cy = p[:, 0], p[:, 1]
        self.target_course.update(cx=cx, cy=cy)

    def update_state_pose(self, pose, is_pub=False):
        pos = pose['pos']
        turn_angle = pose['angle']
        
        logger.debug(f'update state. pos: {pos}, angle: {turn_angle}')
        if self.state is None:
            self.heart_x, self.heart_y = pos
            self.state = State(x=pos[0], y=pos[1], yaw=M.radians(turn_angle))
        else:
            self.state.update_detect(x=pos[0], y=pos[1], yaw=M.radians(turn_angle), is_pub=is_pub)

    def update_pose(self, pose, is_pub=True, verbose=False):
        self.update_state_pose(pose, is_pub=is_pub)

        self.target_ind = self.target_course.search_target_index(self.state)
        self.is_updated = True

    def get_curr_v(self, gap=None):
        if gap is None:
            return 140
        if abs(gap) < 10:
            return 160
        else:
            return 140

    def update_task(self, task):
        self.curr_task = task
        self.slow_stop = False
        # update v
        self.vel = self.get_curr_v()
    
    def resume_task(self):
        self.force_stop = False
        self.curr_task = self.curr_task['movement']
        
        if self.curr_task['points'][0] == None:
            self.curr_task['points'][0] = self.pp.Twc[0][3], self.pp.Twc[1][3]
        path = self.pp(self.curr_task['points'])
        self.update_path(path)
        logger.info(path)
        
        self.heart_pre_t = time.time()
        
        while 1:
            if 1:
                logger.debug('get pose')
                pose = self.pp.locate()
                logger.debug('get pose suc')


                __curr_location_0 = pose['pos'][0] - self.curr_task['end'][0]
                __curr_location_1 = pose['pos'][1] - self.curr_task['end'][1]
                if abs(__curr_location_0) + abs(__curr_location_1) < 0.05:
                    self.stop()
                    self.is_updated = False
                    logger.info('to final!')
                    time.sleep(0.01)
                    break

                self.update_pose(pose, verbose=False)
                time.sleep(0.1)
            else:
                logger.info('return in movement')
                break

    def run(self):
        while True:
            time.sleep(self.interval)
            # heart beat
            curr_t = time.time()
            if (curr_t - self.heart_pre_t) > 5:
                self.robot_status = False
            
            if self.is_updated:
                # Calc control input (turn)
                alpha = pure_pursuit_steer_control(
                    self.state, 
                    self.target_course, 
                    self.target_ind, 
                )
                
                self.vel = self.get_curr_v()

                # kill previous turn spawn
                gevent.killall(self.turn_spawn_queue[-2:])
                self.turn_spawn_queue = self.turn_spawn_queue[-2:]

                self.turn_spawn_queue += [green_spawn(self.turn_spawn, alpha)]
                self.is_updated = False
            self.hik_capture_thread.update(self.state.rear_x, self.state.rear_y)

    def move(self, _vel, _angle_left_right, is_rotate=0):
        _angle_left_right *= 2500/(25*2)

        self.truck_caller.xianfa_move(_vel, _angle_left_right, is_rotate)

    def stop(self):
        self.force_stop = True
    
    def turn_spawn(self, gap):
        self.this_isalive = True
        logger.info(f"{'='*5}\t, {gap}, {self.get_curr_v(gap=gap)}\t, \t")
        self.move(self.get_curr_v(gap=gap), gap, 0)
