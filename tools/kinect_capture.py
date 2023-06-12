#! /usr/bin/env/python3
import os
import os.path as osp
import sys

cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '..'))

import time
import queue
import json
import yaml
import numpy as np
import toml
import socket
import cv2
import rospy
import message_filters

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from cv_bridge import CvBridge
from threading import Thread
from utils import now,today
from loguru import logger


base_dir = "/home/vision/data/lenovo"


class TriggerSocketClient:
    def __init__(self):
        self.config_socket = toml.load(osp.join(cur_d,"../configs/hdl.toml"))
        self.client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addr = (self.config_socket['cam_trigger_ip'], self.config_socket['cam_trigger_port'])
        self.client.bind(self.addr)

    def receive(self):
        data, addrs = self.client.recvfrom(128)
        cur_pos = data.decode()
        return cur_pos

    def sendandreceive(self):
        self.client.sendto('send'.encode('utf-8'), self.addr)
        data, addrs = self.client.recvfrom(1024)
        return data

class CaptureThread(Thread):
    def __init__(self,interval):
        Thread.__init__(self)
        self.queue = queue.Queue()
        self.interval = interval
        self.socket_client = TriggerSocketClient()
    
    def put(self,x):
        self.queue.put(x)

    def get(self):
        return self.queue.get()

    def run(self):
        while True:
            time.sleep(self.interval)
            cur_pose = self.socket_client.receive()
            self.put(json.loads(cur_pose))

capture_thread = CaptureThread(0.05)
capture_thread.daemon = True
capture_thread.start()


class CheckPosition:
    def __init__(self):
        self.cv_bridge = CvBridge()

        
        self.trigger_plane_name_list = ["A","B","C","D"]
        self.trigger_plane_name = None
        
        self.kinect_trigger_info = []
        self.capture_dist = 0.05
        self.mean_val = 160
        self.l, self.w = self.load_wl()

        self.set_kinect_trigger_info()
        
        self.imgsub = message_filters.Subscriber("/rgb/image_raw",Image)
        self.depsub = message_filters.Subscriber("/depth_to_rgb/image_raw",Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.imgsub,self.depsub],100000,0.01,allow_headerless=True)
        self.ts.registerCallback(self.capturecb)
        
        self.date = None
        self.timestamp = None
        self.whole_dir = None
        
        self.stamp_pub = rospy.Publisher("/timestamp",String,queue_size=1)
        
        
        
    def set_date(self):
        self.timestamp = now()
        self.date = self.timestamp[:6] 
        self.whole_dir = osp.join(osp.join(base_dir,self.date),self.timestamp)

        check_dir = osp.join(osp.join(osp.join(base_dir,self.date),"check"),self.timestamp)
        
        while not osp.exists(check_dir):
            self.stamp_pub.publish(self.timestamp)
            time.sleep(0.2)
            
               
        

    def load_wl(self):
        fp = osp.join(cur_d,"../configs/wl.json")
        if osp.exists(fp):
            os.remove(fp)
            
        while 1:
            print("wait",time.time())
            if osp.exists(fp):
                break
            time.sleep(0.1)
            
        with open(fp,'r') as f:
            data = json.load(f)
            l, w = data["l"], data["w"]
        
        return l, w
        


    def set_kinect_trigger_info(self):
        for idx,name in enumerate(self.trigger_plane_name_list):
            trigger_pos_list = []
            for j in range(4):
                trigger_info = {
                    "trigger_plane": name,
                    "trigger_pos": self.compute_trigger_pos(self.w,self.l)[idx][j],
                    "trigger_name": j+1,
                    "acc_control": 0 if idx % 2 == 0 else 1,
                    "is_success": False
                }
                trigger_pos_list.append(trigger_info)
            self.kinect_trigger_info.append(trigger_pos_list)

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

    def trans_curframe_to_img(self,image,ori_depth,timestamp):
        whole_path = osp.join(self.whole_dir,self.trigger_plane_name)

        image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE) 
        ori_depth = cv2.rotate(ori_depth,cv2.ROTATE_90_CLOCKWISE) 
        
        os.makedirs(whole_path,exist_ok=True)
        
        cv2.imwrite(f"{whole_path}/{timestamp}_color.jpg",image)
        cv2.imwrite(f"{whole_path}/{timestamp}_rdepth.png",ori_depth)
    
    def capturecb(self,img,re_depth):
        names = []
        for i, name in enumerate(self.kinect_trigger_info):
            names += name 
        for idx, pos_info in enumerate(names):
            is_success = pos_info["is_success"]
            if idx == 0 :
                flag = True
            else:
                flag = names[idx - 1]["is_success" ] is True

            acc_control = int(pos_info["acc_control"])
                
            if not is_success and flag and capture_thread.get() is not None:
                cur_pose = capture_thread.get()
                
                gap = abs(cur_pose[acc_control] - pos_info["trigger_pos"][acc_control])
                print("gap:",gap)
                
                if gap < self.capture_dist:
                    self.trigger_plane_name = pos_info["trigger_plane"] + str(pos_info["trigger_name"])
                    
                    image = self.cv_bridge.imgmsg_to_cv2(img,desired_encoding="bgr8")
                    re_depth = self.cv_bridge.imgmsg_to_cv2(re_depth,desired_encoding="16UC1")        
                    
                    image_mean = cv2.mean(image)[0]
                    if image_mean > self.mean_val:
                        logger.error("overexposure!!!")
                        break
                    else:
                        self.trans_curframe_to_img(image,re_depth,now())
                        pos_info["is_success"] = True
                    
                else:
                    ...

if __name__ == '__main__':
    rospy.init_node("check_position",anonymous=True)
    
    check = CheckPosition()

    rospy.spin()
