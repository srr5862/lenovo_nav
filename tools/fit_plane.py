#! /usr/bin/env/python3
import os
import os.path as osp
import sys

cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '..'))

import json
import yaml
import numpy as np
import numpy.linalg as npl
import open3d as o3d
import math
import time
import toml
import socket
import rospy
import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters


class FitPlane:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.config = self.load_params(path=osp.join(cur_d, "../configs/plane.yaml"))

        self.imgsub = message_filters.Subscriber("/rgb/image_raw",Image)
        self.depsub = message_filters.Subscriber("/depth_to_rgb/image_raw",Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.imgsub,self.depsub],100000,0.01,allow_headerless=True)
        self.ts.registerCallback(self.imgdepCb)

        self.K = self.config["camera_K"]
        self.sensor_height = float(self.config["sensor_height"])
        self.base_height = float(self.config["base_height"])
        self.box_min_height = float(self.config["box_min_height"])

        self.max_depth = self.config["max_depth"] * 1000
        self.verbose = self.config["verbose"]

        self.image_pubulish=rospy.Publisher('/detect', Image,queue_size=1)
    

    def publish_image(self, imgdata):
        image_temp=Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'map'
        image_temp.height=imgdata.shape[0]
        image_temp.width=imgdata.shape[1]
        image_temp.encoding='rgb8'
        image_temp.data=np.array(imgdata).tostring()
        image_temp.header=header
        image_temp.step=image_temp.width * 3
        self.image_pubulish.publish(image_temp)

    def load_params(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def imgdepCb(self,img_msg,dep_msg):
        image = self.cv_bridge.imgmsg_to_cv2(img_msg,desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(dep_msg,desired_encoding="16UC1")
        depth[:, 0:700] = 0
        depth[depth > self.max_depth] = 0
        
        rois_p = []
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if depth[j][i] != 0:
                    rois_p.append(0.001 * depth[j][i]*np.array([i, j, 1]))

        rois_p = np.vstack(rois_p)

        rois_p3 = (npl.inv(self.K) @ rois_p.T).T
        
        rois_p3 = rois_p3[np.where(rois_p3[:, 0] < self.sensor_height - self.base_height)]
        rois_p3 = rois_p3[np.where(rois_p3[:, 0] > self.box_min_height)]

        sorted_rois_p3 = self.find_depth(rois_p3, 2)
 
        side = sorted_rois_p3[:10000]
        
        plane_mean_y = np.mean(side, axis=0)[1]

        plane0 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] > plane_mean_y)]
        plane1 = sorted_rois_p3[np.where(sorted_rois_p3[:, 1] <= plane_mean_y)]
        
        means_0 = []
        means_1 = []
        for a, k in enumerate(range(int(self.box_min_height*10), int(10*(self.sensor_height-self.base_height)))):
            i = k/10
            plane0_ = plane0[np.where((plane0[:, 0] > i) & (plane0[:, 0] < i+0.1))]
            mix0 = self.find_depth(plane0_, 1, descending=1)[:10]
            means_0.append(np.mean(mix0, axis=0))

            plane1_ = plane1[np.where((plane1[:, 0] > i) & (plane1[:, 0] < i+0.1))]
            mix1 = self.find_depth(plane1_, 1)[:10]
            means_1.append(np.mean(mix1, axis=0))
        
        means_0_yz = np.mean(np.vstack(means_0), axis=0)
        means_1_yz = np.mean(np.vstack(means_1), axis=0)

        means_side = np.mean(side, axis=0)
        re = np.vstack([means_1_yz, means_side, means_0_yz])
        re[:, 0] = 1
        self.write_to_json(re.tolist())
        
    def find_depth(self, ps, axis, descending=0):
        if descending==0:
            return ps[ps[:, axis].argsort()]
        else:
            return ps[ps[:, axis].argsort()][::-1]

    def write_to_json(self, ar):
        dic = {"plane_points": ar}
        with open(osp.join(cur_d, "../configs/points.json"), 'w') as f:
            json.dump(dic, f, indent=2)

if __name__ == '__main__':
    rospy.init_node("depth_op",anonymous=True)

    fit = FitPlane()
    
    rospy.spin()
