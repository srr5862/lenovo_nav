import os
import os.path as osp
import sys

cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '..'))
import numpy as np
import numpy.linalg as npl

from scipy.spatial.transform import Rotation as R
import socket
import time
import toml

class FitPlaneSocket:
    def __init__(self):
        self.conf = toml.load(osp.join(cur_d, "../configs/hdl.toml"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (self.conf['plane_ip'], self.conf['plane_port'])

    def sendandreceive(self):
        self.client.sendto('send'.encode('utf-8'), self.addr)
        data, addrs = self.client.recvfrom(1024)
        return data

    def __call__(self):
        data = self.sendandreceive()

        points = data.decode()
        x1, y1, z1, x0, y0, z0, x2, y2, z2 = list(map(float, points.splie(",")))
        return np.array([
            [x1, y1, z1],
            [x0, y0, z0],
            [x2, y2, z2],
        ])

class Hdl:
    def __init__(self):
        self.conf = toml.load(osp.join(cur_d, "../configs/hdl.toml"))
        self.trans_matrix = None

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (self.conf['ip'], self.conf['port'])

    def sendandreceive(self):
        # print('send')
        self.client.sendto('send'.encode('utf-8'), self.addr)
        data, addrs = self.client.recvfrom(1024)
        return data

    def __call__(self):
        data = self.sendandreceive()

        pose = data.decode()
        if pose == 'failed':
            return
        Pose = pose.splitlines()
        p_x = float(Pose[1].split(":")[1])
        p_y = float(Pose[2].split(":")[1])
        p_z = float(Pose[3].split(":")[1])
        o_x = float(Pose[5].split(":")[1])
        o_y = float(Pose[6].split(":")[1])
        o_z = float(Pose[7].split(":")[1])
        o_w = float(Pose[8].split(":")[1])
        Rq = [o_x, o_y, o_z, o_w]
        Rm = R.from_quat(Rq)
        rotation_matrix = Rm.as_matrix()

        T = np.eye(4)
        T[:3, :3] = rotation_matrix

        T[:3, [3]] = np.array([[p_x], [p_y],  [p_z]])

        return T

if __name__ == '__main__':
    hdl =  Hdl()
    while 1:
        print("trans_matrix:\n", hdl())
        time.sleep(1)
