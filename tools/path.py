import os
import os.path as osp
import sys
import time

cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '..'))

from tools.track_client import Hdl
hdl = Hdl()

import numpy as np
import numpy.linalg as npl
import json
os.environ['GREEN_BACKEND'] = 'gevent'
import gevent
from greenthread.monkey import monkey_patch; monkey_patch()

from tools.utils import clockwise_angle, rigid_transform_3D
np.set_printoptions(suppress=True)

class PathManage:
    def __init__(self):
        self.l, self.w, self.Twc = self.load_twc()
        print("length:", self.l, 'width:', self.w, '\n',self.Twc)
        self.T_c_car = None

    def load_twc(self):
        fp = osp.join(cur_d, "../configs/points.json")
        print(fp)
        if osp.exists(fp):
            print('remove')
            os.remove(fp)
        while 1:
            print("wait", time.time())
            if osp.exists(fp):
                break
            time.sleep(0.1)

        with open(fp, 'r') as f:
            box_points = json.load(f)['plane_points']
        
        box_points = np.array(box_points)
        box_points = self.calc_4(box_points)
        
        world_axis = np.array([
                [0, 0, 0],
                [npl.norm(box_points[0]-box_points[1]), 0, 0],
                [0, npl.norm(box_points[0]-box_points[2]), 0],
                [0, 0, npl.norm(box_points[0]-box_points[3])],
        ])              
        print("world_axis", world_axis)
        print("box_points", box_points)
        print(npl.norm(box_points[0] - box_points[1]))
        print(npl.norm(world_axis[0] - world_axis[1]))
        print(npl.norm(box_points[0] - box_points[2]))
        print(npl.norm(world_axis[0] - world_axis[2]))
        print(npl.norm(box_points[0] - box_points[3]))
        print(npl.norm(world_axis[0] - world_axis[3]))
        T = rigid_transform_3D(np.mat(box_points), np.mat(world_axis))
        print("T", T)
        print(box_points)
        box_points = np.hstack([box_points, np.ones((4, 1))])
        print(box_points)
        print(world_axis)
        A2 = (T @ box_points.T).T
        print(A2)
        return npl.norm(box_points[0]-box_points[1]), npl.norm(box_points[0]-box_points[2]), np.array(T)

    def calc_4(self, box_points):
        x = box_points[0]
        o = box_points[1]
        y = box_points[2]
        z = np.cross(x-o, y-o)
        z+=o
        cos_xy = (x-o).dot(y-o) / (np.linalg.norm(x-o) * np.linalg.norm(y-o))
        print(cos_xy)
        assert abs(cos_xy)<0.1, "vector < 0.1"

        cos_xz = (x-o).dot(z-o) / (np.linalg.norm(x-o) * np.linalg.norm(z-o))
        print(cos_xz)
        assert abs(cos_xz)<0.1, "vector < 0.1"

        cos_zy = (z-o).dot(y-o) / (np.linalg.norm(z-o) * np.linalg.norm(y-o))
        print(cos_zy)
        assert abs(cos_zy)<0.1, "vector < 0.1"

        return np.vstack([o, x, y, z])

    def __call__(self, point_list):
        print('pathplan...')
        paths = []
        print("point_list", point_list)
        for i in range(len(point_list)-1):
            start, goal = point_list[i], point_list[i+1]
            paths += self.potential_field_planning(start, goal) 
        print("paths", paths)
        return paths

    def potential_field_planning(self, start, goal):
        [sx, sy], [gx, gy] = start, goal
        sx *= 10
        sy *= 10
        gx *= 10
        gy *= 10
        if sx == gx:
            step = 1 if gy > sy else -1
            return [[sx/10, (sy + i * step)/10] for i in range(int(abs(gy - sy)))] + [[gx/10, gy/10]]

        if sy == gy:
            step = 1 if gx > sx else -1
            return [[(sx + i * step)/10, sy/10] for i in range(int(abs(gx - sx)))] + [[gx/10, gy/10]]
        
        ll = int(max(abs(gx - sx), abs(gy - sy)))
        
        step_x = abs(gx - sx) / ll
        step_y = abs(gy - sy) / ll

        step = step_y if gy > sy else -step_y
        yy = [sy+i*step for i in range(ll + 1)]
        
        step = step_x if gx > sx else -step_x
        xx = [sx+i*step for i in range(ll + 1)]
        print(xx)
        print(yy)
        return [[_x/10, _y/10] for _x, _y in zip(xx[1:], yy[1:])]

    def get_leishen_angle(self, pos_leishen):
        o = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ])
        
        p = pos_leishen.dot(o.T).T[:, :3]
        p_ = p[1] - p[0]

        turn_angle = clockwise_angle((1, 0), p_[:2])
        #return turn_angle
        return (turn_angle+270)%360

    def locate(self):
        while 1:
            with gevent.Timeout(1, False) as to:
                pos_leishen = hdl()
#print(pos_leishen)
                pos_leishen = np.array((self.Twc @ pos_leishen))
                if pos_leishen is None:
                    print('lost')
                    continue
                curr_pose_leishen = {
                    'pos': [pos_leishen[0][3], pos_leishen[1][3]],
                    'angle': self.get_leishen_angle(pos_leishen),
                }
                return curr_pose_leishen


if __name__ == '__main__':
    print("potential_field_planning start")
    
    pp = PathManage()
    '''
    print()
    print(pp([[0, 2.43], [1, 2.53]]))
    print()
    print(pp([[0, 2.43], [0.1, -2.43]]))
    print()
    '''
    while 1:
        print(pp.locate())
    # print(pp([[5.6, -3.9], [6.3, 3.7], [2.7, 9.9], [-3.8, 12.4], [-9.3, 12], [-30, 12]]))
