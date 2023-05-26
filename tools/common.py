import os.path as osp

import cv2
import numpy as np
import open3d as o3d
import math as M

cur_d = osp.join(osp.dirname(__file__),"../")

def depth_to_8bit(dep):
    dep -= dep.min()
    dep = dep / (dep.max() - dep.min())
    dep *= 255
    dep_8 = dep.astype(np.uint8)
    return dep_8
    
def gen_mask(image):
    ret,binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
    return binary
    
def trans_to_camera(K,pts,d,factor):
    z = d / 1000
    x = (pts[0] - K[0][2]) * z / K[0][0]
    y = (pts[1] - K[1][2]) * z / K[1][1]
    return [x,y,z]
    
def compute_normal_vector(self,vec):
    u = (1, 0, vec[2])
    v = (0, 1, vec[2])
    n = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
    return n

def comput_edge(points_list,point):
        x,y,z = point
        re_list = []
        for i in range(len(points_list)):
            if points_list[i][1] - y < 0.01:
                re_list.append(points_list[i])

def compute_plane(points3d,is_success,base_point):
    pcd = o3d.geometry.PointCloud()
    pcd_lines = o3d.geometry.PointCloud()
    points3d = np.array(points3d).reshape(-1,3)
    
    line_points = points3d[np.where(np.abs(points3d[:,0] -  base_point[0] > 0.2 ))]
    sorted_points = points3d[points3d[:,1].argsort()]
    min_point,max_point = sorted_points[0],sorted_points[-1]
    end_list = np.vstack([min_point,base_point,max_point])
    
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd_lines.points = o3d.utility.Vector3dVector(np.vstack([end_list]))
#pcd_lines.points = o3d.utility.Vector3dVector(np.vstack([line_points,line_points1]))
    #o3d.io.write_point_cloud(osp.join(cur_d,"debug","plane.pcd"),pcd)
    #o3d.io.write_point_cloud(osp.join(cur_d,"debug","line.pcd"),pcd_lines)
    # o3d.visualization.draw_geometries([pcd,pcd_lines])
    return end_list
    
