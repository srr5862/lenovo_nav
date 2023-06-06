from datetime import datetime
import numpy as np
import numpy.linalg as npl
import time



def rigid_transform_3D(pc1, pc2):
    assert len(pc1) == len(pc2)
    N = pc1.shape[0]
    mu_pc1 = np.mean(pc1, axis=0)
    mu_pc2 = np.mean(pc2, axis=0)
    AA = pc1 - np.tile(mu_pc1, (N, 1))
    BB = pc2 - np.tile(mu_pc2, (N, 1))
    H = np.transpose(AA).dot( BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    t = -R * mu_pc1.T + mu_pc2.T
    T = np.hstack([R, t])
    T = np.vstack([T, np.array([0, 0, 0, 1])])
    return T

def clockwise_angle(v1, v2):
    v1 = v1 / npl.norm(v1)
    v2 = v2 / npl.norm(v2)
    #y1, x1 = v1
    #y2, x2 = v2
    x1, y1 = v1
    x2, y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    theta = theta if theta>0 else 2*np.pi+theta
    return theta*180/np.pi
def now():
    return datetime.now().strftime("%y%m%dT%H%M%SS%f")[:-3]

def today():
    return time.strftime("%Y%m%d",time.localtime())

