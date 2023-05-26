import os
import time

os.environ["GREEN_BACKEND"] = "gevent"
from greenthread.monkey import monkey_patch; monkey_patch()
from mqsrv.client import make_client

if __name__ == "__main__":
    client = make_client()
    caller = client.get_caller("light_rpc_queue")
    
    """光源初始化设置"""
    # 设置模式
    exc, res = caller.control("set_ctrl_mode", mode=0)
    print(exc, res)
    # 设置亮度为56
    exc, res = caller.control("set_brightness", brightness=200)
    print(exc, res)
    # 读取亮度
    exc, res = caller.control("get_brightness")
    print(exc, res)
    # # 设置频闪时间为990
    # exc, res = caller.control("set_strobe_time", time=990, unit="us")
    # print(exc, res)
    # for i in range(10):
    #     # 触发频闪
    #     exc, res = caller.control("trigger_strobe")
    #     print(exc, res)
    #     time.sleep(0.1)
    
    # 打开
    exc, res = caller.control("open")
    print(exc, res)
    
    # TODO: 控制相机拍照
    
    # 关闭
    exc, res = caller.control("close")
    #print(exc, res)
    
