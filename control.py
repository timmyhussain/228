# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:11:43 2021

@author: user
"""
import airsim
import time
from helper import parse_lidarData
import tensorflow as tf
from airgym.envs.airsim_env import AirSimEnv
# import pptk
# import numpy as np


client = airsim.CarClient()
client.confirmConnection()

client.enableApiControl(True, "Car1")

start = time.time()


# while time.time() - start < 15:
car1_state = client.getCarState("Car1")
print("speed: ", car1_state.speed)


client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])

lidarData = client.getLidarData("LidarSensor1", "Car1")


class LidarCarEnvironment(AirSimEnv):
    def __init__(self):
        pass
    
    

# if (len(lidarData.point_cloud) > 3):
# #    print(type(lidarData.point_cloud))
#     v = pptk.viewer(parse_lidarData(lidarData))   

# print(lidarData.segmentation)
    # time.sleep(0.5)


client.reset()
client.enableApiControl(False)
# time.sleep(5)