#######################################################################
#                               Imports                               #
#######################################################################

# import cv2
import numpy as np
# import matplotlib.pyplot as plt

# import camera
# import Bildverarbeitung as bv
# import objectclassification as oc
import tcpClient as tcp
#import communication as cm 


#######################################################################
#                            Main Programm                            #
#######################################################################

print("Start Loop")
# TODO: path as @para is only a temporary solution as long as the
#       the camera function isn't implemented.
#img = camera.execute("Kamerabilder/TX2_SM_kontakt.png")

#img = bv.homography(img)

#img_array, x, y, angle, orientation = bv.object_detection(img)
# TODO: break when liste is empty

#obj_class = oc.execute(img_array)

#type, orientation, x, y, angle, error = cm.execute(obj_class,orientation, x, y, angle)
obj_class = [2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 3, 2]
orientation = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1]
x = [345, 256, 144,  71, 449, 163, 347,  84, 263, 406, 251, 430, 138, 324, 339, 498, 495]
y = [445, 423, 415, 410, 370, 356, 337, 333, 246, 235, 233, 228, 215, 184, 172, 140, 121]
angle = [350.4, 348.4, 354.3, 323.8, 224.,  335.2, 182.2 ,333.4,523.6 ,510.9 ,530.4 ,515.1, 331.2, 533.7, 182.8, 190.3 ,188.2]


tcp.tcp_communication(obj_class, orientation, x, y, angle)





