#######################################################################
#                               Imports                               #
#######################################################################

# import cv2
# import numpy as np
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
obj_class = [1,2]
orientation = [2,1]
x = [380,70]
y = [263,30]
angle = [90,90]
tcp.tcp_communication(obj_class, orientation, x, y, angle)
