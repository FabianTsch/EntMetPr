#######################################################################
#                               Imports                               #
#######################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

import camera
import Bildverarbeitung as bv
import objectclassification as oc
import tcp_socket as tcp


#######################################################################
#                            Main Programm                            #
#######################################################################

# TODO: path as @para is only a temporary solution as long as the
#       the camera function isn't implemented.
img = camera.execute("Kamerabilder/TX2_SM_kontakt.png")

img = bv.homography(img)

img_array, obj_position, obj_orientation = bv.object_detection(img)

obj_class = oc.execute(img_array)

tcp.execute()
