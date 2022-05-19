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

bv.execute(img)

oc.execute()

tcp.execute()
