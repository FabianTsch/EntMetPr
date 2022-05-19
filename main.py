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

camera.execute()
bv.execute()
oc.execute()
tcp.execute()

