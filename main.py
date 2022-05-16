#######################################################################
#                               Imports                               #
#######################################################################

import Bildverarbeitung as bv
import cv2
import numpy as np
import matplotlib.pyplot as plt


#######################################################################
#                            Main Programm                            #
#######################################################################

if __name__ == "__main__":
    cv2.destroyAllWindows()

    path = 'Kamerabilder/TX2_SM_kontakt.png'

    # reading the image
    img = cv2.imread(path)

    b = bv.Bilder(img)
    b.ausrichten()

    for i in range(len(b.j)):
        image = cv2.circle(img, b.mc[i, :], 50, (255, 0, 0), 5)
        cv2.imshow('img', image)

    cv2.imshow('im_dst', b.im_dst)

    b.findObjekte()

    obj = b.obj
    a = 10
    for i in range(len(b.j)):
        #image = cv2.circle(b.mask, b.mc[i,:], 10, (255, 0, 0), 7)

        cv2.rectangle(b.im_dst, (obj[i, 0]-a, obj[i, 1]-a),
                      (obj[i, 0]+obj[i, 2]+a, a+obj[i, 1]+obj[i, 3]), (0, 255, 0), 1)
        cv2.imshow('im_dst', b.im_dst)

    # wait for a key to be pressed to exit
    print('Press any key')
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
