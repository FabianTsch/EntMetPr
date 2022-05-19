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

    path = "Kamerabilder/TX2_SM_kontakt.png"

    # reading the image
    img = cv2.imread(path)

    b = bv.Bilder(img)
    b.homography()
    
    for i in range(len(b.j)):
        image = cv2.circle(img, b.mc[i, :], 50, (255, 0, 0), 5)
        cv2.imshow("img", image)

   

    b.findObject()
    b.display_img()

    
    # wait for a key to be pressed to exit
    print("Press any key")
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
