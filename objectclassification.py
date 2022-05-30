#######################################################################
#                               imports                               #
#######################################################################
import TransferLearning as tl
import numpy as np
import cv2

#######################################################################
#                         Function Definition                         #
#######################################################################

def execute(images):
    # Load Machine Learing File
    checkpoint_path = 'vgg16-transfer-4.pth'
    model, optimizer = tl.load_checkpoint(path=checkpoint_path)

    # predict class
    class_predicted = [None]*len(images) 
    for i in range(0,len(images)):
        img, ps, classes = tl.predict(images[i], model, topk=2)
        # TODO: make clean
        if classes[0] == "Muttern":
            class_predicted[i] = 2 
        elif classes[0] == "Schraube":
            class_predicted[i] = 1
    
    return class_predicted
