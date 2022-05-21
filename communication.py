"""
Created on Fr May  20 15:43:00 2022

@author: Stefan Kaufmann
Abschlussprojekt

Communication: Filters the array of object checks for collisions 
"""

import cv2
import numpy as np
from PIL import Image 

plot = False


def creatObj(type, orientation,x,y,angle, orig):
    """Creats a obj, screw nut, lying/standing 
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:           Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image                       

        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no obj            
                
    """
    img = np.zeros((orig[1], orig[0]))


    #  declared the opbject space as an rect
    if type == 1 and orientation == 2:
        width = 50
        height = 150
    else:
        width = 50
        height = 50
    
    #obj = img         # the same size as the origin img
    obj = np.ones((height,width))

    
    # Rotate the img
    w = int(np.sqrt(np.square(height)+np.square(width)))   # new size of the img
    if plot:
        print(w)
    M = np.float32([[1,0,int((w-width)/2)],[0,1,int((w-height)/2)]])                   # translation to the new center
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper verschoben', obj)
   

   
    rows,cols = obj.shape
    M = cv2.getRotationMatrix2D((w/2,w/2), angle, 1)
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper gedreht', obj)
    

    # Draw the areas 
    rows,cols = obj.shape   
    
    for i in range(0,rows):
        for j in range(0,cols):
            if obj[i,j] == 1 or img[int(i+y-rows/2),int(j+x-cols/2)] == 1:        # if obj or img at the location of the obj is 1 --> img at the location is 1
                img[int(i+y-rows/2),int(j+x-cols/2)] = 1                 

        
    
    return img





# ****************************************************************************************************************************************************************
# main

x = 750
y = 500


img = creatObj(1,1,150,150,30, [x,y])

img2 = creatObj(1,2,150,400,20, [x,y])

bitwiseAnd = cv2.bitwise_or(img, img2)

if plot:
    cv2.imshow('Objekt', bitwiseAnd)
cv2.imshow('Objekt', bitwiseAnd)
cv2.waitKey()





     
# %%
