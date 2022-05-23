"""
Created on Fr May  20 15:43:00 2022

@author: Stefan Kaufmann
Abschlussprojekt

Communication: Filters the array of object checks for collisions 
"""

# %%
import cv2
import numpy as np
from PIL import Image 

plot = True
workspace = 100 # mm  work space is 100 mm bigger than the table x*y 


def creatObj(type, orientation,x,y,angle, orig, tool, workspace):
    """Creats a obj, screw nut, lying/standing 
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image   
        tool:          if we want to grab with tool   # true = tool, false = no tool   
        workspace:     work space is the overlay of the table, we need it for the implementation 
        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no obj            
                
    """
    
    # Creat Workspace and Table
    
    img = np.zeros((orig[1]+2*workspace, orig[0]+2*workspace))
    if plot:
        img[workspace,:] = 1
        img[workspace+orig[1],:] = 1
        img[:,workspace] = 1
        img[:,workspace+orig[0]] = 1
        
    # Security Space
    s = 2 # [mm]

    #  declared the opbject space as an rect
    if tool:
        if type == 1 and orientation == 2:  # tool for screw 
            width = 42+s
            height = 20+s
           
                        
        else:   # tool for nuts
            width = 42+s
            height = 20+s     

    
    elif type == 1 and orientation == 2:   # screws
        width = 20
        height = 50
    else:                                  # nuts
        width = 20
        height = 20
    
    obj = np.ones((height,width))
    if tool:
        obj[:,7:width-7] = 0

        
    
    # Rotate the img
    w = int(np.sqrt(np.square(height)+np.square(width)))                        # new size of the img
    if plot:
        print(w)
        print(obj.shape)
        print(img.shape)
    M = np.float32([[1,0,int((w-width)/2)],[0,1,int((w-height)/2)]])            # translation to the new center
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper verschoben', obj)
      
    rows,cols = obj.shape
    M = cv2.getRotationMatrix2D((w/2,w/2), angle, 1)
    obj = cv2.warpAffine(obj, M, (w, w))
    if plot:
        cv2.imshow('Körper gedreht', obj)
    

    # Draw the obj in the work space 
    rows,cols = obj.shape   
    
    for i in range(0,rows):
        for j in range(0,cols):
            if obj[i,j] == 1 or img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] == 1:        # if obj or img at the location of the obj is 1 --> img at the location is 1
                img[int(i+y-rows/2+workspace),int(j+x-cols/2+workspace)] = 1                 

    if plot:
         return img  
         print(orig)  
    return img[workspace:orig[1]+workspace, workspace:orig[0]+workspace]


def pick(type, orientation,x,y,angle,orig,img, workspace):
    """ Checks if the obj can be picked
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        img:           Binary image with all remaining Objects 1 = opject  
        tool:          if we want to grab with tool   # true = tool, false = no tool                    

        Returns
        --------
        posible:       binary image with 1 = obj ,  0 = no obj            
                
    """
    # creats a tool img
    y_img,x_img = img.shape
    if plot:
        y_img -= workspace*2
        x_img -= workspace*2
    img_tool = creatObj(type, orientation,x,y,angle, (x_img,y_img), True, workspace)

    img_bitwise_and = cv2.bitwise_and(img, img_tool)

    if img_bitwise_and.any():
        possible = False
    else:
        possible = True

    if plot:
        print(' ISt das Objekt audgreifbar?', possible)

    return possible


    

# %%

# ****************************************************************************************************************************************************************
# main
# ****************************************************************************************************************************************************************

# image size
x = 750          
y = 500


img = creatObj(1,1,25,25,20, [x,y],False, workspace)

img3 = creatObj(1,2,200,200,80, [x,y],False, workspace)
#img4 = creatObj(1,2,200,200,80, [x,y],True)

bitwiseOr= img+img3  #+img4 

if plot:
    cv2.imshow('Objekt', bitwiseOr)


offset = 0
pick(1,1,25+offset,25,20,[x,y],bitwiseOr, workspace)


cv2.imshow('Objekt', bitwiseOr+creatObj(1,1,25+offset,25,20, [x,y],True, workspace))

cv2.waitKey()

