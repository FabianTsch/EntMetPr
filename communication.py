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



def createObj(type, orientation,x,y,angle, orig, tool=False, workspace=100):
    """Creats a obj, screw nut, lying/standing 
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image   
        tool:          if we want to grab with tool   # true = tool, false = no tool    default False 
        workspace:     work space is the overlay of the table, we need it for the implementation default 100 mm
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
    s = 5 # [mm]

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
        obj[:,2*s:width-2*s] = 0

        
    
    # Rotate the img
    w = int(np.sqrt(np.square(height)+np.square(width)))                        # new size of the img

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
          
    return img[workspace:orig[1]+workspace, workspace:orig[0]+workspace]

def createImg(type, orientation,x,y,angle, orig,tool = False, workspace=100):
    """Creats a obj, screw nut, lying/standing, can handle scalars and arrays
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        workspace:     work space is the overlay of the table, we need it for the implementation,  default 100 mm

        Returns
        --------
        img:           binary image with 1 = obj ,  0 = no obj            
                
    """
    if plot:
        img = np.zeros((orig[1]+2*workspace, orig[0]+2*workspace))
    else:
        img = np.zeros((orig[1], orig[0]))

    if np.isscalar(type):
        img = createObj(type,orientation,x,y,angle, orig, tool, workspace)
    else:
        for i in range(0,np.size(type)):
            img += createObj(type[i],orientation[i],x[i],y[i],angle[i], orig, tool, workspace)           
    
    return img

def pick(type, orientation,x,y,angle,img, workspace=100):
    """ Checks if the obj can be picked
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        img:           Binary image with all remaining Objects 1 = object  
        tool:          if we want to grab with tool   # true = tool, false = no tool                    

        Returns
        --------
        posible:       True if the obj can be picked           
                
    """
    # creats a tool img
    y_img,x_img = img.shape
    
    if plot:
        y_img -= workspace*2
        x_img -= workspace*2
    img_tool = createObj(type, orientation,x,y,angle, (x_img,y_img), True)

    img_bitwise_and = cv2.bitwise_and(img, img_tool)
    if img_bitwise_and.any():
        possible = False
    else:
        possible = True

    if plot:
        img_bitwise_and = img_bitwise_and[(workspace+1):orig[1]+workspace, (workspace+1):orig[0]+workspace]  # shrinks the image to the area of ​​interest
        #cv2.imshow('bitwise_and', img_bitwise_and)
        #cv2.waitKey()
        if img_bitwise_and.any():
             possible = False
        else:
            possible = True
        print(' ISt das Objekt audgreifbar?', possible)

    return possible

def checkPick(type, orientation,x,y,angle,orig,img = 0):
    """ Checks if the obj can be picked works with scalars and arrays
        Params
         --------
        type:          screw = 1, nut = 2, seperate = 3, Done = 0, GoToHome = 10, 
        orientation:   standing = 1, lying = 2,  perhabs more info    
        x:             Position from origin in mm - x-Position
        y:             Position from origin in mm - y-Position
        angle:         Orientation of the main axis in grad  
        orig:          Original dimensions [x,y] of the image 
        img:           Binary image with all remaining Objects 1 = object, by default creats his own binary
        tool:          if we want to grab with tool   # true = tool, false = no tool                    

        Returns
        --------
        j:             Array of indixes of the obj which can be picked in the correct order  
        posible:       True if the obj can be picked       
                
    """
    # Create reconstructed image
    if not img < 0:
        img = createImg(type, orientation,x,y,angle, orig)       
       
    if plot:
        cv2.imshow('Das Orginal', img) 
    # Helpful variables
    possible = []
    j = []                  # Array of indixes of the obj which can be picked in the correct order

    if np.isscalar(type): 
        possible = pick(type, orientation,x,y,angle,img)
        return possible
    
    
    else:
        for count in range(0,np.size(type)):  # do the check as often as there are elements in obj  
            if plot:
                print ('#################  Durlauf Nr.: ', count)            

            for i in range(0,np.size(type)):  # first loop over all objects  
                if count%2 == 0:
                    i = np.size(type)-i -1
                
                if plot:
                    print('Der Sublauf: ', i)   
                    print('##### j ist: ', j)          
                if not i in j and pick(type[i], orientation[i],x[i],y[i],angle[i],img):   # Controls if any i ist part of j and if obj[i] is pickable             
                                   
                    j = np.append(j,i)   
                    img -=createImg(type[i], orientation[i],x[i],y[i],angle[i],orig) # subtracts the object from the image  
                    if plot:                                             
                        cv2.imshow(str(i), img)  
                        cv2.waitKey()            
                    
            

    return j
        


# %%

# ****************************************************************************************************************************************************************
# main
# ****************************************************************************************************************************************************************

# image size
x = 750          
y = 500

type = [1,1,1,1]
orientation = [1,2, 1, 2]
x_obj = [25,200, 150, 180]
y_obj = [25,200, 250, 200]
angle = [20,80, 0, 0]
orig = [x,y]

img = createImg(type, orientation,x_obj,y_obj,angle, orig)
img_tool =  createImg(type, orientation,x_obj,y_obj,angle, orig, True )


if plot:   
    # Scalar
    print(checkPick(1, 1,25,25,20,orig))
    # Array
    print('Folgende Indizies sind aufgreifbar: ', checkPick(type, orientation,x_obj,y_obj,angle,orig))
    
 
print(checkPick(type, orientation,x_obj,y_obj,angle,orig))
cv2.imshow('Objekt', img+img_tool)

cv2.waitKey()


